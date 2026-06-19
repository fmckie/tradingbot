"""Claude AI trading agent using Anthropic API."""
import os
import json
from datetime import datetime
from typing import Any, cast
import anthropic

from database.json_utils import safe_json_dumps

from .base_agent import (
    BaseTradingAgent,
    TradingDecision,
    MarketContext,
    ActionType,
    StrategyType,
)
from tools.market_tools import MARKET_TOOLS_SCHEMA
from tools.trading_tools import TRADING_TOOLS_SCHEMA
from tools.analysis_tools import ANALYSIS_TOOLS_SCHEMA
from tools.news_tools import NEWS_TOOLS_SCHEMA
from config.settings import LEARNING_ENABLED


CLAUDE_SYSTEM_PROMPT = """You are an autonomous AI trading agent managing a $100,000 paper trading portfolio.
You can ONLY trade GOOGL and TSLA stocks on the NASDAQ.

YOUR GOAL: Maximize risk-adjusted returns over a 1-month competition period.

WHAT YOU RECEIVE EACH HOUR:
- Real-time price data for GOOGL and TSLA
- Technical indicators (VWAP, RSI, MACD, Bollinger Bands, ATR, moving averages)
- NEWS SENTIMENT: Recent headlines and sentiment analysis for each stock
- Your current positions and unrealized P&L
- Your account balance and buying power
- Recent trade history

NEWS ANALYSIS:
You have access to news tools that provide sentiment-scored headlines. News sentiment can be:
- Bullish: Positive news (upgrades, beat earnings, partnerships, etc.)
- Bearish: Negative news (downgrades, misses, lawsuits, recalls, etc.)
- Neutral: No clear sentiment signal

Use news sentiment as a CONFIRMATION or CAUTION signal alongside technical analysis.
Strong technical setups with confirming news sentiment are higher-probability trades.
Be cautious when news sentiment contradicts your technical analysis.

WHAT YOU MUST DECIDE:
1. Your trading STRATEGY for this hour:
   - Momentum: Trade in direction of strong price movement
   - Mean Reversion: Bet on price returning to average after extremes
   - Trend Following: Follow established trends with moving averages
   - Breakout: Trade price breaks through support/resistance
   - Range Trading: Buy support, sell resistance in ranging markets
   - Defensive: Hold cash when conditions are unclear

2. Your ACTION: BUY, SELL, HOLD, or CLOSE a position

3. If trading:
   - Which stock (GOOGL or TSLA)
   - Position size (number of shares)
   - Stop-loss price (REQUIRED)
   - Optional take-profit price

HARD RULES (enforced by system - you cannot override):
- Max 2% account risk per trade
- Max 50% capital deployed at any time
- Max 2 open positions (1 per symbol)
- Stop-loss REQUIRED on all trades
- Daily loss limit: 5%
- NO TRADING in the first 15 minutes after market open (9:30-9:45 AM ET)
- NO TRADING in the last 15 minutes before market close (3:45-4:00 PM ET)
- Trading window: 9:45 AM - 3:45 PM ET only

DECISION FRAMEWORK:
1. Analyze current market conditions (use tools to gather data)
2. Identify which strategy fits current conditions
3. Determine if there's a high-probability setup
4. If yes, calculate position size within risk limits
5. Place order with stop-loss
6. If no clear setup, HOLD (being flat is a valid position)

You are competing against another AI (Grok). Make decisions that maximize risk-adjusted returns.
Quality of decisions matters more than quantity of trades.

Always explain your reasoning clearly. Your decisions and explanations will be logged for analysis."""


class ClaudeAgent(BaseTradingAgent):
    """Trading agent powered by Claude (Anthropic)."""

    def __init__(self, tools: dict):
        super().__init__("claude", tools)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-6"  # Claude Sonnet 4.6
        self.current_strategy_explanation = ""

        # Combine all tool schemas
        self.tool_schemas = MARKET_TOOLS_SCHEMA + TRADING_TOOLS_SCHEMA + ANALYSIS_TOOLS_SCHEMA + NEWS_TOOLS_SCHEMA

    async def analyze_and_decide(self, context: MarketContext) -> TradingDecision:
        """
        Use Claude to analyze market and make trading decision.

        Claude receives the market context and can call tools for additional
        analysis before making its decision.
        """
        # Recall past learnings before making decision
        learnings_context = await self.recall_learnings(context)
        recent_outcomes = await self.get_recent_outcomes(limit=5)

        # Format context as user message
        context_message = self._format_context(context)

        # Add learnings to the context if available
        if learnings_context:
            context_message += learnings_context
        if recent_outcomes:
            context_message += recent_outcomes

        messages: list[dict[str, Any]] = [{"role": "user", "content": context_message}]

        tool_calls_made: list[dict[str, Any]] = []
        max_iterations = 5  # Limit tool call iterations

        for _ in range(max_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=CLAUDE_SYSTEM_PROMPT,
                tools=cast(Any, self.tool_schemas),
                messages=cast(Any, messages),
            )

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Process tool calls
                tool_results: list[dict[str, Any]] = []
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = cast(dict[str, Any], content_block.input)
                        tool_id = content_block.id

                        # Execute the tool
                        result = self.execute_tool(tool_name, tool_input)
                        tool_calls_made.append(
                            {"tool": tool_name, "input": tool_input, "result": result}
                        )

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": safe_json_dumps(result),
                            }
                        )

                # Add assistant response and tool results to messages
                messages.append({"role": "assistant", "content": cast(Any, response.content)})
                messages.append({"role": "user", "content": tool_results})

            else:
                # Claude has finished reasoning, extract decision
                break

        # Parse Claude's final response into a TradingDecision
        decision = self._parse_decision(response, context.timestamp, tool_calls_made)
        self.record_decision(decision)

        return decision

    def _format_context(self, context: MarketContext) -> str:
        """Format market context as a clear message for Claude."""
        # Format positions with enriched data
        positions_str = "None"
        if context.positions:
            position_lines = []
            for p in context.positions:
                lines = [
                    f"  {p['symbol']}: {p['quantity']} shares @ ${p['avg_entry_price']:.2f}"
                ]

                # P&L line (total + today)
                pnl_line = f"    P&L: ${p['unrealized_pnl']:+.2f} ({p['unrealized_pnl_percent']:+.1f}%)"
                if p.get('intraday_pnl') is not None:
                    pnl_line += f" | Today: ${p['intraday_pnl']:+.2f} ({p.get('intraday_pnl_percent', 0):+.1f}%)"
                lines.append(pnl_line)

                # Time context
                if p.get('holding_duration'):
                    time_str = f"    Holding: {p['holding_duration']}"
                    if p.get('entry_time_str') and p['entry_time_str'] != 'Unknown':
                        time_str += f" since {p['entry_time_str']}"
                    lines.append(time_str)

                # Stop/TP with distance
                risk_parts = []
                if p.get('stop_loss'):
                    stop_dist = p.get('stop_distance_pct', 0)
                    risk_parts.append(f"Stop: ${p['stop_loss']:.2f} ({stop_dist:+.1f}% away)")
                if p.get('take_profit'):
                    tp_dist = p.get('tp_distance_pct', 0)
                    risk_parts.append(f"TP: ${p['take_profit']:.2f} ({tp_dist:+.1f}% away)")
                if risk_parts:
                    lines.append(f"    {' | '.join(risk_parts)}")

                # Risk exposure
                if p.get('exposure_percent'):
                    lines.append(f"    Risk: {p['exposure_percent']:.1f}% of equity")

                # Symbol history
                if p.get('symbol_total_trades', 0) > 0:
                    lines.append(
                        f"    Symbol history: {p['symbol_total_trades']} trades, "
                        f"{p.get('symbol_win_rate', 0):.0f}% win rate"
                    )

                position_lines.append("\n".join(lines))

            positions_str = "\n".join(position_lines)

        # Format recent trades
        trades_str = "None" if not context.recent_trades else "\n".join(
            f"  - {t['symbol']} {t['side']} {t['quantity']} @ ${t.get('filled_avg_price', 'pending')}"
            for t in context.recent_trades[:5]
        )

        # Format symbol data
        symbols_str = ""
        for symbol, data in context.symbols.items():
            symbols_str += f"""
{symbol}:
  Current Price: ${data.get('price', 0):.2f}
  Daily Change: {data.get('daily_change_percent', 0):+.2f}%
  RSI: {data.get('rsi', 50):.1f}
  MACD Histogram: {data.get('macd_histogram', 0):.4f}
  Above VWAP: {data.get('above_vwap', False)}
  Short-term Trend: {data.get('trend', 'unknown')}
"""

        return f"""
=== HOURLY TRADING DECISION ===
Timestamp: {context.timestamp.strftime('%Y-%m-%d %H:%M ET')}

ACCOUNT STATUS:
  Equity: ${context.account.get('equity', 100000):.2f}
  Cash: ${context.account.get('cash', 100000):.2f}
  Buying Power: ${context.account.get('buying_power', 100000):.2f}
  Daily P&L: ${context.account.get('daily_pnl', 0):.2f} ({context.account.get('daily_pnl_percent', 0):+.2f}%)

CURRENT POSITIONS:
{positions_str}

RECENT TRADES:
{trades_str}

MARKET DATA:
{symbols_str}

NEWS SENTIMENT:
{self._format_news(context)}

MARKET CONDITION: {context.market_condition}

---

Analyze the market conditions and make your trading decision.
You can use tools to get more detailed data if needed.

After your analysis, clearly state:
1. STRATEGY: Which strategy are you using and why
2. ACTION: BUY/SELL/HOLD/CLOSE
3. If trading: SYMBOL, QUANTITY, STOP_LOSS, and optionally TAKE_PROFIT
4. REASONING: Why this is the right decision

Remember: You're competing against Grok. Make smart, risk-adjusted decisions.
"""

    def _format_news(self, context: MarketContext) -> str:
        """Format news sentiment data for the context."""
        if not context.news_sentiment:
            return "No news data available"

        lines = []
        for symbol in context.symbols.keys():
            sentiment = context.news_sentiment.get(symbol, {})
            if not sentiment:
                lines.append(f"  {symbol}: No recent news")
                continue

            label = sentiment.get("sentiment_label", "neutral").upper()
            score = sentiment.get("avg_sentiment", 0)
            count = sentiment.get("article_count", 0)
            latest = sentiment.get("latest_headline", "")

            score_str = f" ({score:+.2f})" if score != 0 else ""
            lines.append(f"  {symbol}: {label}{score_str} ({count} articles)")

            # Add latest headline if available
            if latest and latest != "No recent news":
                headline_short = latest[:60] + "..." if len(latest) > 60 else latest
                lines.append(f"    Latest: \"{headline_short}\"")

        return "\n".join(lines) if lines else "No news data available"

    def _parse_decision(
        self, response: Any, timestamp: datetime, tool_calls: list
    ) -> TradingDecision:
        """Parse Claude's response into a TradingDecision."""
        # Extract text content from response
        text_content = ""
        for block in response.content:
            if hasattr(block, "text"):
                text_content += block.text

        # Store strategy explanation
        self.current_strategy_explanation = text_content

        # Default to HOLD
        decision = TradingDecision(
            timestamp=timestamp,
            action=ActionType.HOLD,
            strategy_used=StrategyType.DEFENSIVE,
            reasoning=text_content,
            tool_calls=tool_calls,
        )

        # Parse the response for trading signals
        text_upper = text_content.upper()

        # Detect strategy
        if "MOMENTUM" in text_upper:
            decision.strategy_used = StrategyType.MOMENTUM
        elif "MEAN REVERSION" in text_upper or "MEAN-REVERSION" in text_upper:
            decision.strategy_used = StrategyType.MEAN_REVERSION
        elif "TREND FOLLOWING" in text_upper or "TREND-FOLLOWING" in text_upper:
            decision.strategy_used = StrategyType.TREND_FOLLOWING
        elif "BREAKOUT" in text_upper:
            decision.strategy_used = StrategyType.BREAKOUT
        elif "RANGE" in text_upper:
            decision.strategy_used = StrategyType.RANGE_TRADING

        # Detect action
        if "ACTION: BUY" in text_upper or "ACTION:BUY" in text_upper:
            decision.action = ActionType.BUY
        elif "ACTION: SELL" in text_upper or "ACTION:SELL" in text_upper:
            decision.action = ActionType.SELL
        elif "ACTION: CLOSE" in text_upper or "ACTION:CLOSE" in text_upper:
            decision.action = ActionType.CLOSE

        # If action requires trading details, try to parse them
        if decision.action in [ActionType.BUY, ActionType.SELL, ActionType.CLOSE]:
            # Parse symbol
            if "GOOGL" in text_upper:
                decision.symbol = "GOOGL"
            elif "TSLA" in text_upper:
                decision.symbol = "TSLA"

            # Try to parse numbers (simplified - in production use regex)
            lines = text_content.split("\n")
            for line in lines:
                line_upper = line.upper()
                if "QUANTITY" in line_upper or "SHARES" in line_upper:
                    try:
                        # Extract number from line
                        numbers = [int(s) for s in line.split() if s.isdigit()]
                        if numbers:
                            decision.quantity = numbers[0]
                    except ValueError:
                        pass

                if "STOP" in line_upper and "LOSS" in line_upper:
                    try:
                        # Extract price from line
                        parts = line.replace("$", "").replace(",", "").split()
                        for part in parts:
                            try:
                                price = float(part)
                                if price > 10:  # Sanity check for stock price
                                    decision.stop_loss = price
                                    break
                            except ValueError:
                                continue
                    except ValueError:
                        pass

                if "TAKE" in line_upper and "PROFIT" in line_upper:
                    try:
                        parts = line.replace("$", "").replace(",", "").split()
                        for part in parts:
                            try:
                                price = float(part)
                                if price > 10:
                                    decision.take_profit = price
                                    break
                            except ValueError:
                                continue
                    except ValueError:
                        pass

        return decision

    def get_strategy_explanation(self) -> str:
        """Return Claude's latest strategy explanation."""
        return self.current_strategy_explanation

    async def generate_reflection(
        self,
        episode_id: int,
        decision_made: dict,
        market_context: dict,
        outcome_pnl: float,
        outcome_status: str
    ) -> dict:
        """
        Use Claude to reflect on a trade outcome and extract lessons.

        Returns dict with: what_worked, what_failed, lesson_learned, next_time_will, tags
        """
        if not LEARNING_ENABLED:
            return {}

        reflection_prompt = f"""You made a trading decision that has now completed. Analyze what happened and learn from it.

DECISION MADE:
- Action: {decision_made.get('action', 'HOLD')}
- Symbol: {decision_made.get('symbol', 'N/A')}
- Strategy: {decision_made.get('strategy', 'unknown')}
- Your reasoning at the time: {decision_made.get('reasoning', 'N/A')[:500]}

MARKET CONTEXT AT DECISION TIME:
{json.dumps(market_context, indent=2, default=str)[:1000]}

OUTCOME:
- P&L: ${outcome_pnl:+.2f}
- Status: {outcome_status}

Reflect honestly on this trade. Respond in JSON format:
{{
    "what_worked": "What aspects of your analysis or reasoning were correct",
    "what_failed": "What you got wrong or missed",
    "lesson_learned": "The key takeaway from this experience",
    "next_time_will": "What you'll do differently in similar situations",
    "tags": ["RELEVANT", "TAGS", "FOR", "SEARCH"]
}}

Tags should include: the symbol, strategy used, market condition, and any relevant indicators.
Be specific and actionable in your reflections."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system="You are analyzing your own trading decisions to learn and improve. Be honest and specific in your reflections. Respond only with valid JSON.",
                messages=[{"role": "user", "content": reflection_prompt}]
            )

            # Parse JSON from response
            text_content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text_content += block.text

            # Try to extract JSON from the response
            try:
                # Handle case where JSON is wrapped in markdown code blocks
                if "```json" in text_content:
                    text_content = text_content.split("```json")[1].split("```")[0]
                elif "```" in text_content:
                    text_content = text_content.split("```")[1].split("```")[0]

                reflection = json.loads(text_content.strip())
                return {
                    "what_worked": reflection.get("what_worked", ""),
                    "what_failed": reflection.get("what_failed", ""),
                    "lesson_learned": reflection.get("lesson_learned", ""),
                    "next_time_will": reflection.get("next_time_will", ""),
                    "tags": reflection.get("tags", [])
                }
            except json.JSONDecodeError:
                # Fallback: extract what we can
                return {
                    "what_worked": "",
                    "what_failed": "",
                    "lesson_learned": text_content[:500] if text_content else "",
                    "next_time_will": "",
                    "tags": [decision_made.get("symbol", ""), decision_made.get("strategy", "")]
                }

        except Exception as e:
            print(f"Error generating reflection: {e}")
            return {
                "what_worked": "",
                "what_failed": "",
                "lesson_learned": f"Reflection failed: {str(e)}",
                "next_time_will": "",
                "tags": []
            }
