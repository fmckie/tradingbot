"""Grok AI trading agent using xAI API."""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any

import httpx

from config.settings import LEARNING_ENABLED
from database.json_utils import safe_json_dumps
from tools.analysis_tools import ANALYSIS_TOOLS_SCHEMA
from tools.market_tools import MARKET_TOOLS_SCHEMA
from tools.news_tools import NEWS_TOOLS_SCHEMA
from tools.trading_tools import READ_ONLY_TRADING_TOOLS_SCHEMA

from .base_agent import (
    ActionType,
    BaseTradingAgent,
    MarketContext,
    StrategyType,
    TradingDecision,
)

logger = logging.getLogger(__name__)

GROK_SYSTEM_PROMPT = (
    "You are an autonomous AI trading agent managing a $100,000 paper "
    """trading portfolio.
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
You have access to news tools that provide sentiment-scored headlines. """
    """News sentiment can be:
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
5. Emit the structured ACTION block (below) to request the trade
6. If no clear setup, HOLD (being flat is a valid position)

HOW ORDERS ARE EXECUTED (IMPORTANT):
You CANNOT place, close, or cancel orders yourself. You have no order-execution
tools. To trade, you MUST end your response with a strict decision block, one
field per line, exactly:

ACTION: BUY
SYMBOL: TSLA
QUANTITY: 120
STOP_LOSS: 393.00
TAKE_PROFIT: 418.00   (optional)

Use ACTION: SELL or ACTION: CLOSE the same way (SYMBOL required; QUANTITY for
SELL). To stay flat, end with exactly: ACTION: HOLD

The system parses that block and routes the trade through the hard risk layer,
which sizes it, attaches the protective stop on the broker, and logs it. If you
omit the block (or write prose like "order placed"), NO trade happens — it is
treated as HOLD. Do not wrap the labels in markdown; write them plainly.

You are competing against another AI (Claude). Make decisions that """
    """maximize risk-adjusted returns.
Quality of decisions matters more than quantity of trades.

Always explain your reasoning clearly. Your decisions and explanations """
    "will be logged for analysis."
)


class GrokAgent(BaseTradingAgent):
    """Trading agent powered by Grok (xAI)."""

    # Retry configuration
    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 5.0  # seconds
    REQUEST_TIMEOUT = 360.0  # seconds (increased from 180)

    def __init__(self, tools: dict[str, Any]):
        super().__init__("grok", tools)
        self.api_key = os.getenv("XAI_API_KEY")
        self.base_url = "https://api.x.ai/v1"
        self.model = "grok-4.3"  # Grok 4.3
        self.current_strategy_explanation = ""

        # Validate API key at startup
        if not self.api_key:
            raise ValueError("XAI_API_KEY environment variable not set")
        if not self.api_key.startswith("xai-"):
            logger.warning("XAI_API_KEY may be invalid - expected 'xai-' prefix")

        logger.info(f"GrokAgent initialized with model: {self.model}")

        # Combine all tool schemas for OpenAI-compatible format. Only the
        # READ-ONLY trading tools are exposed: order execution is a privileged,
        # gated operation, not an agent tool. Trades are requested via the
        # structured ACTION block (see system prompt).
        self.tool_schemas = self._convert_tools_to_openai_format(
            MARKET_TOOLS_SCHEMA
            + READ_ONLY_TRADING_TOOLS_SCHEMA
            + ANALYSIS_TOOLS_SCHEMA
            + NEWS_TOOLS_SCHEMA
        )

    def _convert_tools_to_openai_format(
        self, anthropic_tools: list[Any]
    ) -> list[dict[str, Any]]:
        """Convert Anthropic tool format to OpenAI format (used by xAI)."""
        openai_tools = []
        for tool in anthropic_tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
            )
        return openai_tools

    async def analyze_and_decide(self, context: MarketContext) -> TradingDecision:
        """
        Use Grok to analyze market and make trading decision.

        Grok receives the market context and can call tools for additional
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

        messages = [
            {"role": "system", "content": GROK_SYSTEM_PROMPT},
            {"role": "user", "content": context_message},
        ]

        tool_calls_made: list[dict[str, Any]] = []
        max_iterations = 5

        async with httpx.AsyncClient() as client:
            for iteration in range(max_iterations):
                # Retry logic with exponential backoff
                response = None
                last_error = None

                for attempt in range(self.MAX_RETRIES):
                    try:
                        logger.debug(
                            f"API call attempt {attempt + 1}/{self.MAX_RETRIES}"
                        )
                        response = await client.post(
                            f"{self.base_url}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": self.model,
                                "messages": messages,
                                "tools": self.tool_schemas,
                                "tool_choice": "auto",
                                "max_tokens": 4096,
                            },
                            timeout=self.REQUEST_TIMEOUT,
                        )

                        if response.status_code == 200:
                            break  # Success
                        elif response.status_code == 429:
                            # Rate limited - wait and retry
                            wait_time = self.BASE_RETRY_DELAY * (2**attempt)
                            logger.warning(
                                f"Rate limited (429), retrying in {wait_time}s"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        elif response.status_code >= 500:
                            # Server error - retry
                            wait_time = self.BASE_RETRY_DELAY * (2**attempt)
                            logger.warning(
                                f"Server error ({response.status_code}), "
                                f"retrying in {wait_time}s"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            # Client error - don't retry
                            break

                    except httpx.TimeoutException:
                        last_error = "Request timed out"
                        logger.warning(
                            f"Timeout on attempt {attempt + 1}/{self.MAX_RETRIES}"
                        )
                        if attempt < self.MAX_RETRIES - 1:
                            await asyncio.sleep(self.BASE_RETRY_DELAY * (2**attempt))
                        continue
                    except httpx.ConnectError as e:
                        last_error = f"Connection failed: {e}"
                        logger.warning(
                            f"Connection error on attempt {attempt + 1}: {e}"
                        )
                        if attempt < self.MAX_RETRIES - 1:
                            await asyncio.sleep(self.BASE_RETRY_DELAY * (2**attempt))
                        continue

                # Check if we got a successful response
                if response is None or response.status_code != 200:
                    error_msg = (
                        last_error
                        if last_error
                        else (
                            "API error: "
                            f"{response.status_code if response else 'no response'}"
                        )
                    )
                    if response:
                        error_msg += f" - {response.text[:500]}"
                    logger.error(
                        f"Grok API failed after {self.MAX_RETRIES} attempts: "
                        f"{error_msg}"
                    )
                    return TradingDecision(
                        timestamp=context.timestamp,
                        action=ActionType.HOLD,
                        strategy_used=StrategyType.DEFENSIVE,
                        reasoning=f"API error (falling back to HOLD): {error_msg}",
                        tool_calls=tool_calls_made,
                    )

                result = response.json()
                choice = result["choices"][0]
                message = choice["message"]

                # Check if Grok wants to use tools
                if message.get("tool_calls"):
                    # Process tool calls
                    messages.append(message)

                    for tool_call in message["tool_calls"]:
                        tool_name = tool_call["function"]["name"]
                        tool_input = json.loads(tool_call["function"]["arguments"])
                        tool_id = tool_call["id"]

                        logger.info(f"Grok calling tool: {tool_name} with {tool_input}")

                        # Execute the tool
                        result = self.execute_tool(tool_name, tool_input)
                        tool_calls_made.append(
                            {"tool": tool_name, "input": tool_input, "result": result}
                        )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": safe_json_dumps(result),
                            }
                        )
                else:
                    # Grok has finished reasoning
                    logger.info(
                        f"Grok completed analysis after {iteration + 1} iteration(s)"
                    )
                    break

        # Parse Grok's final response into a TradingDecision
        decision = self._parse_decision(message, context.timestamp, tool_calls_made)
        self.record_decision(decision)

        return decision

    def _format_context(self, context: MarketContext) -> str:
        """Format market context as a clear message for Grok."""
        # Format positions with enriched data
        positions_str = "None"
        if context.positions:
            position_lines = []
            for p in context.positions:
                lines = [
                    f"  {p['symbol']}: {p['quantity']} shares "
                    f"@ ${p['avg_entry_price']:.2f}"
                ]

                # P&L line (total + today)
                pnl_line = (
                    f"    P&L: ${p['unrealized_pnl']:+.2f} "
                    f"({p['unrealized_pnl_percent']:+.1f}%)"
                )
                if p.get("intraday_pnl") is not None:
                    pnl_line += (
                        f" | Today: ${p['intraday_pnl']:+.2f} "
                        f"({p.get('intraday_pnl_percent', 0):+.1f}%)"
                    )
                lines.append(pnl_line)

                # Time context
                if p.get("holding_duration"):
                    time_str = f"    Holding: {p['holding_duration']}"
                    if p.get("entry_time_str") and p["entry_time_str"] != "Unknown":
                        time_str += f" since {p['entry_time_str']}"
                    lines.append(time_str)

                # Stop/TP with distance
                risk_parts = []
                if p.get("stop_loss"):
                    stop_dist = p.get("stop_distance_pct", 0)
                    risk_parts.append(
                        f"Stop: ${p['stop_loss']:.2f} ({stop_dist:+.1f}% away)"
                    )
                if p.get("take_profit"):
                    tp_dist = p.get("tp_distance_pct", 0)
                    risk_parts.append(
                        f"TP: ${p['take_profit']:.2f} ({tp_dist:+.1f}% away)"
                    )
                if risk_parts:
                    lines.append(f"    {' | '.join(risk_parts)}")

                # Risk exposure
                if p.get("exposure_percent"):
                    lines.append(f"    Risk: {p['exposure_percent']:.1f}% of equity")

                # Symbol history
                if p.get("symbol_total_trades", 0) > 0:
                    lines.append(
                        f"    Symbol history: {p['symbol_total_trades']} trades, "
                        f"{p.get('symbol_win_rate', 0):.0f}% win rate"
                    )

                position_lines.append("\n".join(lines))

            positions_str = "\n".join(position_lines)

        # Format recent trades
        trades_str = (
            "None"
            if not context.recent_trades
            else "\n".join(
                f"  - {t['symbol']} {t['side']} {t['quantity']} "
                f"@ ${t.get('filled_avg_price', 'pending')}"
                for t in context.recent_trades[:5]
            )
        )

        # Format symbol data
        symbols_str = ""
        for symbol, data in context.symbols.items():
            symbols_str += f"""
{symbol}:
  Current Price: ${data.get("price", 0):.2f}
  Daily Change: {data.get("daily_change_percent", 0):+.2f}%
  RSI: {data.get("rsi", 50):.1f}
  MACD Histogram: {data.get("macd_histogram", 0):.4f}
  Above VWAP: {data.get("above_vwap", False)}
  Short-term Trend: {data.get("trend", "unknown")}
"""

        return (
            f"""
=== HOURLY TRADING DECISION ===
Timestamp: {context.timestamp.strftime("%Y-%m-%d %H:%M ET")}

ACCOUNT STATUS:
  Equity: ${context.account.get("equity", 100000):.2f}
  Cash: ${context.account.get("cash", 100000):.2f}
  Buying Power: ${context.account.get("buying_power", 100000):.2f}
  Daily P&L: ${context.account.get("daily_pnl", 0):.2f} """
            f"""({context.account.get("daily_pnl_percent", 0):+.2f}%)

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
You can use tools to get more detailed data if needed (these are read-only — you
cannot place orders yourself).

After your analysis, explain your STRATEGY and REASONING, then END your response
with a strict decision block, one field per line, no markdown around the labels:

To trade:
ACTION: BUY            (or SELL / CLOSE)
SYMBOL: TSLA           (GOOGL or TSLA)
QUANTITY: 120          (shares; required for BUY/SELL)
STOP_LOSS: 393.00      (required for BUY/SELL)
TAKE_PROFIT: 418.00    (optional)

To stay flat, end with exactly:
ACTION: HOLD

The system reads this block and routes the trade through the hard risk gate,
which sizes it, attaches the protective stop, and logs it. Omit the block and
nothing trades (treated as HOLD).

Remember: You're competing against Claude. Make smart, risk-adjusted decisions.
"""
        )

    def _format_news(self, context: MarketContext) -> str:
        """Format news sentiment data for the context."""
        if not context.news_sentiment:
            return "No news data available"

        lines = []
        for symbol in context.symbols:
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
                lines.append(f'    Latest: "{headline_short}"')

        return "\n".join(lines) if lines else "No news data available"

    def _parse_decision(
        self, message: dict[str, Any], timestamp: datetime, tool_calls: list[Any]
    ) -> TradingDecision:
        """Parse Grok's response into a TradingDecision."""
        text_content = message.get("content", "")

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

        if not text_content:
            logger.warning("Grok returned empty content, defaulting to HOLD")
            decision.reasoning = (
                "API returned empty response - defaulting to defensive HOLD"
            )
            return decision

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

        # Detect action. Tolerate markdown/whitespace around the label, e.g.
        # "ACTION: BUY", "ACTION:BUY", "**ACTION: BUY**", "**ACTION:** BUY".
        # re.search returns the leftmost match, so the first stated action wins.
        action_match = re.search(
            r"\bACTION\b[\s:*\-]*\b(BUY|SELL|HOLD|CLOSE)\b",
            text_content,
            re.IGNORECASE,
        )
        if action_match:
            decision.action = {
                "BUY": ActionType.BUY,
                "SELL": ActionType.SELL,
                "HOLD": ActionType.HOLD,
                "CLOSE": ActionType.CLOSE,
            }[action_match.group(1).upper()]

        # If action requires trading details, try to parse them
        if decision.action in [ActionType.BUY, ActionType.SELL, ActionType.CLOSE]:
            # Parse symbol
            if "GOOGL" in text_upper:
                decision.symbol = "GOOGL"
            elif "TSLA" in text_upper:
                decision.symbol = "TSLA"

            # Parse numbers with regex. Grok formats these many ways, e.g.:
            #   "QUANTITY: 150", "QUANTITY: 15 shares", "Shares: 20"
            #   "STOP LOSS: $175", "STOP_LOSS: 175", "Stop-Loss: $175",
            #   "set stop loss at $175", "$1,175.00" (thousands separator)
            qty_match = re.search(
                r"(?:QUANTITY|SHARES)[:\s]+(\d+)", text_content, re.IGNORECASE
            )
            if qty_match:
                decision.quantity = int(qty_match.group(1))

            def _extract_price(label_regex: str) -> float | None:
                """Extract a dollar price following a label, or None.

                Tolerates space/underscore/hyphen inside the label, an optional
                "at" connector and "$" sign, and thousands separators. Values
                below $10 are treated as parse noise and rejected (the traded
                universe sits well above $10), which guards against grabbing a
                stray fragment as a price.
                """
                match = re.search(
                    label_regex + r"[:\s]*(?:at\s+)?\$?\s*([\d,]+(?:\.\d+)?)",
                    text_content,
                    re.IGNORECASE,
                )
                if not match:
                    return None
                value = float(match.group(1).replace(",", ""))
                return value if value >= 10 else None

            stop_loss = _extract_price(r"STOP[\s_-]*LOSS")
            if stop_loss is not None:
                decision.stop_loss = stop_loss

            take_profit = _extract_price(r"TAKE[\s_-]*PROFIT")
            if take_profit is not None:
                decision.take_profit = take_profit

        # Log the parsed decision
        logger.info(
            f"Grok decision: {decision.action.value} "
            f"strategy={decision.strategy_used.value} "
            f"symbol={decision.symbol or 'N/A'} "
            f"qty={decision.quantity or 'N/A'}"
        )

        return decision

    def get_strategy_explanation(self) -> str:
        """Return Grok's latest strategy explanation."""
        return self.current_strategy_explanation

    async def generate_reflection(
        self,
        episode_id: int,
        decision_made: dict[str, Any],
        market_context: dict[str, Any],
        outcome_pnl: float,
        outcome_status: str,
    ) -> dict[str, Any]:
        """
        Use Grok to reflect on a trade outcome and extract lessons.

        Returns dict with: what_worked, what_failed, lesson_learned,
        next_time_will, tags
        """
        if not LEARNING_ENABLED:
            return {}

        reflection_prompt = (
            f"""You made a trading decision that has now completed. """
            f"""Analyze what happened and learn from it.

DECISION MADE:
- Action: {decision_made.get("action", "HOLD")}
- Symbol: {decision_made.get("symbol", "N/A")}
- Strategy: {decision_made.get("strategy", "unknown")}
- Your reasoning at the time: {decision_made.get("reasoning", "N/A")[:500]}

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

Tags should include: the symbol, strategy used, market condition, """
            f"""and any relevant indicators.
Be specific and actionable in your reflections."""
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are analyzing your own trading "
                                    "decisions to learn and improve. Be honest "
                                    "and specific in your reflections. Respond "
                                    "only with valid JSON."
                                ),
                            },
                            {"role": "user", "content": reflection_prompt},
                        ],
                        "max_tokens": 1024,
                    },
                    timeout=30.0,
                )

                if response.status_code != 200:
                    return {
                        "what_worked": "",
                        "what_failed": "",
                        "lesson_learned": f"API error: {response.status_code}",
                        "next_time_will": "",
                        "tags": [],
                    }

                result = response.json()
                text_content = result["choices"][0]["message"].get("content", "")

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
                        "tags": reflection.get("tags", []),
                    }
                except json.JSONDecodeError:
                    # Fallback: extract what we can
                    return {
                        "what_worked": "",
                        "what_failed": "",
                        "lesson_learned": text_content[:500] if text_content else "",
                        "next_time_will": "",
                        "tags": [
                            decision_made.get("symbol", ""),
                            decision_made.get("strategy", ""),
                        ],
                    }

        except Exception as e:
            logger.error(f"Error generating reflection: {e}")
            return {
                "what_worked": "",
                "what_failed": "",
                "lesson_learned": f"Reflection failed: {str(e)}",
                "next_time_will": "",
                "tags": [],
            }
