#!/usr/bin/env python3
"""
AI Trading Competition: Claude Opus 4.5 vs Grok

This is the main competition runner that orchestrates hourly trading decisions
from both AI agents on GOOGL and TSLA.
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Any, Optional
import pytz
from alpaca.trading.models import TradeAccount, Position, Order
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
import schedule
from rich.console import Console
from rich.live import Live
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config.alpaca_config import get_claude_client, get_grok_client
from config.settings import (
    TRADING_HOURS, SYMBOLS, STARTING_CAPITAL,
    LEARNING_ENABLED, SIGNIFICANT_PNL_THRESHOLD, POSTGRES_URL
)
from data.market_data import MarketDataProvider
from data.indicators import TechnicalIndicators
from data.news import NewsProvider
from tools.market_tools import MarketTools
from tools.trading_tools import TradingTools
from tools.analysis_tools import AnalysisTools
from tools.news_tools import NewsTools
from agents.claude_agent import ClaudeAgent
from agents.grok_agent import GrokAgent
from agents.base_agent import MarketContext, TradingDecision, ActionType
from risk.risk_manager import RiskManager
from execution.order_executor import OrderExecutor
from monitoring.scoreboard import Scoreboard
from monitoring.logger import TradeLogger


console = Console()
ET = pytz.timezone("America/New_York")

# Conditionally import learning system
if LEARNING_ENABLED and POSTGRES_URL:
    try:
        from database.postgres_client import PostgresClient, init_database
        from database.learning_store import (
            LearningStore, Episode, Reflection, Learning,
            CompetitionScore, OutcomeStatus
        )
        LEARNING_AVAILABLE = True
    except ImportError as e:
        console.print(f"[yellow]Learning system not available: {e}[/yellow]")
        LEARNING_AVAILABLE = False
else:
    LEARNING_AVAILABLE = False


class TradingCompetition:
    """
    Orchestrates the AI trading competition between Claude and Grok.

    Each hour during market hours:
    1. Gather market context for GOOGL and TSLA
    2. Claude makes its decision
    3. Grok makes its decision
    4. Execute trades with risk validation
    5. Update scoreboard
    6. Log everything
    """

    def __init__(self, skip_market_check: bool = False):
        console.print("[bold yellow]Initializing AI Trading Competition...[/bold yellow]")
        self.skip_market_check = skip_market_check

        # Initialize Alpaca clients for both accounts
        self.claude_trading, self.claude_data = get_claude_client()
        self.grok_trading, self.grok_data = get_grok_client()

        # Market data providers (same data for both - fair comparison)
        self.claude_market_data = MarketDataProvider(self.claude_data)
        self.grok_market_data = MarketDataProvider(self.grok_data)

        # Technical indicators
        self.claude_indicators = TechnicalIndicators(self.claude_market_data)
        self.grok_indicators = TechnicalIndicators(self.grok_market_data)

        # News provider (shared - same news for both agents)
        self.news_provider = NewsProvider()

        # Initialize tools for each agent
        self.claude_tools = self._create_tools(
            self.claude_market_data, self.claude_indicators, self.claude_trading, "claude"
        )
        self.grok_tools = self._create_tools(
            self.grok_market_data, self.grok_indicators, self.grok_trading, "grok"
        )

        # Initialize AI agents
        self.claude_agent = ClaudeAgent(self.claude_tools)
        self.grok_agent = GrokAgent(self.grok_tools)

        # Risk managers
        self.claude_risk = RiskManager(self.claude_trading, "claude")
        self.grok_risk = RiskManager(self.grok_trading, "grok")

        # Order executors
        self.claude_executor = OrderExecutor(self.claude_trading, self.claude_risk, "claude")
        self.grok_executor = OrderExecutor(self.grok_trading, self.grok_risk, "grok")

        # Monitoring
        self.scoreboard = Scoreboard()
        self.scoreboard.register_agent("claude")
        self.scoreboard.register_agent("grok")
        self.logger = TradeLogger()

        # State
        self.is_running = False
        self.last_decision_time: Optional[datetime] = None

        # Learning system state - tracks pending episodes awaiting outcome
        self.pending_episodes: dict[str, list[int]] = {"claude": [], "grok": []}

        console.print("[bold green]Competition initialized successfully![/bold green]")

    def _create_tools(
        self,
        market_data: MarketDataProvider,
        indicators: TechnicalIndicators,
        trading_client,
        agent_name: str,
    ) -> dict:
        """Create tool handlers for an agent."""
        return {
            "market": MarketTools(market_data, indicators),
            "trading": TradingTools(trading_client, agent_name),
            "analysis": AnalysisTools(market_data, indicators),
            "news": NewsTools(self.news_provider),
        }

    def _find_position_entry_time(
        self, trading_client, symbol: str
    ) -> Optional[datetime]:
        """Find the entry time for a position by looking at recent orders."""
        try:
            orders = trading_client.get_orders(
                GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED,
                    symbols=[symbol],
                    limit=20
                )
            )
            # Find the most recent BUY order that opened the position
            for o in orders:
                if isinstance(o, Order) and o.side and o.side.value == "buy" and o.filled_at:
                    return o.filled_at
        except Exception:
            pass
        return None

    def _calculate_holding_duration(
        self, entry_time: Optional[datetime]
    ) -> tuple[float, str]:
        """Calculate holding duration in hours and formatted string."""
        if not entry_time:
            return 0.0, "Unknown"

        now = datetime.now(ET)
        # Make entry_time timezone-aware if it isn't
        if entry_time.tzinfo is None:
            entry_time = ET.localize(entry_time)
        else:
            entry_time = entry_time.astimezone(ET)

        delta = now - entry_time
        total_hours = delta.total_seconds() / 3600

        if total_hours < 1:
            minutes = int(delta.total_seconds() / 60)
            return total_hours, f"{minutes}m"
        elif total_hours < 24:
            hours = int(total_hours)
            minutes = int((total_hours - hours) * 60)
            return total_hours, f"{hours}h {minutes}m"
        else:
            days = int(total_hours / 24)
            hours = int(total_hours % 24)
            return total_hours, f"{days}d {hours}h"

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(ET)

        # Check if weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        market_open = now.replace(
            hour=TRADING_HOURS.market_open_hour,
            minute=TRADING_HOURS.market_open_minute,
            second=0,
            microsecond=0,
        )
        market_close = now.replace(
            hour=TRADING_HOURS.market_close_hour,
            minute=TRADING_HOURS.market_close_minute,
            second=0,
            microsecond=0,
        )

        return market_open <= now <= market_close

    async def gather_market_context(self, agent_name: str) -> MarketContext:
        """Gather current market context for an agent."""
        if agent_name == "claude":
            market_data = self.claude_market_data
            indicators = self.claude_indicators
            trading_client = self.claude_trading
            tools = self.claude_tools
        else:
            market_data = self.grok_market_data
            indicators = self.grok_indicators
            trading_client = self.grok_trading
            tools = self.grok_tools

        # Get market data for each symbol
        symbols_data = {}
        for symbol in SYMBOLS:
            try:
                snap = market_data.get_snapshot(symbol)
                ind = indicators.get_all_indicators(symbol)

                daily_change = 0.0
                if snap.prev_daily_bar_close > 0:
                    daily_change = (
                        (snap.daily_bar_close - snap.prev_daily_bar_close)
                        / snap.prev_daily_bar_close
                        * 100
                    )

                symbols_data[symbol] = {
                    "price": snap.latest_trade_price,
                    "bid": snap.latest_quote_bid,
                    "ask": snap.latest_quote_ask,
                    "daily_open": snap.daily_bar_open,
                    "daily_high": snap.daily_bar_high,
                    "daily_low": snap.daily_bar_low,
                    "daily_close": snap.daily_bar_close,
                    "daily_volume": snap.daily_bar_volume,
                    "daily_change_percent": daily_change,
                    "prev_close": snap.prev_daily_bar_close,
                    "rsi": ind.rsi,
                    "macd_histogram": ind.macd.histogram,
                    "bollinger_percent_b": ind.bollinger.percent_b,
                    "atr": ind.atr,
                    "vwap": ind.vwap,
                    "above_vwap": snap.latest_trade_price > ind.vwap,
                    "ema_9": ind.ema_9,
                    "ema_21": ind.ema_21,
                    "trend": "bullish" if ind.ema_9 > ind.ema_21 else "bearish",
                }
            except Exception as e:
                console.print(f"[red]Error getting data for {symbol}: {e}[/red]")
                symbols_data[symbol] = {"price": 0, "error": str(e)}

        # Get account info
        account_data: dict[str, Any] = {}
        try:
            account = trading_client.get_account()
            if isinstance(account, TradeAccount):
                equity = float(account.equity or 0)
                cash = float(account.cash or 0)
                buying_power = float(account.buying_power or 0)
                portfolio_value = float(account.portfolio_value or 0)
                last_equity = float(account.last_equity or 0)
                account_data = {
                    "equity": equity,
                    "cash": cash,
                    "buying_power": buying_power,
                    "portfolio_value": portfolio_value,
                    "daily_pnl": equity - last_equity,
                    "daily_pnl_percent": (
                        (equity - last_equity) / last_equity * 100
                    )
                    if last_equity > 0
                    else 0.0,
                }
            else:
                account_data = {"equity": STARTING_CAPITAL, "cash": STARTING_CAPITAL, "error": "Invalid account type"}
        except Exception as e:
            console.print(f"[red]Error getting account for {agent_name}: {e}[/red]")
            account_data = {"equity": STARTING_CAPITAL, "cash": STARTING_CAPITAL, "error": str(e)}

        # Get positions with enriched data
        positions_data: list[dict[str, Any]] = []
        equity = account_data.get("equity", STARTING_CAPITAL)
        try:
            positions = trading_client.get_all_positions()
            for p in positions:
                if isinstance(p, Position) and p.symbol in SYMBOLS:
                    # Basic position data
                    avg_entry = float(p.avg_entry_price or 0)
                    current = float(p.current_price or 0)
                    market_val = float(p.market_value or 0)

                    position_dict: dict[str, Any] = {
                        "symbol": p.symbol,
                        "quantity": int(p.qty or 0),
                        "avg_entry_price": avg_entry,
                        "current_price": current,
                        "market_value": market_val,
                        "unrealized_pnl": float(p.unrealized_pl or 0),
                        "unrealized_pnl_percent": float(p.unrealized_plpc or 0) * 100,
                        # Phase 1: Alpaca intraday fields
                        "intraday_pnl": float(p.unrealized_intraday_pl or 0),
                        "intraday_pnl_percent": float(p.unrealized_intraday_plpc or 0) * 100,
                        "change_today_percent": float(p.change_today or 0) * 100,
                    }

                    # Phase 2: Time context
                    entry_time = self._find_position_entry_time(trading_client, p.symbol)
                    holding_hours, holding_str = self._calculate_holding_duration(entry_time)
                    position_dict["entry_time"] = entry_time.isoformat() if entry_time else None
                    position_dict["entry_time_str"] = entry_time.strftime("%H:%M ET") if entry_time else "Unknown"
                    position_dict["holding_hours"] = holding_hours
                    position_dict["holding_duration"] = holding_str

                    # Phase 3: Risk context - exposure percentage
                    position_dict["exposure_percent"] = (market_val / equity * 100) if equity > 0 else 0.0

                    # Phase 3 & 4: Get stop/TP and symbol history from learning system
                    if LEARNING_AVAILABLE:
                        try:
                            # Get stop/TP from stored episode
                            entry_details = await LearningStore.get_position_entry_details(
                                agent_name, p.symbol
                            )
                            if entry_details:
                                stop_loss = entry_details.get("stop_loss")
                                take_profit = entry_details.get("take_profit")
                                position_dict["stop_loss"] = stop_loss
                                position_dict["take_profit"] = take_profit
                                position_dict["entry_strategy"] = entry_details.get("strategy")
                                position_dict["entry_confidence"] = entry_details.get("confidence")

                                # Calculate distance to stop/TP as percentage
                                if stop_loss and current > 0:
                                    position_dict["stop_distance_pct"] = (
                                        (current - stop_loss) / current * 100
                                    )
                                if take_profit and current > 0:
                                    position_dict["tp_distance_pct"] = (
                                        (take_profit - current) / current * 100
                                    )

                            # Get symbol trading history
                            symbol_history = await LearningStore.get_symbol_trade_history(
                                agent_name, p.symbol
                            )
                            position_dict["symbol_total_trades"] = symbol_history["total_trades"]
                            position_dict["symbol_wins"] = symbol_history["wins"]
                            position_dict["symbol_losses"] = symbol_history["losses"]
                            position_dict["symbol_win_rate"] = symbol_history["win_rate"]
                            position_dict["symbol_avg_pnl"] = symbol_history["avg_pnl"]
                        except Exception as e:
                            console.print(f"[yellow]Warning: Failed to get position details: {e}[/yellow]")

                    positions_data.append(position_dict)
        except Exception as e:
            console.print(f"[red]Error getting positions for {agent_name}: {e}[/red]")
            positions_data = []

        # Get recent orders
        recent_trades: list[dict[str, Any]] = []
        try:
            orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=10))
            for o in orders:
                if isinstance(o, Order) and o.symbol in SYMBOLS and o.side is not None:
                    recent_trades.append({
                        "symbol": o.symbol,
                        "side": o.side.value,
                        "quantity": int(o.qty or 0),
                        "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                        "status": o.status.value if o.status else None,
                        "filled_at": o.filled_at.isoformat() if o.filled_at else None,
                    })
        except Exception as e:
            recent_trades = []

        # Determine market condition
        googl_trend = symbols_data.get("GOOGL", {}).get("trend", "unknown")
        tsla_trend = symbols_data.get("TSLA", {}).get("trend", "unknown")

        if googl_trend == tsla_trend == "bullish":
            market_condition = "bullish - both stocks trending up"
        elif googl_trend == tsla_trend == "bearish":
            market_condition = "bearish - both stocks trending down"
        else:
            market_condition = "mixed - stocks showing different trends"

        # Fetch news data
        news_data: dict[str, list[dict]] = {}
        news_sentiment: dict[str, dict] = {}
        try:
            for symbol in SYMBOLS:
                articles = self.news_provider.get_news_for_symbol(symbol, hours_back=24, limit=5)
                news_data[symbol] = [a.to_dict() for a in articles]
                sentiment = self.news_provider.get_sentiment_summary(symbol, hours_back=24)
                news_sentiment[symbol] = sentiment.to_dict()
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to fetch news: {e}[/yellow]")

        return MarketContext(
            timestamp=datetime.now(ET),
            symbols=symbols_data,
            account=account_data,
            positions=positions_data,
            recent_trades=recent_trades,
            market_condition=market_condition,
            news=news_data,
            news_sentiment=news_sentiment,
        )

    async def run_agent_decision(
        self, agent_name: str, agent, executor, context: MarketContext
    ) -> TradingDecision:
        """Run a single agent's decision cycle."""
        console.print(f"\n[bold cyan]{agent_name.upper()}'s turn...[/bold cyan]")

        try:
            # Get AI decision
            decision = await agent.analyze_and_decide(context)

            # Log the decision
            market_context_dict = {
                "symbols": context.symbols,
                "account": context.account,
                "positions": context.positions,
            }
            self.logger.log_decision(agent_name, decision, market_context_dict)

            # Create episode in learning system
            episode_id = await self._create_episode(
                agent_name, context, decision, market_context_dict
            )

            # Execute if it's a trade action
            if decision.action != ActionType.HOLD:
                # Get current price for the symbol
                current_price = 0.0
                if decision.symbol and decision.symbol in context.symbols:
                    current_price = context.symbols[decision.symbol].get("price", 0.0)

                result = await executor.execute_decision(decision, current_price)

                # Log the trade
                self.logger.log_trade(agent_name, decision, result, current_price)

                if result.success:
                    console.print(
                        f"[green]{agent_name.upper()}: {decision.action.value.upper()} "
                        f"{decision.symbol} - {result.message}[/green]"
                    )
                    # Track episode for outcome processing
                    if episode_id and LEARNING_AVAILABLE:
                        self.pending_episodes[agent_name].append(episode_id)
                else:
                    console.print(
                        f"[red]{agent_name.upper()}: Trade failed - {result.message}[/red]"
                    )
            else:
                console.print(
                    f"[yellow]{agent_name.upper()}: HOLD - {decision.strategy_used.value}[/yellow]"
                )
                # Mark HOLD episodes as complete immediately
                if episode_id and LEARNING_AVAILABLE:
                    await LearningStore.update_episode_outcome(
                        episode_id, Decimal("0"), OutcomeStatus.HOLD.value
                    )

            return decision

        except Exception as e:
            console.print(f"[red]Error in {agent_name}'s decision: {e}[/red]")
            return TradingDecision(
                timestamp=datetime.now(ET),
                action=ActionType.HOLD,
                reasoning=f"Error: {str(e)}",
            )

    async def _create_episode(
        self,
        agent_name: str,
        context: MarketContext,
        decision: TradingDecision,
        market_context_dict: dict
    ) -> Optional[int]:
        """Create a learning episode for this decision."""
        if not LEARNING_AVAILABLE:
            return None

        try:
            episode = Episode(
                agent_name=agent_name,
                timestamp=context.timestamp,
                market_regime=context.market_condition,
                symbols_context=context.symbols,
                account_state=context.account,
                decision_made={
                    "action": decision.action.value,
                    "symbol": decision.symbol,
                    "quantity": decision.quantity,
                    "strategy": decision.strategy_used.value,
                    "reasoning": decision.reasoning[:500],
                    "confidence": decision.confidence,
                },
                outcome_pnl=None,
                outcome_status=OutcomeStatus.PENDING.value
            )
            return await LearningStore.create_episode(episode)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to create episode: {e}[/yellow]")
            return None

    async def process_pending_outcomes(self):
        """
        Check pending episodes and process completed trades.

        This analyzes closed positions to determine trade outcomes and triggers
        reflection generation for significant trades.
        """
        if not LEARNING_AVAILABLE:
            return

        for agent_name in ["claude", "grok"]:
            pending = self.pending_episodes[agent_name]
            if not pending:
                continue

            agent = self.claude_agent if agent_name == "claude" else self.grok_agent
            trading_client = self.claude_trading if agent_name == "claude" else self.grok_trading

            # Get current positions
            try:
                positions = trading_client.get_all_positions()
                position_symbols = {p.symbol for p in positions if isinstance(p, Position) and p.symbol in SYMBOLS}
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to get positions for {agent_name}: {e}[/yellow]")
                continue

            # Process each pending episode
            completed_episodes = []
            for episode_id in pending:
                try:
                    episode = await LearningStore.get_episode(episode_id)
                    if not episode:
                        completed_episodes.append(episode_id)
                        continue

                    decision = episode.decision_made
                    symbol = decision.get("symbol")
                    action = decision.get("action")

                    # Skip if not a trade action
                    if action == "hold":
                        completed_episodes.append(episode_id)
                        continue

                    # Check if position is still open
                    if symbol in position_symbols:
                        # Position still open - check for closed orders to calculate P&L
                        continue

                    # Position closed - calculate outcome
                    pnl = await self._calculate_trade_pnl(
                        trading_client, episode.timestamp, symbol
                    )

                    # Determine outcome status
                    if pnl > 0:
                        outcome_status = OutcomeStatus.WIN.value
                    elif pnl < 0:
                        outcome_status = OutcomeStatus.LOSS.value
                    else:
                        outcome_status = OutcomeStatus.BREAKEVEN.value

                    # Update episode
                    await LearningStore.update_episode_outcome(
                        episode_id, Decimal(str(pnl)), outcome_status
                    )

                    # Generate reflection for significant trades
                    if abs(pnl) >= SIGNIFICANT_PNL_THRESHOLD:
                        await self._generate_and_store_reflection(
                            agent, episode_id, decision,
                            episode.symbols_context, pnl, outcome_status
                        )

                    completed_episodes.append(episode_id)
                    console.print(
                        f"[blue]{agent_name.upper()}: Processed outcome for {symbol} - "
                        f"{outcome_status} (${pnl:+.2f})[/blue]"
                    )

                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to process episode {episode_id}: {e}[/yellow]")

            # Remove completed episodes from pending
            for ep_id in completed_episodes:
                if ep_id in self.pending_episodes[agent_name]:
                    self.pending_episodes[agent_name].remove(ep_id)

    async def _calculate_trade_pnl(
        self,
        trading_client,
        since: datetime,
        symbol: str
    ) -> float:
        """Calculate P&L for a completed trade."""
        try:
            # Get recent closed orders for the symbol
            orders = trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol], limit=20)
            )

            # Find orders after the decision timestamp
            relevant_orders = [
                o for o in orders
                if isinstance(o, Order) and o.filled_at and o.filled_at >= since
            ]

            if not relevant_orders:
                return 0.0

            # Simple P&L calculation: sum of (side * qty * price)
            total_pnl = 0.0
            for order in relevant_orders:
                if order.filled_avg_price and order.filled_qty and order.side is not None:
                    price = float(order.filled_avg_price)
                    qty = float(order.filled_qty)
                    # Buy orders reduce P&L, sell orders increase it
                    if order.side.value == "buy":
                        total_pnl -= price * qty
                    else:
                        total_pnl += price * qty

            return total_pnl

        except Exception as e:
            console.print(f"[yellow]Warning: P&L calculation error: {e}[/yellow]")
            return 0.0

    async def _generate_and_store_reflection(
        self,
        agent,
        episode_id: int,
        decision_made: dict,
        market_context: dict,
        outcome_pnl: float,
        outcome_status: str
    ):
        """Generate and store a reflection for a completed trade."""
        if not LEARNING_AVAILABLE:
            return

        try:
            # Have the agent generate a reflection
            reflection_data = await agent.generate_reflection(
                episode_id, decision_made, market_context, outcome_pnl, outcome_status
            )

            if not reflection_data:
                return

            # Determine confidence adjustment based on outcome
            if outcome_status == OutcomeStatus.WIN.value:
                confidence_adj = Decimal("0.05")
            elif outcome_status == OutcomeStatus.LOSS.value:
                confidence_adj = Decimal("-0.05")
            else:
                confidence_adj = Decimal("0")

            # Store the reflection
            reflection = Reflection(
                episode_id=episode_id,
                agent_name=agent.name,
                what_worked=reflection_data.get("what_worked", ""),
                what_failed=reflection_data.get("what_failed", ""),
                lesson_learned=reflection_data.get("lesson_learned", ""),
                next_time_will=reflection_data.get("next_time_will", ""),
                confidence_adjustment=confidence_adj,
                tags=reflection_data.get("tags", [])
            )
            await LearningStore.create_reflection(reflection)

            # Potentially distill into a learning
            await self._maybe_create_learning(agent.name, reflection_data, outcome_status)

        except Exception as e:
            console.print(f"[yellow]Warning: Failed to generate reflection: {e}[/yellow]")

    async def _maybe_create_learning(
        self,
        agent_name: str,
        reflection_data: dict,
        outcome_status: str
    ):
        """Create or update a learning from significant reflection."""
        if not LEARNING_AVAILABLE:
            return

        lesson = reflection_data.get("lesson_learned", "")
        tags = reflection_data.get("tags", [])

        if not lesson or not tags:
            return

        try:
            # Check if similar learning exists
            existing = await LearningStore.find_similar_learning(
                agent_name, lesson[:100], tags
            )

            if existing and existing.id is not None:
                # Update existing learning
                if outcome_status == OutcomeStatus.WIN.value:
                    await LearningStore.increment_learning_success(existing.id)
                else:
                    await LearningStore.increment_learning_failure(existing.id)
            else:
                # Create new learning
                category = "strategy"  # Default category
                for tag in tags:
                    tag_lower = tag.lower()
                    if tag_lower in ["rsi", "macd", "vwap", "ema"]:
                        category = "indicator"
                        break
                    elif tag_lower in ["timing", "entry", "exit"]:
                        category = "timing"
                        break
                    elif tag_lower in ["risk", "stop", "position"]:
                        category = "risk"
                        break

                learning = Learning(
                    agent_name=agent_name,
                    category=category,
                    pattern=reflection_data.get("next_time_will", lesson)[:200],
                    insight=lesson[:500],
                    success_count=1 if outcome_status == OutcomeStatus.WIN.value else 0,
                    failure_count=1 if outcome_status == OutcomeStatus.LOSS.value else 0,
                    is_active=True,
                    tags=tags
                )
                await LearningStore.create_learning(learning)

        except Exception as e:
            console.print(f"[yellow]Warning: Failed to create learning: {e}[/yellow]")

    async def update_scoreboard(self):
        """Update the scoreboard with current performance."""
        for agent_name, trading_client in [
            ("claude", self.claude_trading),
            ("grok", self.grok_trading),
        ]:
            try:
                account = trading_client.get_account()
                positions = trading_client.get_all_positions()

                if not isinstance(account, TradeAccount):
                    continue

                equity = float(account.equity)
                cash = float(account.cash)
                positions_value = sum(
                    float(p.market_value) for p in positions if isinstance(p, Position) and p.symbol in SYMBOLS
                )
                unrealized_pnl = sum(
                    float(p.unrealized_pl) for p in positions if isinstance(p, Position) and p.symbol in SYMBOLS
                )

                # Get agent for strategy info
                agent = self.claude_agent if agent_name == "claude" else self.grok_agent

                self.scoreboard.update_agent(
                    agent_name,
                    equity=equity,
                    cash=cash,
                    positions_value=positions_value,
                    unrealized_pnl=unrealized_pnl,
                    strategy=agent.most_used_strategy,
                    reasoning=agent.get_strategy_explanation()[:200],
                )

                # Log performance snapshot
                score = self.scoreboard.scores[agent_name]
                self.logger.log_performance(
                    agent_name,
                    equity=equity,
                    cash=cash,
                    positions_value=positions_value,
                    unrealized_pnl=unrealized_pnl,
                    daily_pnl=score.daily_pnl,
                    total_trades=score.total_trades,
                    win_rate=score.win_rate,
                    max_drawdown=score.max_drawdown,
                )

            except Exception as e:
                console.print(f"[red]Error updating scoreboard for {agent_name}: {e}[/red]")

    async def run_hourly_cycle(self):
        """Run the hourly decision cycle for both agents."""
        if not self.skip_market_check and not self.is_market_open():
            console.print("[yellow]Market is closed. Waiting...[/yellow]")
            return

        console.print("\n" + "=" * 60)
        console.print(f"[bold]HOURLY CYCLE - {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}[/bold]")
        console.print("=" * 60)

        # Process any pending trade outcomes from previous cycles
        await self.process_pending_outcomes()

        # Gather context for both agents
        claude_context = await self.gather_market_context("claude")
        grok_context = await self.gather_market_context("grok")

        # Run Claude's decision
        await self.run_agent_decision(
            "claude", self.claude_agent, self.claude_executor, claude_context
        )

        # Small delay between agents
        await asyncio.sleep(2)

        # Run Grok's decision
        await self.run_agent_decision("grok", self.grok_agent, self.grok_executor, grok_context)

        # Update scoreboard
        await self.update_scoreboard()

        # Update daily competition scores in learning system
        await self._update_competition_scores()

        # Display scoreboard
        self.scoreboard.display()

        self.last_decision_time = datetime.now(ET)

    async def _update_competition_scores(self):
        """Update daily competition scores in the learning system."""
        if not LEARNING_AVAILABLE:
            return

        today = date.today()

        for agent_name, trading_client in [
            ("claude", self.claude_trading),
            ("grok", self.grok_trading),
        ]:
            try:
                account = trading_client.get_account()
                agent = self.claude_agent if agent_name == "claude" else self.grok_agent

                if not isinstance(account, TradeAccount):
                    continue

                score = CompetitionScore(
                    agent_name=agent_name,
                    date=today,
                    starting_equity=Decimal(str(STARTING_CAPITAL)),
                    ending_equity=Decimal(str(float(account.equity))),
                    daily_pnl=Decimal(str(float(account.equity) - float(account.last_equity))),
                    trades_count=agent.state.total_trades,
                    wins=agent.state.winning_trades,
                    losses=agent.state.losing_trades,
                    strategies_used=agent.state.strategies_used,
                )
                await LearningStore.upsert_daily_score(score)

            except Exception as e:
                console.print(f"[yellow]Warning: Failed to update competition score for {agent_name}: {e}[/yellow]")

    async def start(self):
        """Start the competition."""
        self.is_running = True
        console.print("\n[bold green]STARTING AI TRADING COMPETITION[/bold green]")
        console.print(f"Starting Capital: ${STARTING_CAPITAL:,.2f} per account")
        console.print(f"Symbols: {', '.join(SYMBOLS)}")
        console.print("Decision Frequency: Hourly")

        # Initialize learning system
        if LEARNING_AVAILABLE:
            try:
                await init_database()
                console.print("[green]Learning system: PostgreSQL connected[/green]")
            except Exception as e:
                console.print(f"[yellow]Learning system: Failed to initialize ({e})[/yellow]")
        else:
            console.print("[yellow]Learning system: Disabled (no DATABASE_URL)[/yellow]")

        console.print("")

        # Initial scoreboard update
        await self.update_scoreboard()
        self.scoreboard.display()

        # Run initial cycle if market is open
        if self.is_market_open():
            await self.run_hourly_cycle()

        # Schedule hourly runs
        schedule.every().hour.at(":00").do(
            lambda: asyncio.create_task(self.run_hourly_cycle())
        )

        # Main loop
        while self.is_running:
            schedule.run_pending()
            await asyncio.sleep(10)

    async def stop(self):
        """Stop the competition."""
        self.is_running = False
        console.print("\n[bold yellow]Competition stopped.[/bold yellow]")

        # Process any remaining pending outcomes
        await self.process_pending_outcomes()

        # Display final summary
        console.print("\n[bold]FINAL SUMMARY[/bold]")
        self.scoreboard.display()

        summary = self.logger.get_competition_summary()
        console.print(f"\nCompetition data saved to {self.logger.db_path}")

        # Close PostgreSQL connection pool
        if LEARNING_AVAILABLE:
            try:
                await PostgresClient.close()
                console.print("[green]Learning system: PostgreSQL connection closed[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to close PostgreSQL: {e}[/yellow]")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Trading Competition: Claude Opus 4.5 vs Grok"
    )
    parser.add_argument(
        "--single-cycle",
        action="store_true",
        help="Run a single trading cycle and exit (for cloud deployments)",
    )
    parser.add_argument(
        "--skip-market-check",
        action="store_true",
        help="Skip market hours check (for testing)",
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Check for required environment variables
    required_vars = [
        "CLAUDE_ALPACA_API_KEY",
        "CLAUDE_ALPACA_SECRET_KEY",
        "GROK_ALPACA_API_KEY",
        "GROK_ALPACA_SECRET_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
    ]

    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        console.print(f"[red]Missing environment variables: {', '.join(missing)}[/red]")
        console.print("Please copy .env.template to .env and fill in your credentials.")
        sys.exit(1)

    competition = TradingCompetition()

    try:
        if args.single_cycle:
            # Single cycle mode for cloud deployments (Modal, etc.)
            console.print("[bold cyan]Running single trading cycle...[/bold cyan]")

            # Initialize learning system for single cycle
            if LEARNING_AVAILABLE:
                try:
                    await init_database()
                    console.print("[green]Learning system: PostgreSQL connected[/green]")
                except Exception as e:
                    console.print(f"[yellow]Learning system: Failed to initialize ({e})[/yellow]")

            if args.skip_market_check or competition.is_market_open():
                await competition.run_hourly_cycle()
            else:
                console.print("[yellow]Market is closed. Skipping cycle.[/yellow]")

            # Close PostgreSQL on single cycle completion
            if LEARNING_AVAILABLE:
                await PostgresClient.close()
        else:
            # Continuous mode for local running
            await competition.start()
    except KeyboardInterrupt:
        await competition.stop()
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        if not args.single_cycle:
            await competition.stop()
        raise


if __name__ == "__main__":
    asyncio.run(main())
