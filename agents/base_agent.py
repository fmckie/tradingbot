"""Base class for AI trading agents."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional
from enum import Enum

from config.settings import LEARNING_ENABLED, MAX_LEARNINGS_PER_RECALL


class ActionType(Enum):
    """Types of trading actions."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class StrategyType(Enum):
    """Trading strategy categories for analysis."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    BREAKOUT = "breakout"
    RANGE_TRADING = "range_trading"
    DEFENSIVE = "defensive"  # Holding cash


@dataclass
class MarketContext:
    """Market context provided to AI for decision making."""

    timestamp: datetime
    symbols: dict[str, dict]  # Price and indicator data per symbol
    account: dict  # Account balance, equity, etc.
    positions: list[dict]  # Current open positions
    recent_trades: list[dict]  # Recent trade history
    market_condition: str  # Overall market assessment


@dataclass
class TradingDecision:
    """Decision made by an AI agent."""

    timestamp: datetime
    action: ActionType
    symbol: Optional[str] = None
    quantity: Optional[int] = None
    order_type: str = "market"
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_used: StrategyType = StrategyType.DEFENSIVE
    reasoning: str = ""
    confidence: float = 0.5  # 0-1 confidence score
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class AgentState:
    """Persistent state for an agent across sessions."""

    agent_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    strategies_used: dict[str, int] = field(default_factory=dict)
    last_decision_time: Optional[datetime] = None
    consecutive_losses: int = 0
    peak_equity: float = 100_000.0
    current_drawdown: float = 0.0


class BaseTradingAgent(ABC):
    """
    Abstract base class for AI trading agents.

    Both Claude and Grok agents inherit from this to ensure fair comparison.
    The interface is identical - only the AI decision-making differs.
    """

    def __init__(self, name: str, tools: dict):
        """
        Initialize the trading agent.

        Args:
            name: Agent identifier (e.g., "claude", "grok")
            tools: Dictionary of tool handlers
        """
        self.name = name
        self.tools = tools
        self.state = AgentState(agent_name=name)
        self.decision_history: list[TradingDecision] = []

    @abstractmethod
    async def analyze_and_decide(self, context: MarketContext) -> TradingDecision:
        """
        Main decision method - AI analyzes market and decides action.

        This is where the AI:
        1. Receives market context (prices, indicators, positions)
        2. Can call tools for additional analysis
        3. Reasons about strategy selection
        4. Makes trading decision (buy/sell/hold)
        5. Explains its reasoning

        Args:
            context: Current market context with all relevant data

        Returns:
            TradingDecision with action, reasoning, and strategy used
        """
        pass

    @abstractmethod
    def get_strategy_explanation(self) -> str:
        """
        Get the AI's explanation of its current strategy approach.

        Used for logging and comparison of AI strategies.
        """
        pass

    def execute_tool(self, tool_name: str, parameters: dict) -> dict[str, Any]:
        """Execute a tool and return results."""
        tool_category = self._get_tool_category(tool_name)

        if tool_category and tool_category in self.tools:
            return self.tools[tool_category].execute(tool_name, parameters)

        return {"error": f"Tool {tool_name} not found"}

    def _get_tool_category(self, tool_name: str) -> Optional[str]:
        """Map tool name to its category."""
        market_tools = [
            "get_stock_price",
            "get_price_history",
            "get_technical_indicators",
            "get_market_snapshot",
        ]
        trading_tools = [
            "get_positions",
            "get_account",
            "get_orders",
            "place_order",
            "close_position",
            "cancel_order",
        ]
        analysis_tools = [
            "get_market_context",
            "compare_stocks",
            "get_support_resistance",
            "analyze_trend",
        ]

        if tool_name in market_tools:
            return "market"
        elif tool_name in trading_tools:
            return "trading"
        elif tool_name in analysis_tools:
            return "analysis"

        return None

    def record_decision(self, decision: TradingDecision):
        """Record a decision for history and analysis."""
        self.decision_history.append(decision)
        self.state.last_decision_time = decision.timestamp

        # Track strategy usage
        strategy_name = decision.strategy_used.value
        self.state.strategies_used[strategy_name] = (
            self.state.strategies_used.get(strategy_name, 0) + 1
        )

    def record_trade_result(self, profit: float):
        """Record the result of a completed trade."""
        self.state.total_trades += 1

        if profit > 0:
            self.state.winning_trades += 1
            self.state.consecutive_losses = 0
        else:
            self.state.losing_trades += 1
            self.state.consecutive_losses += 1

    def update_equity(self, current_equity: float):
        """Update peak equity and drawdown tracking."""
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity

        self.state.current_drawdown = (
            (self.state.peak_equity - current_equity) / self.state.peak_equity * 100
        )

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.state.total_trades == 0:
            return 0.0
        return self.state.winning_trades / self.state.total_trades * 100

    @property
    def most_used_strategy(self) -> str:
        """Get the most frequently used strategy."""
        if not self.state.strategies_used:
            return "none"
        return max(self.state.strategies_used, key=self.state.strategies_used.get)

    def get_performance_summary(self) -> dict:
        """Get agent performance summary."""
        return {
            "agent": self.name,
            "total_trades": self.state.total_trades,
            "winning_trades": self.state.winning_trades,
            "losing_trades": self.state.losing_trades,
            "win_rate": round(self.win_rate, 1),
            "strategies_used": self.state.strategies_used,
            "most_used_strategy": self.most_used_strategy,
            "consecutive_losses": self.state.consecutive_losses,
            "peak_equity": self.state.peak_equity,
            "current_drawdown": round(self.state.current_drawdown, 2),
        }

    # ==================== Learning System Methods ====================

    async def recall_learnings(self, context: "MarketContext") -> str:
        """
        Query relevant past learnings before making decision.

        Returns formatted string of learnings to include in the prompt.
        """
        if not LEARNING_ENABLED:
            return ""

        try:
            from database.learning_store import LearningStore, Learning

            learnings: list[Learning] = []

            # 1. Get learnings matching current market regime
            regime = context.market_condition.split(" - ")[0].lower()  # "bullish", "bearish", "mixed"
            regime_learnings = await LearningStore.get_learnings_by_tags(
                agent_name=self.name,
                tags=[regime],
                limit=3
            )
            learnings.extend(regime_learnings)

            # 2. Get learnings matching symbols we're looking at
            symbols = list(context.symbols.keys())
            symbol_learnings = await LearningStore.get_learnings_by_tags(
                agent_name=self.name,
                tags=symbols,
                limit=3
            )
            learnings.extend(symbol_learnings)

            # 3. Get top performing learnings
            top_learnings = await LearningStore.get_top_learnings(
                agent_name=self.name,
                limit=4
            )
            learnings.extend(top_learnings)

            # Deduplicate and limit
            seen_ids = set()
            unique_learnings = []
            for learning in learnings:
                if learning.id not in seen_ids:
                    seen_ids.add(learning.id)
                    unique_learnings.append(learning)
                if len(unique_learnings) >= MAX_LEARNINGS_PER_RECALL:
                    break

            if not unique_learnings:
                return ""

            # Format learnings for prompt
            formatted = "\n\nPAST LEARNINGS (from your previous trading experience):\n"
            for i, learning in enumerate(unique_learnings, 1):
                success_rate = learning.success_rate
                formatted += f"""
{i}. [{learning.category.upper()}] {learning.pattern}
   Insight: {learning.insight}
   Track record: {learning.success_count} wins, {learning.failure_count} losses ({success_rate:.0f}% success)
"""
            formatted += "\nConsider these learnings when making your decision, but don't be rigidly bound by them if current conditions differ."
            return formatted

        except ImportError:
            # Database not configured
            return ""
        except Exception as e:
            # Log but don't fail the decision
            print(f"Warning: Failed to recall learnings: {e}")
            return ""

    async def get_recent_outcomes(self, limit: int = 5) -> str:
        """Get recent episode outcomes for context."""
        if not LEARNING_ENABLED:
            return ""

        try:
            from database.learning_store import LearningStore

            episodes = await LearningStore.get_recent_episodes(
                agent_name=self.name,
                limit=limit
            )

            if not episodes:
                return ""

            # Only include completed episodes (not pending)
            completed = [e for e in episodes if e.outcome_status != "pending"]
            if not completed:
                return ""

            formatted = "\n\nRECENT TRADE OUTCOMES:\n"
            for ep in completed[:3]:
                decision = ep.decision_made
                action = decision.get("action", "HOLD")
                symbol = decision.get("symbol", "-")
                pnl = ep.outcome_pnl or Decimal("0")
                status = ep.outcome_status

                formatted += f"  - {action} {symbol}: {status} (P&L: ${pnl:+.2f})\n"

            return formatted

        except ImportError:
            return ""
        except Exception as e:
            print(f"Warning: Failed to get recent outcomes: {e}")
            return ""

    @abstractmethod
    async def generate_reflection(
        self,
        episode_id: int,
        decision_made: dict,
        market_context: dict,
        outcome_pnl: float,
        outcome_status: str
    ) -> dict:
        """
        Generate a reflection on a trade outcome.

        This is called after a trade closes to analyze what happened.
        Each agent (Claude/Grok) implements this with their own AI.

        Returns:
            dict with keys: what_worked, what_failed, lesson_learned, next_time_will, tags
        """
        pass
