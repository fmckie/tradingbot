"""Comprehensive unit tests for BaseTradingAgent.

Tests cover:
- record_trade_result() - wins, losses, consecutive streaks
- update_equity() - peak equity, drawdown calculation
- win_rate property - division by zero, edge cases
- record_decision() - decision history, strategy tracking
- execute_tool() - tool routing
- most_used_strategy property
- get_performance_summary()
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, AsyncMock, patch

from agents.base_agent import (
    BaseTradingAgent,
    TradingDecision,
    ActionType,
    StrategyType,
    AgentState,
    MarketContext,
)


class ConcreteTestAgent(BaseTradingAgent):
    """Concrete implementation for testing abstract BaseTradingAgent."""

    async def analyze_and_decide(self, context: MarketContext) -> TradingDecision:
        """Test implementation - just returns HOLD."""
        return TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.HOLD,
            strategy_used=StrategyType.DEFENSIVE,
            reasoning="Test agent always holds",
            confidence=0.5,
        )

    def get_strategy_explanation(self) -> str:
        """Test implementation."""
        return "Test agent strategy"

    async def generate_reflection(
        self,
        episode_id: int,
        decision_made: dict,
        market_context: dict,
        outcome_pnl: float,
        outcome_status: str,
    ) -> dict:
        """Test implementation."""
        return {
            "what_worked": "test",
            "what_failed": "test",
            "lesson_learned": "test",
            "next_time_will": "test",
            "tags": ["test"],
        }


@pytest.fixture
def test_agent():
    """Create a test agent with mocked tools."""
    tools = {
        "market": Mock(),
        "trading": Mock(),
        "analysis": Mock(),
    }
    return ConcreteTestAgent(name="test_agent", tools=tools)


@pytest.fixture
def experienced_agent():
    """Create an agent with pre-existing trade history."""
    tools = {
        "market": Mock(),
        "trading": Mock(),
        "analysis": Mock(),
    }
    agent = ConcreteTestAgent(name="experienced_agent", tools=tools)

    # Set up state manually
    agent.state.total_trades = 20
    agent.state.winning_trades = 12
    agent.state.losing_trades = 8
    agent.state.strategies_used = {
        "momentum": 8,
        "mean_reversion": 6,
        "trend_following": 4,
        "breakout": 2,
    }
    agent.state.consecutive_losses = 0
    agent.state.peak_equity = 105000.0
    agent.state.current_drawdown = 2.0

    return agent


class TestRecordTradeResult:
    """Test record_trade_result method."""

    def test_winning_trade_updates_stats(self, test_agent):
        """Winning trade should increment winning_trades and total_trades."""
        test_agent.record_trade_result(profit=100.0)

        assert test_agent.state.total_trades == 1
        assert test_agent.state.winning_trades == 1
        assert test_agent.state.losing_trades == 0

    def test_losing_trade_updates_stats(self, test_agent):
        """Losing trade should increment losing_trades and total_trades."""
        test_agent.record_trade_result(profit=-50.0)

        assert test_agent.state.total_trades == 1
        assert test_agent.state.winning_trades == 0
        assert test_agent.state.losing_trades == 1

    def test_winning_trade_resets_consecutive_losses(self, test_agent):
        """Winning trade should reset consecutive losses counter."""
        test_agent.state.consecutive_losses = 5

        test_agent.record_trade_result(profit=100.0)

        assert test_agent.state.consecutive_losses == 0

    def test_losing_trade_increments_consecutive_losses(self, test_agent):
        """Losing trade should increment consecutive losses."""
        test_agent.record_trade_result(profit=-50.0)
        test_agent.record_trade_result(profit=-30.0)
        test_agent.record_trade_result(profit=-20.0)

        assert test_agent.state.consecutive_losses == 3

    def test_consecutive_losses_reset_after_win(self, test_agent):
        """Consecutive losses should reset after a win."""
        # Build up losses
        test_agent.record_trade_result(profit=-50.0)
        test_agent.record_trade_result(profit=-30.0)
        test_agent.record_trade_result(profit=-20.0)

        assert test_agent.state.consecutive_losses == 3

        # Win resets
        test_agent.record_trade_result(profit=100.0)

        assert test_agent.state.consecutive_losses == 0

    def test_breakeven_trade_counted_as_loss(self, test_agent):
        """Zero profit should be counted as loss (not > 0)."""
        test_agent.record_trade_result(profit=0.0)

        assert test_agent.state.losing_trades == 1
        assert test_agent.state.winning_trades == 0

    def test_very_small_profit_counted_as_win(self, test_agent):
        """Very small positive profit should be counted as win."""
        test_agent.record_trade_result(profit=0.01)

        assert test_agent.state.winning_trades == 1

    def test_multiple_trades_accumulate(self, test_agent):
        """Multiple trades should accumulate correctly."""
        test_agent.record_trade_result(profit=100.0)
        test_agent.record_trade_result(profit=-50.0)
        test_agent.record_trade_result(profit=75.0)
        test_agent.record_trade_result(profit=-25.0)
        test_agent.record_trade_result(profit=150.0)

        assert test_agent.state.total_trades == 5
        assert test_agent.state.winning_trades == 3
        assert test_agent.state.losing_trades == 2


class TestUpdateEquity:
    """Test update_equity method."""

    def test_new_peak_equity_updated(self, test_agent):
        """New high should update peak equity."""
        test_agent.state.peak_equity = 100000.0

        test_agent.update_equity(110000.0)

        assert test_agent.state.peak_equity == 110000.0

    def test_below_peak_no_peak_update(self, test_agent):
        """Below peak should not update peak equity."""
        test_agent.state.peak_equity = 100000.0

        test_agent.update_equity(95000.0)

        assert test_agent.state.peak_equity == 100000.0

    def test_drawdown_calculated_correctly(self, test_agent):
        """Drawdown should be calculated as percentage from peak."""
        test_agent.state.peak_equity = 100000.0

        test_agent.update_equity(90000.0)

        # (100000 - 90000) / 100000 * 100 = 10%
        assert test_agent.state.current_drawdown == 10.0

    def test_no_drawdown_at_peak(self, test_agent):
        """Drawdown should be 0 at peak equity."""
        test_agent.state.peak_equity = 100000.0

        test_agent.update_equity(100000.0)

        assert test_agent.state.current_drawdown == 0.0

    def test_drawdown_calculation_with_gains(self, test_agent):
        """Drawdown at new peak should be 0."""
        test_agent.state.peak_equity = 100000.0

        test_agent.update_equity(120000.0)

        assert test_agent.state.peak_equity == 120000.0
        assert test_agent.state.current_drawdown == 0.0

    def test_drawdown_after_multiple_updates(self, test_agent):
        """Drawdown should track correctly through multiple updates."""
        # Start at 100k
        test_agent.update_equity(100000.0)
        assert test_agent.state.current_drawdown == 0.0

        # Rise to 110k
        test_agent.update_equity(110000.0)
        assert test_agent.state.peak_equity == 110000.0
        assert test_agent.state.current_drawdown == 0.0

        # Drop to 100k - 9.09% drawdown
        test_agent.update_equity(100000.0)
        expected_drawdown = (110000 - 100000) / 110000 * 100
        assert abs(test_agent.state.current_drawdown - expected_drawdown) < 0.01

    def test_large_drawdown(self, test_agent):
        """Large drawdown should be calculated correctly."""
        test_agent.state.peak_equity = 100000.0

        test_agent.update_equity(50000.0)

        assert test_agent.state.current_drawdown == 50.0


class TestWinRateProperty:
    """Test win_rate property."""

    def test_win_rate_with_no_trades(self, test_agent):
        """Win rate should be 0 with no trades (avoid division by zero)."""
        assert test_agent.state.total_trades == 0

        assert test_agent.win_rate == 0.0

    def test_win_rate_all_wins(self, test_agent):
        """Win rate should be 100 with all wins."""
        test_agent.state.total_trades = 10
        test_agent.state.winning_trades = 10

        assert test_agent.win_rate == 100.0

    def test_win_rate_all_losses(self, test_agent):
        """Win rate should be 0 with all losses."""
        test_agent.state.total_trades = 10
        test_agent.state.winning_trades = 0

        assert test_agent.win_rate == 0.0

    def test_win_rate_mixed(self, test_agent):
        """Win rate should calculate correctly with mixed results."""
        test_agent.state.total_trades = 20
        test_agent.state.winning_trades = 12

        assert test_agent.win_rate == 60.0

    def test_win_rate_fractional(self, test_agent):
        """Win rate should handle fractional percentages."""
        test_agent.state.total_trades = 3
        test_agent.state.winning_trades = 1

        assert abs(test_agent.win_rate - 33.333) < 0.01

    def test_win_rate_single_trade_win(self, test_agent):
        """Win rate should be 100 for single winning trade."""
        test_agent.state.total_trades = 1
        test_agent.state.winning_trades = 1

        assert test_agent.win_rate == 100.0

    def test_win_rate_single_trade_loss(self, test_agent):
        """Win rate should be 0 for single losing trade."""
        test_agent.state.total_trades = 1
        test_agent.state.winning_trades = 0

        assert test_agent.win_rate == 0.0


class TestRecordDecision:
    """Test record_decision method."""

    def test_decision_added_to_history(self, test_agent):
        """Decision should be added to history."""
        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            stop_loss=95.0,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Test",
            confidence=0.7,
        )

        test_agent.record_decision(decision)

        assert len(test_agent.decision_history) == 1
        assert test_agent.decision_history[0] == decision

    def test_last_decision_time_updated(self, test_agent):
        """Last decision time should be updated."""
        decision_time = datetime.now()
        decision = TradingDecision(
            timestamp=decision_time,
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            stop_loss=95.0,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Test",
            confidence=0.7,
        )

        test_agent.record_decision(decision)

        assert test_agent.state.last_decision_time == decision_time

    def test_strategy_usage_tracked(self, test_agent):
        """Strategy usage should be tracked."""
        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            stop_loss=95.0,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Test",
            confidence=0.7,
        )

        test_agent.record_decision(decision)

        assert test_agent.state.strategies_used.get("momentum") == 1

    def test_strategy_usage_accumulates(self, test_agent):
        """Strategy usage should accumulate."""
        for _ in range(3):
            decision = TradingDecision(
                timestamp=datetime.now(),
                action=ActionType.BUY,
                symbol="GOOGL",
                quantity=10,
                stop_loss=95.0,
                strategy_used=StrategyType.MOMENTUM,
                reasoning="Test",
                confidence=0.7,
            )
            test_agent.record_decision(decision)

        assert test_agent.state.strategies_used.get("momentum") == 3


class TestMostUsedStrategy:
    """Test most_used_strategy property."""

    def test_most_used_with_no_strategies(self, test_agent):
        """Should return 'none' when no strategies used."""
        assert test_agent.most_used_strategy == "none"

    def test_most_used_single_strategy(self, test_agent):
        """Should return the only strategy used."""
        test_agent.state.strategies_used = {"momentum": 5}

        assert test_agent.most_used_strategy == "momentum"

    def test_most_used_multiple_strategies(self, experienced_agent):
        """Should return the most frequently used strategy."""
        assert experienced_agent.most_used_strategy == "momentum"

    def test_most_used_tie_returns_one(self, test_agent):
        """Should return one of the tied strategies."""
        test_agent.state.strategies_used = {"momentum": 5, "mean_reversion": 5}

        # Should return one of them (implementation uses max)
        assert test_agent.most_used_strategy in ["momentum", "mean_reversion"]


class TestExecuteTool:
    """Test execute_tool method."""

    def test_execute_market_tool(self, test_agent):
        """Should route market tools to market handler."""
        test_agent.tools["market"].execute.return_value = {"price": 100.0}

        result = test_agent.execute_tool("get_stock_price", {"symbol": "GOOGL"})

        test_agent.tools["market"].execute.assert_called_once_with(
            "get_stock_price", {"symbol": "GOOGL"}
        )
        assert result == {"price": 100.0}

    def test_execute_trading_tool(self, test_agent):
        """Should route trading tools to trading handler."""
        test_agent.tools["trading"].execute.return_value = {"order_id": "123"}

        result = test_agent.execute_tool("place_order", {"symbol": "GOOGL"})

        test_agent.tools["trading"].execute.assert_called_once()
        assert result == {"order_id": "123"}

    def test_execute_analysis_tool(self, test_agent):
        """Should route analysis tools to analysis handler."""
        test_agent.tools["analysis"].execute.return_value = {"trend": "bullish"}

        result = test_agent.execute_tool("analyze_trend", {"symbol": "GOOGL"})

        test_agent.tools["analysis"].execute.assert_called_once()

    def test_execute_unknown_tool(self, test_agent):
        """Should return error for unknown tool."""
        result = test_agent.execute_tool("unknown_tool", {})

        assert "error" in result
        assert "not found" in result["error"]

    def test_tool_category_mapping_market(self, test_agent):
        """Market tools should be correctly categorized."""
        market_tools = [
            "get_stock_price",
            "get_price_history",
            "get_technical_indicators",
            "get_market_snapshot",
        ]

        for tool in market_tools:
            category = test_agent._get_tool_category(tool)
            assert category == "market"

    def test_tool_category_mapping_trading(self, test_agent):
        """Trading tools should be correctly categorized."""
        trading_tools = [
            "get_positions",
            "get_account",
            "get_orders",
            "place_order",
            "close_position",
            "cancel_order",
        ]

        for tool in trading_tools:
            category = test_agent._get_tool_category(tool)
            assert category == "trading"

    def test_tool_category_mapping_analysis(self, test_agent):
        """Analysis tools should be correctly categorized."""
        analysis_tools = [
            "get_market_context",
            "compare_stocks",
            "get_support_resistance",
            "analyze_trend",
        ]

        for tool in analysis_tools:
            category = test_agent._get_tool_category(tool)
            assert category == "analysis"


class TestGetPerformanceSummary:
    """Test get_performance_summary method."""

    def test_summary_structure(self, test_agent):
        """Summary should have all expected keys."""
        summary = test_agent.get_performance_summary()

        expected_keys = [
            "agent",
            "total_trades",
            "winning_trades",
            "losing_trades",
            "win_rate",
            "strategies_used",
            "most_used_strategy",
            "consecutive_losses",
            "peak_equity",
            "current_drawdown",
        ]

        for key in expected_keys:
            assert key in summary

    def test_summary_agent_name(self, test_agent):
        """Summary should include agent name."""
        summary = test_agent.get_performance_summary()

        assert summary["agent"] == "test_agent"

    def test_summary_with_experienced_agent(self, experienced_agent):
        """Summary should reflect agent's history."""
        summary = experienced_agent.get_performance_summary()

        assert summary["total_trades"] == 20
        assert summary["winning_trades"] == 12
        assert summary["losing_trades"] == 8
        assert summary["win_rate"] == 60.0
        assert summary["most_used_strategy"] == "momentum"

    def test_summary_win_rate_rounded(self, test_agent):
        """Win rate should be rounded to 1 decimal."""
        test_agent.state.total_trades = 3
        test_agent.state.winning_trades = 1

        summary = test_agent.get_performance_summary()

        assert summary["win_rate"] == 33.3


class TestAgentState:
    """Test AgentState dataclass."""

    def test_default_values(self):
        """AgentState should have correct defaults."""
        state = AgentState(agent_name="test")

        assert state.total_trades == 0
        assert state.winning_trades == 0
        assert state.losing_trades == 0
        assert state.strategies_used == {}
        assert state.last_decision_time is None
        assert state.consecutive_losses == 0
        assert state.peak_equity == 100_000.0
        assert state.current_drawdown == 0.0


class TestTradingDecision:
    """Test TradingDecision dataclass."""

    def test_default_values(self):
        """TradingDecision should have correct defaults."""
        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.HOLD,
        )

        assert decision.symbol is None
        assert decision.quantity is None
        assert decision.order_type == "market"
        assert decision.limit_price is None
        assert decision.stop_loss is None
        assert decision.take_profit is None
        assert decision.strategy_used == StrategyType.DEFENSIVE
        assert decision.reasoning == ""
        assert decision.confidence == 0.5
        assert decision.tool_calls == []


class TestActionType:
    """Test ActionType enum."""

    def test_action_values(self):
        """ActionType should have correct values."""
        assert ActionType.BUY.value == "buy"
        assert ActionType.SELL.value == "sell"
        assert ActionType.HOLD.value == "hold"
        assert ActionType.CLOSE.value == "close"


class TestStrategyType:
    """Test StrategyType enum."""

    def test_strategy_values(self):
        """StrategyType should have correct values."""
        assert StrategyType.MOMENTUM.value == "momentum"
        assert StrategyType.MEAN_REVERSION.value == "mean_reversion"
        assert StrategyType.TREND_FOLLOWING.value == "trend_following"
        assert StrategyType.BREAKOUT.value == "breakout"
        assert StrategyType.RANGE_TRADING.value == "range_trading"
        assert StrategyType.DEFENSIVE.value == "defensive"


class TestMarketContext:
    """Test MarketContext dataclass."""

    def test_market_context_creation(self):
        """MarketContext should be creatable with required fields."""
        context = MarketContext(
            timestamp=datetime.now(),
            symbols={"GOOGL": {"price": 150.0}},
            account={"equity": 100000.0},
            positions=[],
            recent_trades=[],
            market_condition="bullish",
        )

        assert context.symbols["GOOGL"]["price"] == 150.0
        assert context.market_condition == "bullish"


class TestRecallLearningsEdgeCases:
    """Test recall_learnings edge cases."""

    @pytest.mark.asyncio
    async def test_recall_learnings_disabled(self, test_agent):
        """Should return empty string when learning disabled."""
        with patch("agents.base_agent.LEARNING_ENABLED", False):
            context = MarketContext(
                timestamp=datetime.now(),
                symbols={"GOOGL": {"price": 150.0}},
                account={"equity": 100000.0},
                positions=[],
                recent_trades=[],
                market_condition="bullish - up",
            )

            result = await test_agent.recall_learnings(context)

            assert result == ""

    @pytest.mark.asyncio
    async def test_recall_learnings_import_error(self, test_agent):
        """Should return empty string on import error."""
        with patch("agents.base_agent.LEARNING_ENABLED", True):
            context = MarketContext(
                timestamp=datetime.now(),
                symbols={"GOOGL": {"price": 150.0}},
                account={"equity": 100000.0},
                positions=[],
                recent_trades=[],
                market_condition="bullish - up",
            )

            # The import will fail since database isn't configured
            result = await test_agent.recall_learnings(context)

            assert result == ""
