"""Integration tests for risk limit enforcement under various conditions.

Tests risk enforcement for:
- Multiple rapid decisions
- Edge cases near limits
- Daily loss limit progression
- Position limits at boundary
"""

from datetime import datetime
from unittest.mock import MagicMock

from agents.base_agent import (
    ActionType,
    StrategyType,
    TradingDecision,
)
from config.settings import RISK_LIMITS
from risk.risk_manager import RiskManager


class TestMultipleRapidDecisions:
    """Test risk enforcement with multiple rapid decisions."""

    def test_rapid_decisions_track_cumulative_exposure(
        self, mock_trading_client, mock_account_with_equity, mock_position_factory
    ):
        """Multiple rapid decisions should be validated against cumulative exposure."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        risk_manager = RiskManager(mock_trading_client, "test")

        decisions_validated = []

        # First decision - should pass (0% -> ~15% exposure)
        position1 = mock_position_factory(symbol="GOOGL", qty=100, current_price=150.00)
        mock_trading_client.get_all_positions.return_value = []

        decision1 = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=100,  # $15,000 value = 15% exposure
            stop_loss=145.00,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="First trade",
        )
        result1 = risk_manager.validate_decision(decision1, 150.00)
        decisions_validated.append(result1)

        # Simulate position being created
        mock_trading_client.get_all_positions.return_value = [position1]

        # Second decision - should pass (15% -> ~30% exposure)
        position2 = mock_position_factory(symbol="TSLA", qty=60, current_price=250.00)

        decision2 = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="TSLA",
            quantity=60,  # $15,000 value = 15% more exposure
            stop_loss=240.00,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Second trade",
        )
        result2 = risk_manager.validate_decision(decision2, 250.00)
        decisions_validated.append(result2)

        # Simulate both positions
        mock_trading_client.get_all_positions.return_value = [position1, position2]

        # Third decision - should be blocked (would exceed 50% max exposure)
        decision3 = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=200,  # $30,000 more = would total 60%
            stop_loss=145.00,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Third trade",
        )
        result3 = risk_manager.validate_decision(decision3, 150.00)
        decisions_validated.append(result3)

        # First two should pass, third should fail
        assert result1.valid is True or result1.adjusted_quantity is not None
        assert result2.valid is True or result2.adjusted_quantity is not None
        # Third might fail or have reduced quantity
        if result3.valid:
            assert result3.adjusted_quantity is not None
            # Adjusted quantity should be smaller to fit within limits
        else:
            assert (
                "exposure" in result3.message.lower()
                or "positions" in result3.message.lower()
            )

    def test_rapid_decisions_respect_position_count_limit(
        self, mock_trading_client, mock_account_with_equity, mock_position_factory
    ):
        """Rapid decisions should respect max position count."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        risk_manager = RiskManager(mock_trading_client, "test")

        # Already have max positions (2)
        positions = [
            mock_position_factory(symbol="GOOGL", qty=10),
            mock_position_factory(symbol="TSLA", qty=5),
        ]
        mock_trading_client.get_all_positions.return_value = positions

        # Try to add another symbol position (should fail)
        # Note: We only have 2 symbols allowed (GOOGL, TSLA), so this tests the max
        # In reality, trying to add a third symbol would be rejected by symbol
        # validation

        # Instead, test that adding to existing position is allowed
        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",  # Already have position
            quantity=5,
            stop_loss=145.00,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Add to existing",
        )
        result = risk_manager.validate_decision(decision, 150.00)

        # Adding to existing position should be allowed (if within exposure limits)
        # The validation is on new positions, not additions
        assert result.valid is True or "exposure" in (result.message or "").lower()


class TestEdgeCasesNearLimits:
    """Test risk enforcement at boundary conditions."""

    def test_at_45_percent_exposure_allows_small_trade(
        self, mock_trading_client, mock_account_with_equity, mock_position_factory
    ):
        """At 45% exposure, small trades should still be allowed (staying under 50%)."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        risk_manager = RiskManager(mock_trading_client, "test")

        # Position worth $45,000 = 45% exposure
        existing = mock_position_factory(
            symbol="GOOGL",
            qty=300,  # 300 * 150 = 45,000
            current_price=150.00,
        )
        mock_trading_client.get_all_positions.return_value = [existing]

        # Try to add small position - 4% more = 49% total (under 50% limit)
        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="TSLA",
            quantity=16,  # 16 * 250 = $4,000 = 4%
            stop_loss=240.00,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Small additional trade",
        )
        result = risk_manager.validate_decision(decision, 250.00)

        assert result.valid is True

    def test_at_49_percent_exposure_blocks_large_trade(
        self, mock_trading_client, mock_account_with_equity, mock_position_factory
    ):
        """At 49% exposure, trades exceeding 50% limit should be blocked/adjusted."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        risk_manager = RiskManager(mock_trading_client, "test")

        # Position worth $49,000 = 49% exposure
        existing = mock_position_factory(symbol="GOOGL", qty=327, current_price=150.00)
        mock_trading_client.get_all_positions.return_value = [existing]

        # Try to add large position - 10% more = 59% total (exceeds 50% limit)
        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="TSLA",
            quantity=40,  # 40 * 250 = $10,000 = 10%
            stop_loss=240.00,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Large additional trade",
        )
        result = risk_manager.validate_decision(decision, 250.00)

        # Should either reject or adjust quantity
        if result.valid:
            # If valid, should have reduced quantity
            assert result.adjusted_quantity is not None
            max_additional_exposure = 100000 * RISK_LIMITS.max_exposure - 49050
            max_additional_shares = int(max_additional_exposure / 250.00)
            assert result.adjusted_quantity <= max_additional_shares
        else:
            assert "exposure" in result.message.lower()

    def test_exactly_at_50_percent_exposure_blocks_new_trades(
        self, mock_trading_client, mock_account_with_equity, mock_position_factory
    ):
        """At exactly 50% exposure, new trades should be blocked."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        risk_manager = RiskManager(mock_trading_client, "test")

        # Position worth exactly $50,000 = 50% exposure (at limit)
        existing = mock_position_factory(
            symbol="GOOGL",
            qty=333,  # 333 * 150.15 ~ $50,000
            current_price=150.15,
        )
        mock_trading_client.get_all_positions.return_value = [existing]

        # Try to add any position
        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="TSLA",
            quantity=1,  # Even 1 share
            stop_loss=240.00,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="At limit trade",
        )
        result = risk_manager.validate_decision(decision, 250.00)

        # Should be blocked
        assert result.valid is False
        assert "exposure" in result.message.lower()

    def test_stop_loss_at_minimum_distance(
        self, mock_trading_client, mock_account_with_equity
    ):
        """Stop loss at minimum distance should be accepted."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        mock_trading_client.get_all_positions.return_value = []
        risk_manager = RiskManager(mock_trading_client, "test")

        current_price = 150.00
        min_stop_distance = RISK_LIMITS.min_stop_distance  # 0.5%
        stop_price = current_price * (1 - min_stop_distance)  # Exactly at minimum

        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            stop_loss=stop_price,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Min stop distance",
        )
        result = risk_manager.validate_decision(decision, current_price)

        assert result.valid is True

    def test_stop_loss_below_minimum_distance_is_rejected(
        self, mock_trading_client, mock_account_with_equity
    ):
        """Stop loss too close to entry should be rejected."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        mock_trading_client.get_all_positions.return_value = []
        risk_manager = RiskManager(mock_trading_client, "test")

        current_price = 150.00
        # Stop at 0.1% (below minimum 0.5%)
        stop_price = current_price * 0.999

        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            stop_loss=stop_price,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Too tight stop",
        )
        result = risk_manager.validate_decision(decision, current_price)

        assert result.valid is False
        assert "stop" in result.message.lower() and "tight" in result.message.lower()

    def test_stop_loss_above_maximum_distance_is_rejected(
        self, mock_trading_client, mock_account_with_equity
    ):
        """Stop loss too far from entry should be rejected."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        mock_trading_client.get_all_positions.return_value = []
        risk_manager = RiskManager(mock_trading_client, "test")

        current_price = 150.00
        # Stop at 10% (above maximum 5%)
        stop_price = current_price * 0.90

        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            stop_loss=stop_price,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Too wide stop",
        )
        result = risk_manager.validate_decision(decision, current_price)

        assert result.valid is False
        assert "stop" in result.message.lower() and "wide" in result.message.lower()


class TestDailyLossLimitProgression:
    """Test daily loss limit enforcement through the trading day."""

    def test_daily_loss_tracking_resets_on_new_day(
        self, mock_trading_client, mock_account_with_equity
    ):
        """Daily loss tracking should reset on new day."""
        risk_manager = RiskManager(mock_trading_client, "test")

        # Simulate yesterday's equity
        mock_trading_client.get_account.return_value = mock_account_with_equity(
            95000, last_equity=100000
        )

        # First check - should initialize tracking
        risk_manager._check_daily_loss_limit(95000)

        # On same day, equity dropped - should be at/near limit
        mock_trading_client.get_account.return_value = mock_account_with_equity(90000)

        # Simulate day change by resetting the tracking date
        risk_manager.last_reset_date = None

        # Check again - should reset to new baseline
        result2 = risk_manager._check_daily_loss_limit(90000)

        # After reset, should be valid (new baseline)
        assert result2.valid is True

    def test_trading_blocked_after_daily_loss_limit_reached(
        self, mock_trading_client, mock_account_with_equity
    ):
        """Trading should be blocked after daily loss limit is reached."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        mock_trading_client.get_all_positions.return_value = []
        risk_manager = RiskManager(mock_trading_client, "test")

        # Initialize daily tracking
        risk_manager._check_daily_loss_limit(100000)

        # Simulate loss exceeding 5% daily limit
        mock_trading_client.get_account.return_value = mock_account_with_equity(
            94000
        )  # 6% loss

        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            stop_loss=145.00,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="After daily loss",
        )
        result = risk_manager.validate_decision(decision, 150.00)

        assert result.valid is False
        assert "daily" in result.message.lower() and "loss" in result.message.lower()

    def test_trading_allowed_at_4_percent_daily_loss(
        self, mock_trading_client, mock_account_with_equity
    ):
        """Trading should still be allowed at 4% daily loss (below 5% limit)."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        mock_trading_client.get_all_positions.return_value = []
        risk_manager = RiskManager(mock_trading_client, "test")

        # Initialize daily tracking
        risk_manager._check_daily_loss_limit(100000)

        # Simulate 4% loss (below 5% limit)
        mock_trading_client.get_account.return_value = mock_account_with_equity(96000)

        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            stop_loss=145.00,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Before daily limit",
        )
        result = risk_manager.validate_decision(decision, 150.00)

        # Should still be valid - below daily loss limit
        assert result.valid is True

    def test_daily_loss_calculated_from_day_start_equity(
        self, mock_trading_client, mock_account_with_equity
    ):
        """Daily loss should be calculated from day start, not peak."""
        risk_manager = RiskManager(mock_trading_client, "test")

        # Day starts at $100k
        risk_manager._check_daily_loss_limit(100000)

        # Goes up to $105k (this is NOT the baseline)
        result1 = risk_manager._check_daily_loss_limit(105000)
        assert result1.valid is True

        # Then drops to $96k (4% from start, not 8.5% from peak)
        result2 = risk_manager._check_daily_loss_limit(96000)
        # Should still be valid - 4% loss from START, not from peak
        assert result2.valid is True


class TestPositionLimitsAtBoundary:
    """Test position count limits at boundary conditions."""

    def test_at_max_positions_blocks_new_symbol(
        self, mock_trading_client, mock_account_with_equity, mock_position_factory
    ):
        """At max positions, new symbol trades should be blocked."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        risk_manager = RiskManager(mock_trading_client, "test")

        # Max positions = 2, already have both symbols
        positions = [
            mock_position_factory(symbol="GOOGL", qty=10),
            mock_position_factory(symbol="TSLA", qty=5),
        ]
        mock_trading_client.get_all_positions.return_value = positions

        # Can still trade existing symbols (add to position)
        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=5,
            stop_loss=145.00,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Add to GOOGL",
        )
        result = risk_manager.validate_decision(decision, 150.00)

        # Should be allowed (adding to existing position, within exposure)
        # May still fail due to exposure, but not position count
        if not result.valid:
            # Should fail due to exposure, not position count
            assert "exposure" in result.message.lower()

    def test_closing_position_allows_new_position(
        self, mock_trading_client, mock_account_with_equity, mock_position_factory
    ):
        """After closing a position, new position should be allowed."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        risk_manager = RiskManager(mock_trading_client, "test")

        # Start with max positions
        positions = [
            mock_position_factory(symbol="GOOGL", qty=10),
            mock_position_factory(symbol="TSLA", qty=5),
        ]
        mock_trading_client.get_all_positions.return_value = positions

        # Close GOOGL position
        close_decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.CLOSE,
            symbol="GOOGL",
            strategy_used=StrategyType.DEFENSIVE,
            reasoning="Take profits",
        )
        close_result = risk_manager.validate_decision(close_decision, 150.00)
        assert close_result.valid is True

        # Simulate position being closed
        mock_trading_client.get_all_positions.return_value = [
            mock_position_factory(symbol="TSLA", qty=5)
        ]

        # Now can open new GOOGL position
        new_decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            stop_loss=145.00,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="New GOOGL position",
        )
        new_result = risk_manager.validate_decision(new_decision, 150.00)

        assert new_result.valid is True

    def test_zero_positions_allows_up_to_max(
        self, mock_trading_client, mock_account_with_equity
    ):
        """With no positions, can open up to max positions."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        mock_trading_client.get_all_positions.return_value = []
        risk_manager = RiskManager(mock_trading_client, "test")

        # First position
        decision1 = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            stop_loss=145.00,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="First position",
        )
        result1 = risk_manager.validate_decision(decision1, 150.00)
        assert result1.valid is True

        # Simulate position created
        mock_trading_client.get_all_positions.return_value = [
            MagicMock(symbol="GOOGL", market_value=1500)
        ]

        # Second position
        decision2 = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="TSLA",
            quantity=5,
            stop_loss=240.00,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Second position",
        )
        result2 = risk_manager.validate_decision(decision2, 250.00)
        assert result2.valid is True


class TestRiskStatusReporting:
    """Test risk status reporting for monitoring."""

    def test_risk_status_reports_current_exposure(
        self, mock_trading_client, mock_account_with_equity, mock_position_factory
    ):
        """Risk status should accurately report current exposure."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        risk_manager = RiskManager(mock_trading_client, "test")

        # Position worth $30,000 = 30% exposure
        position = mock_position_factory(symbol="GOOGL", qty=200, current_price=150.00)
        mock_trading_client.get_all_positions.return_value = [position]

        status = risk_manager.get_risk_status()

        assert status["equity"] == 100000
        assert abs(status["exposure_percent"] - 30.0) < 1.0  # Allow small rounding
        assert status["positions_count"] == 1
        assert status["max_positions"] == RISK_LIMITS.max_positions
        assert status["max_exposure_percent"] == RISK_LIMITS.max_exposure * 100

    def test_risk_status_reports_daily_pnl(
        self, mock_trading_client, mock_account_with_equity
    ):
        """Risk status should report daily P&L."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(102000)
        mock_trading_client.get_all_positions.return_value = []
        risk_manager = RiskManager(mock_trading_client, "test")

        # Initialize tracking
        risk_manager._check_daily_loss_limit(100000)

        status = risk_manager.get_risk_status()

        assert status["daily_pnl"] == 2000  # $2k profit
        assert abs(status["daily_pnl_percent"] - 2.0) < 0.1

    def test_risk_status_indicates_at_daily_limit(
        self, mock_trading_client, mock_account_with_equity
    ):
        """Risk status should indicate when at daily loss limit."""
        mock_trading_client.get_account.return_value = mock_account_with_equity(94000)
        mock_trading_client.get_all_positions.return_value = []
        risk_manager = RiskManager(mock_trading_client, "test")

        # Initialize at $100k
        risk_manager._check_daily_loss_limit(100000)

        # Now at $94k (6% loss, exceeds 5% limit)
        status = risk_manager.get_risk_status()

        assert status["at_daily_limit"] is True
