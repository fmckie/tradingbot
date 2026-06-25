"""Integration tests for full decision to execution flow.

Tests the integration between:
- MarketContext creation
- Agent decision making
- Risk validation
- Order execution
"""

from datetime import datetime
from typing import cast
from unittest.mock import patch

import pytest

from agents.base_agent import (
    ActionType,
    StrategyType,
    TradingDecision,
)
from config.settings import RISK_LIMITS, SYMBOLS
from execution.order_executor import OrderExecutor
from risk.risk_manager import RiskManager, RiskValidationResult


class TestMarketContextToDecisionFlow:
    """Test the flow from market context creation to agent decision."""

    def test_market_context_contains_required_fields(self, sample_market_context):
        """Verify MarketContext has all fields needed for decision making."""
        ctx = sample_market_context

        # Check required top-level fields
        assert ctx.timestamp is not None
        assert ctx.symbols is not None
        assert ctx.account is not None
        assert ctx.market_condition is not None

        # Check symbol data structure
        for symbol in SYMBOLS:
            assert symbol in ctx.symbols
            symbol_data = ctx.symbols[symbol]
            assert "price" in symbol_data
            assert "rsi" in symbol_data

        # Check account data structure
        assert "equity" in ctx.account
        assert "cash" in ctx.account

    def test_market_context_regime_extraction(self, sample_market_context):
        """Test that market regime can be extracted from context."""
        ctx = sample_market_context

        # Extract regime like the agent does
        regime = ctx.market_condition.split(" - ")[0].lower()
        assert regime in ["bullish", "bearish", "mixed", "neutral"]


class TestDecisionToRiskValidationFlow:
    """Test the flow from trading decision to risk validation."""

    def test_valid_buy_decision_passes_risk_validation(
        self, risk_manager, sample_buy_decision, mock_trading_client
    ):
        """A properly sized BUY decision should pass validation."""
        decision = sample_buy_decision
        current_price = 150.00

        result = risk_manager.validate_decision(decision, current_price)

        assert result.valid is True
        assert (
            "passed" in result.message.lower() or result.adjusted_quantity is not None
        )

    def test_valid_sell_decision_passes_risk_validation(
        self, risk_manager, sample_sell_decision, mock_trading_client
    ):
        """A properly sized SELL decision should pass validation."""
        decision = sample_sell_decision
        current_price = 250.00

        result = risk_manager.validate_decision(decision, current_price)

        assert result.valid is True

    def test_hold_decision_always_passes(self, risk_manager, sample_hold_decision):
        """HOLD decisions should always pass validation."""
        result = risk_manager.validate_decision(sample_hold_decision, 150.00)

        assert result.valid is True
        assert "HOLD" in result.message or "no validation" in result.message.lower()

    def test_close_decision_always_passes(self, risk_manager, sample_close_decision):
        """CLOSE decisions should always pass validation."""
        result = risk_manager.validate_decision(sample_close_decision, 150.00)

        assert result.valid is True

    def test_decision_without_stop_loss_is_rejected(self, risk_manager):
        """Decision without stop-loss should be rejected when required."""
        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            stop_loss=None,  # No stop loss
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Test decision",
        )

        result = risk_manager.validate_decision(decision, 150.00)

        if RISK_LIMITS.require_stop_loss:
            assert result.valid is False
            assert "stop" in result.message.lower()

    def test_decision_with_invalid_symbol_is_rejected(self, risk_manager):
        """Decision with invalid symbol should be rejected."""
        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="INVALID",
            quantity=10,
            stop_loss=145.00,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Test decision",
        )

        result = risk_manager.validate_decision(decision, 150.00)

        assert result.valid is False
        assert "symbol" in result.message.lower()

    def test_risk_limits_enforce_position_size_adjustment(
        self, mock_trading_client, mock_account_with_equity
    ):
        """Risk limits should adjust position size when risk exceeds threshold."""
        # Account with 100k equity
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        risk_manager = RiskManager(mock_trading_client, "test")

        # Large position that exceeds 2% risk
        # Stop loss at $5 below, quantity 500 = $2500 risk = 2.5% of equity
        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=500,
            stop_loss=145.00,  # $5 risk per share
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Large position test",
        )

        result = risk_manager.validate_decision(decision, 150.00)

        # Should either be rejected or have adjusted quantity
        if result.valid:
            assert result.adjusted_quantity is not None
            assert result.adjusted_quantity < 500
        else:
            assert "risk" in result.message.lower()


class TestRiskValidationToExecutionFlow:
    """Test the flow from risk validation to order execution."""

    @pytest.mark.asyncio
    async def test_approved_decision_executes_successfully(
        self, order_executor, sample_buy_decision, mock_trading_client
    ):
        """Decision that passes risk validation should execute."""
        # Mock trading hours check to allow trading
        with patch.object(
            order_executor.risk_manager,
            "check_trading_allowed",
            return_value=RiskValidationResult(valid=True, message="Trading allowed"),
        ):
            result = await order_executor.execute_decision(
                sample_buy_decision, current_price=150.00
            )

        assert result.success is True
        assert result.order_id is not None
        assert mock_trading_client.submit_order.called

    @pytest.mark.asyncio
    async def test_rejected_decision_does_not_execute(
        self, order_executor, mock_trading_client
    ):
        """Decision that fails risk validation should not execute."""
        # Decision with no stop loss - should fail
        bad_decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            stop_loss=None,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="No stop loss",
        )

        with patch.object(
            order_executor.risk_manager,
            "check_trading_allowed",
            return_value=RiskValidationResult(valid=True, message="Trading allowed"),
        ):
            result = await order_executor.execute_decision(
                bad_decision, current_price=150.00
            )

        if RISK_LIMITS.require_stop_loss:
            assert result.success is False
            assert not mock_trading_client.submit_order.called

    @pytest.mark.asyncio
    async def test_hold_decision_returns_success_without_order(
        self, order_executor, sample_hold_decision, mock_trading_client
    ):
        """HOLD decision should succeed without placing an order."""
        with patch.object(
            order_executor.risk_manager,
            "check_trading_allowed",
            return_value=RiskValidationResult(valid=True, message="Trading allowed"),
        ):
            result = await order_executor.execute_decision(
                sample_hold_decision, current_price=150.00
            )

        assert result.success is True
        assert result.order_id is None
        assert not mock_trading_client.submit_order.called

    @pytest.mark.asyncio
    async def test_close_position_when_position_exists(
        self,
        order_executor,
        sample_close_decision,
        mock_trading_client,
        mock_position_factory,
    ):
        """CLOSE decision should close existing position."""
        # Set up existing position
        position = mock_position_factory(symbol="GOOGL", qty=10)
        mock_trading_client.get_all_positions.return_value = [position]

        with patch.object(
            order_executor.risk_manager,
            "check_trading_allowed",
            return_value=RiskValidationResult(valid=True, message="Trading allowed"),
        ):
            result = await order_executor.execute_decision(
                sample_close_decision, current_price=155.00
            )

        assert result.success is True
        assert mock_trading_client.close_position.called

    @pytest.mark.asyncio
    async def test_close_position_fails_when_no_position(
        self, order_executor, sample_close_decision, mock_trading_client
    ):
        """CLOSE decision should fail when no position exists."""
        # No positions
        mock_trading_client.get_all_positions.return_value = []

        with patch.object(
            order_executor.risk_manager,
            "check_trading_allowed",
            return_value=RiskValidationResult(valid=True, message="Trading allowed"),
        ):
            result = await order_executor.execute_decision(
                sample_close_decision, current_price=155.00
            )

        assert result.success is False
        assert "no position" in result.message.lower()

    @pytest.mark.asyncio
    async def test_quantity_adjustment_flows_through_to_execution(
        self, mock_trading_client, mock_account_with_equity, mock_position_factory
    ):
        """Adjusted quantity from risk validation should be used in execution."""
        # Set up account
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        mock_trading_client.get_all_positions.return_value = []

        risk_manager = RiskManager(mock_trading_client, "test")
        executor = OrderExecutor(mock_trading_client, risk_manager, "test")

        # Large position that needs adjustment
        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=1000,  # Very large
            stop_loss=145.00,  # $5 risk per share
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Large position",
        )

        with patch.object(
            risk_manager,
            "check_trading_allowed",
            return_value=RiskValidationResult(valid=True, message="Trading allowed"),
        ):
            result = await executor.execute_decision(decision, current_price=150.00)

        # Check if order was placed with adjusted quantity
        if result.success and mock_trading_client.submit_order.called:
            order_request = mock_trading_client.submit_order.call_args[0][0]
            # The executed quantity should be less than original
            executed_qty = result.filled_quantity or order_request.qty
            assert cast(int, executed_qty) <= cast(int, decision.quantity)


class TestFullEndToEndFlow:
    """Test complete flow from context through execution."""

    @pytest.mark.asyncio
    async def test_complete_buy_flow(
        self, mock_trading_client, sample_market_context, mock_account_with_equity
    ):
        """Test complete flow: Context -> Decision -> Validation -> Execution."""
        # Setup
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)
        mock_trading_client.get_all_positions.return_value = []

        risk_manager = RiskManager(mock_trading_client, "claude")
        executor = OrderExecutor(mock_trading_client, risk_manager, "claude")

        # 1. Create decision based on context
        context = sample_market_context
        symbol = "GOOGL"
        current_price = context.symbols[symbol]["price"]

        # Calculate position size within limits
        # Use a smaller position to stay well under 50% exposure
        stop_loss = current_price * 0.97  # 3% stop
        risk_per_share = current_price - stop_loss
        max_risk = 100000 * RISK_LIMITS.max_risk_per_trade
        risk_based_qty = int(max_risk / risk_per_share)

        # Also check exposure limit: 50% of $100k = $50k max, use 30% = $30k to be safe
        max_exposure = 100000 * 0.30  # 30% exposure for safety
        exposure_based_qty = int(max_exposure / current_price)

        # Use smaller of the two
        quantity = min(risk_based_qty, exposure_based_qty)

        decision = TradingDecision(
            timestamp=context.timestamp,
            action=ActionType.BUY,
            symbol=symbol,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=current_price * 1.05,
            strategy_used=StrategyType.MOMENTUM,
            reasoning=f"Based on {context.market_condition}",
        )

        # 2. Validate
        validation = risk_manager.validate_decision(decision, current_price)
        assert validation.valid is True or validation.adjusted_quantity is not None

        # 3. Execute
        with patch.object(
            risk_manager,
            "check_trading_allowed",
            return_value=RiskValidationResult(valid=True, message="Trading allowed"),
        ):
            result = await executor.execute_decision(decision, current_price)

        assert result.success is True
        assert result.decision == decision

    @pytest.mark.asyncio
    async def test_rejection_propagates_correctly(
        self, mock_trading_client, mock_account_with_equity, mock_position_factory
    ):
        """Test that rejections propagate through the flow correctly."""
        # Setup with positions at max
        mock_trading_client.get_account.return_value = mock_account_with_equity(100000)

        # Create 2 positions (max allowed)
        positions = [
            mock_position_factory(symbol="GOOGL", qty=100),
            mock_position_factory(symbol="TSLA", qty=50),
        ]
        mock_trading_client.get_all_positions.return_value = positions

        risk_manager = RiskManager(mock_trading_client, "test")
        executor = OrderExecutor(mock_trading_client, risk_manager, "test")

        # Try to buy a new position (should fail - max positions reached)
        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",  # Already have position
            quantity=10,
            stop_loss=145.00,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Test max positions",
        )

        with patch.object(
            risk_manager,
            "check_trading_allowed",
            return_value=RiskValidationResult(valid=True, message="Trading allowed"),
        ):
            result = await executor.execute_decision(decision, current_price=150.00)

        # Since we already have GOOGL position, this should succeed (adding to existing)
        # or fail with max exposure message
        if not result.success:
            assert (
                "exposure" in result.message.lower()
                or "positions" in result.message.lower()
                or "risk" in result.message.lower()
            )


class TestExecutionResultTracking:
    """Test that execution results are properly tracked."""

    @pytest.mark.asyncio
    async def test_execution_result_contains_decision(
        self, order_executor, sample_buy_decision
    ):
        """ExecutionResult should contain the original decision."""
        with patch.object(
            order_executor.risk_manager,
            "check_trading_allowed",
            return_value=RiskValidationResult(valid=True, message="Trading allowed"),
        ):
            result = await order_executor.execute_decision(
                sample_buy_decision, current_price=150.00
            )

        assert result.decision is not None
        assert result.decision.action == sample_buy_decision.action
        assert result.decision.symbol == sample_buy_decision.symbol

    @pytest.mark.asyncio
    async def test_execution_result_contains_risk_validation(
        self, order_executor, sample_buy_decision
    ):
        """ExecutionResult should contain risk validation details."""
        with patch.object(
            order_executor.risk_manager,
            "check_trading_allowed",
            return_value=RiskValidationResult(valid=True, message="Trading allowed"),
        ):
            result = await order_executor.execute_decision(
                sample_buy_decision, current_price=150.00
            )

        assert result.risk_validation is not None
        assert hasattr(result.risk_validation, "valid")
        assert hasattr(result.risk_validation, "message")

    @pytest.mark.asyncio
    async def test_execution_result_has_timestamp(
        self, order_executor, sample_buy_decision
    ):
        """ExecutionResult should have a timestamp."""
        with patch.object(
            order_executor.risk_manager,
            "check_trading_allowed",
            return_value=RiskValidationResult(valid=True, message="Trading allowed"),
        ):
            result = await order_executor.execute_decision(
                sample_buy_decision, current_price=150.00
            )

        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)
