"""
Comprehensive tests for OrderExecutor execution logic.

Tests the execute_decision() method and related order handling
with mocked Alpaca API client.
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass

from agents.base_agent import ActionType, StrategyType, TradingDecision
from execution.order_executor import OrderExecutor, ExecutionResult
from risk.risk_manager import RiskManager, RiskValidationResult


@dataclass
class MockOrder:
    """Mock Alpaca order object."""
    id: str = "order-123"
    symbol: str = "GOOGL"
    _side: str = "buy"
    qty: int = 10
    filled_qty: int = 0
    _type: str = "market"
    _status: str = "new"
    created_at: datetime | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    @property
    def side(self):
        return MagicMock(value=self._side)

    @side.setter
    def side(self, value):
        self._side = value

    @property
    def type(self):
        return MagicMock(value=self._type)

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def status(self):
        return MagicMock(value=self._status)

    @status.setter
    def status(self, value):
        self._status = value


@dataclass
class MockPosition:
    """Mock Alpaca position object."""
    symbol: str = "GOOGL"
    qty: str = "10"
    market_value: str = "1850.00"
    avg_entry_price: str = "180.00"


class TestOrderExecutorExecuteDecision:
    """Test suite for OrderExecutor.execute_decision() method."""

    @pytest.fixture
    def mock_trading_client(self):
        """Create mock Alpaca trading client."""
        client = MagicMock()
        # Default account response
        account = MagicMock()
        account.equity = "100000.00"
        client.get_account.return_value = account
        # Default empty positions
        client.get_all_positions.return_value = []
        return client

    @pytest.fixture
    def mock_risk_manager(self, mock_trading_client):
        """Create mock risk manager."""
        risk_manager = MagicMock(spec=RiskManager)
        # Default: allow trading and pass validation
        risk_manager.check_trading_allowed.return_value = RiskValidationResult(
            valid=True, message="Trading allowed"
        )
        risk_manager.validate_decision.return_value = RiskValidationResult(
            valid=True, message="All risk checks passed"
        )
        return risk_manager

    @pytest.fixture
    def executor(self, mock_trading_client, mock_risk_manager):
        """Create OrderExecutor with mocked dependencies."""
        return OrderExecutor(
            trading_client=mock_trading_client,
            risk_manager=mock_risk_manager,
            agent_name="test_agent"
        )

    @pytest.fixture
    def timestamp(self):
        """Standard timestamp for tests."""
        return datetime(2024, 1, 15, 10, 30, 0)

    def _make_decision(
        self,
        action: ActionType = ActionType.BUY,
        symbol: str = "GOOGL",
        quantity: int = 10,
        stop_loss: float | None = 175.00,
        take_profit: float | None = None,
        timestamp: datetime | None = None
    ) -> TradingDecision:
        """Helper to create TradingDecision."""
        if timestamp is None:
            timestamp = datetime(2024, 1, 15, 10, 30, 0)
        return TradingDecision(
            timestamp=timestamp,
            action=action,
            symbol=symbol,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_used=StrategyType.MOMENTUM,
            reasoning="Test decision"
        )

    # ==================== HOLD ACTION TESTS ====================

    @pytest.mark.asyncio
    async def test_hold_action_returns_success(self, executor, mock_risk_manager):
        """Test HOLD action returns success without placing orders."""
        decision = self._make_decision(action=ActionType.HOLD)

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is True
        assert result.order_id is None
        assert "HOLD" in result.message
        executor.client.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_hold_action_bypasses_order_placement(self, executor, mock_trading_client):
        """Test HOLD doesn't call submit_order."""
        decision = self._make_decision(action=ActionType.HOLD)

        await executor.execute_decision(decision, current_price=185.00)

        mock_trading_client.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_hold_preserves_decision_in_result(self, executor):
        """Test HOLD result contains original decision."""
        decision = self._make_decision(action=ActionType.HOLD)

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.decision == decision

    # ==================== CLOSE ACTION TESTS ====================

    @pytest.mark.asyncio
    async def test_close_action_closes_existing_position(self, executor, mock_trading_client):
        """Test CLOSE action closes an existing position."""
        # Setup position
        position = MockPosition(symbol="GOOGL", qty="15")
        mock_trading_client.get_all_positions.return_value = [position]
        mock_order = MagicMock()
        mock_order.id = "close-order-123"
        mock_trading_client.close_position.return_value = mock_order

        decision = self._make_decision(action=ActionType.CLOSE, symbol="GOOGL")

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is True
        assert result.order_id == "close-order-123"
        mock_trading_client.close_position.assert_called_once_with("GOOGL")

    @pytest.mark.asyncio
    async def test_close_action_no_position_fails(self, executor, mock_trading_client):
        """Test CLOSE fails when no position exists."""
        mock_trading_client.get_all_positions.return_value = []

        decision = self._make_decision(action=ActionType.CLOSE, symbol="GOOGL")

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is False
        assert "No position found" in result.message

    @pytest.mark.asyncio
    async def test_close_action_wrong_symbol_fails(self, executor, mock_trading_client):
        """Test CLOSE fails when position is for different symbol."""
        position = MockPosition(symbol="TSLA", qty="10")
        mock_trading_client.get_all_positions.return_value = [position]

        decision = self._make_decision(action=ActionType.CLOSE, symbol="GOOGL")

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is False
        assert "No position found" in result.message

    @pytest.mark.asyncio
    async def test_close_action_no_symbol_fails(self, executor):
        """Test CLOSE without symbol fails gracefully."""
        decision = self._make_decision(action=ActionType.CLOSE, symbol=None)

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is False
        assert "No symbol specified" in result.message

    @pytest.mark.asyncio
    async def test_close_reports_quantity_closed(self, executor, mock_trading_client):
        """Test CLOSE result includes quantity closed."""
        position = MockPosition(symbol="GOOGL", qty="25")
        mock_trading_client.get_all_positions.return_value = [position]
        mock_order = MagicMock()
        mock_order.id = "close-order-123"
        mock_trading_client.close_position.return_value = mock_order

        decision = self._make_decision(action=ActionType.CLOSE, symbol="GOOGL")

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.filled_quantity == 25

    @pytest.mark.asyncio
    async def test_close_handles_api_error(self, executor, mock_trading_client):
        """Test CLOSE handles API errors gracefully."""
        position = MockPosition(symbol="GOOGL", qty="10")
        mock_trading_client.get_all_positions.return_value = [position]
        mock_trading_client.close_position.side_effect = Exception("API Error")

        decision = self._make_decision(action=ActionType.CLOSE, symbol="GOOGL")

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is False
        assert "Failed to close position" in result.message
        assert "API Error" in result.message

    # ==================== BUY ACTION TESTS ====================

    @pytest.mark.asyncio
    async def test_buy_action_places_market_order(self, executor, mock_trading_client):
        """Test BUY places a market order."""
        mock_order = MagicMock()
        mock_order.id = "buy-order-123"
        mock_trading_client.submit_order.return_value = mock_order

        decision = self._make_decision(
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            stop_loss=175.00
        )

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is True
        assert result.order_id == "buy-order-123"
        mock_trading_client.submit_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_buy_with_stop_loss_uses_oto(self, executor, mock_trading_client):
        """Test BUY with stop-loss only uses OTO order class."""
        mock_order = MagicMock()
        mock_order.id = "buy-order-123"
        mock_trading_client.submit_order.return_value = mock_order

        decision = self._make_decision(
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            stop_loss=175.00,
            take_profit=None
        )

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is True
        # Verify OTO order was submitted
        call_args = mock_trading_client.submit_order.call_args
        order_request = call_args[0][0]
        # The request should have stop_loss
        assert hasattr(order_request, 'stop_loss') or 'stop_loss' in str(order_request)

    @pytest.mark.asyncio
    async def test_buy_with_bracket_order(self, executor, mock_trading_client):
        """Test BUY with both stop-loss and take-profit creates bracket order."""
        mock_order = MagicMock()
        mock_order.id = "bracket-order-123"
        mock_trading_client.submit_order.return_value = mock_order

        decision = self._make_decision(
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            stop_loss=175.00,
            take_profit=200.00
        )

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is True
        assert result.order_id == "bracket-order-123"
        # Verify bracket order was submitted
        call_args = mock_trading_client.submit_order.call_args
        order_request = call_args[0][0]
        assert hasattr(order_request, 'take_profit') or 'take_profit' in str(order_request)

    @pytest.mark.asyncio
    async def test_buy_reports_filled_quantity(self, executor, mock_trading_client):
        """Test BUY result includes filled quantity."""
        mock_order = MagicMock()
        mock_order.id = "buy-order-123"
        mock_trading_client.submit_order.return_value = mock_order

        decision = self._make_decision(
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=15
        )

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.filled_quantity == 15

    @pytest.mark.asyncio
    async def test_buy_handles_api_error(self, executor, mock_trading_client):
        """Test BUY handles API errors gracefully."""
        mock_trading_client.submit_order.side_effect = Exception("Insufficient buying power")

        decision = self._make_decision(action=ActionType.BUY)

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is False
        assert "Order execution failed" in result.message
        assert "Insufficient buying power" in result.message

    # ==================== SELL ACTION TESTS ====================

    @pytest.mark.asyncio
    async def test_sell_action_places_market_order(self, executor, mock_trading_client):
        """Test SELL places a market order."""
        mock_order = MagicMock()
        mock_order.id = "sell-order-123"
        mock_trading_client.submit_order.return_value = mock_order

        decision = self._make_decision(
            action=ActionType.SELL,
            symbol="TSLA",
            quantity=5,
            stop_loss=260.00
        )

        result = await executor.execute_decision(decision, current_price=250.00)

        assert result.success is True
        assert result.order_id == "sell-order-123"

    @pytest.mark.asyncio
    async def test_sell_with_stop_loss_and_take_profit(self, executor, mock_trading_client):
        """Test SELL with bracket order."""
        mock_order = MagicMock()
        mock_order.id = "sell-bracket-123"
        mock_trading_client.submit_order.return_value = mock_order

        decision = self._make_decision(
            action=ActionType.SELL,
            symbol="TSLA",
            quantity=5,
            stop_loss=260.00,
            take_profit=230.00
        )

        result = await executor.execute_decision(decision, current_price=250.00)

        assert result.success is True

    # ==================== RISK VALIDATION TESTS ====================

    @pytest.mark.asyncio
    async def test_trading_blocked_by_time(self, executor, mock_risk_manager):
        """Test execution blocked when outside trading hours."""
        mock_risk_manager.check_trading_allowed.return_value = RiskValidationResult(
            valid=False, message="Market is closed"
        )

        decision = self._make_decision(action=ActionType.BUY)

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is False
        assert "Trading blocked" in result.message
        assert "Market is closed" in result.message
        executor.client.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_trading_blocked_during_buffer(self, executor, mock_risk_manager):
        """Test execution blocked during buffer periods."""
        mock_risk_manager.check_trading_allowed.return_value = RiskValidationResult(
            valid=False, message="No trading in first 15 minutes"
        )

        decision = self._make_decision(action=ActionType.BUY)

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is False
        assert "Trading blocked" in result.message

    @pytest.mark.asyncio
    async def test_risk_validation_failure_blocks_trade(self, executor, mock_risk_manager):
        """Test execution blocked on risk validation failure."""
        mock_risk_manager.validate_decision.return_value = RiskValidationResult(
            valid=False,
            message="Risk too high: 5.00% (max 2%)",
            violations=["Risk too high"]
        )

        decision = self._make_decision(action=ActionType.BUY)

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is False
        assert "Risk validation failed" in result.message
        executor.client.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_stop_loss_blocked(self, executor, mock_risk_manager):
        """Test execution blocked when stop-loss is missing."""
        mock_risk_manager.validate_decision.return_value = RiskValidationResult(
            valid=False,
            message="Stop-loss is REQUIRED for all trades",
            violations=["Stop-loss is REQUIRED"]
        )

        decision = self._make_decision(action=ActionType.BUY, stop_loss=None)

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is False
        assert "Risk validation failed" in result.message

    # ==================== QUANTITY ADJUSTMENT TESTS ====================

    @pytest.mark.asyncio
    async def test_quantity_adjusted_by_risk_manager(self, executor, mock_risk_manager, mock_trading_client):
        """Test quantity is adjusted when risk manager returns adjusted_quantity."""
        mock_risk_manager.validate_decision.return_value = RiskValidationResult(
            valid=True,
            message="Quantity adjusted to 5 to meet risk limits",
            adjusted_quantity=5,
            violations=["Risk adjusted"]
        )
        mock_order = MagicMock()
        mock_order.id = "adjusted-order-123"
        mock_trading_client.submit_order.return_value = mock_order

        decision = self._make_decision(
            action=ActionType.BUY,
            quantity=20  # Original quantity
        )

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is True
        assert result.filled_quantity == 5  # Adjusted quantity

    @pytest.mark.asyncio
    async def test_original_quantity_used_when_no_adjustment(self, executor, mock_risk_manager, mock_trading_client):
        """Test original quantity used when no adjustment needed."""
        mock_risk_manager.validate_decision.return_value = RiskValidationResult(
            valid=True,
            message="All risk checks passed",
            adjusted_quantity=None
        )
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_trading_client.submit_order.return_value = mock_order

        decision = self._make_decision(
            action=ActionType.BUY,
            quantity=10
        )

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.filled_quantity == 10

    # ==================== ERROR HANDLING TESTS ====================

    @pytest.mark.asyncio
    async def test_unknown_action_type_fails(self, executor):
        """Test unknown action type returns failure."""
        decision = self._make_decision(action=ActionType.BUY)
        # Manually set to invalid action (simulate future enum addition)
        decision.action = MagicMock()
        decision.action.value = "UNKNOWN"

        # This will fail because it doesn't match any known action
        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.success is False
        assert "Unknown action type" in result.message

    @pytest.mark.asyncio
    async def test_result_includes_risk_validation(self, executor, mock_risk_manager):
        """Test result always includes risk validation info."""
        decision = self._make_decision(action=ActionType.HOLD)

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.risk_validation is not None

    @pytest.mark.asyncio
    async def test_result_has_timestamp(self, executor):
        """Test result has timestamp set."""
        decision = self._make_decision(action=ActionType.HOLD)

        result = await executor.execute_decision(decision, current_price=185.00)

        assert result.timestamp is not None


class TestOrderExecutorOpenOrders:
    """Tests for get_open_orders() method."""

    @pytest.fixture
    def mock_trading_client(self):
        """Create mock trading client."""
        return MagicMock()

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        return MagicMock(spec=RiskManager)

    @pytest.fixture
    def executor(self, mock_trading_client, mock_risk_manager):
        """Create OrderExecutor."""
        return OrderExecutor(
            trading_client=mock_trading_client,
            risk_manager=mock_risk_manager,
            agent_name="test_agent"
        )

    def test_get_open_orders_returns_list(self, executor, mock_trading_client):
        """Test get_open_orders returns list of orders."""
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.symbol = "GOOGL"
        mock_order.side = MagicMock(value="buy")
        mock_order.qty = "10"
        mock_order.filled_qty = "0"
        mock_order.type = MagicMock(value="market")
        mock_order.status = MagicMock(value="new")
        mock_order.created_at = datetime.now()

        mock_trading_client.get_orders.return_value = [mock_order]

        orders = executor.get_open_orders()

        assert len(orders) == 1
        assert orders[0]["order_id"] == "order-123"
        assert orders[0]["symbol"] == "GOOGL"
        assert orders[0]["side"] == "buy"

    def test_get_open_orders_filters_symbols(self, executor, mock_trading_client):
        """Test get_open_orders only returns orders for allowed symbols."""
        googl_order = MagicMock()
        googl_order.id = "order-1"
        googl_order.symbol = "GOOGL"
        googl_order.side = MagicMock(value="buy")
        googl_order.qty = "10"
        googl_order.filled_qty = None
        googl_order.type = MagicMock(value="market")
        googl_order.status = MagicMock(value="new")
        googl_order.created_at = datetime.now()

        aapl_order = MagicMock()
        aapl_order.id = "order-2"
        aapl_order.symbol = "AAPL"  # Not in SYMBOLS
        aapl_order.side = MagicMock(value="buy")
        aapl_order.qty = "5"
        aapl_order.filled_qty = None
        aapl_order.type = MagicMock(value="market")
        aapl_order.status = MagicMock(value="new")
        aapl_order.created_at = datetime.now()

        mock_trading_client.get_orders.return_value = [googl_order, aapl_order]

        with patch('execution.order_executor.SYMBOLS', ["GOOGL", "TSLA"]):
            orders = executor.get_open_orders()

        # Only GOOGL should be returned
        assert len(orders) == 1
        assert orders[0]["symbol"] == "GOOGL"

    def test_get_open_orders_handles_filled_qty_none(self, executor, mock_trading_client):
        """Test get_open_orders handles None filled_qty."""
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.symbol = "GOOGL"
        mock_order.side = MagicMock(value="buy")
        mock_order.qty = "10"
        mock_order.filled_qty = None
        mock_order.type = MagicMock(value="market")
        mock_order.status = MagicMock(value="new")
        mock_order.created_at = datetime.now()

        mock_trading_client.get_orders.return_value = [mock_order]

        orders = executor.get_open_orders()

        assert orders[0]["filled"] == 0


class TestOrderExecutorCancelOrders:
    """Tests for cancel_all_orders() method."""

    @pytest.fixture
    def mock_trading_client(self):
        """Create mock trading client."""
        return MagicMock()

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        return MagicMock(spec=RiskManager)

    @pytest.fixture
    def executor(self, mock_trading_client, mock_risk_manager):
        """Create OrderExecutor."""
        return OrderExecutor(
            trading_client=mock_trading_client,
            risk_manager=mock_risk_manager,
            agent_name="test_agent"
        )

    def test_cancel_all_orders_calls_api(self, executor, mock_trading_client):
        """Test cancel_all_orders calls the API."""
        mock_trading_client.get_orders.return_value = []

        executor.cancel_all_orders()

        mock_trading_client.cancel_orders.assert_called_once()

    def test_cancel_all_orders_returns_count(self, executor, mock_trading_client):
        """Test cancel_all_orders returns cancelled count."""
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.symbol = "GOOGL"
        mock_order.side = MagicMock(value="buy")
        mock_order.qty = "10"
        mock_order.filled_qty = None
        mock_order.type = MagicMock(value="market")
        mock_order.status = MagicMock(value="new")
        mock_order.created_at = datetime.now()

        mock_trading_client.get_orders.return_value = [mock_order]

        with patch('execution.order_executor.SYMBOLS', ["GOOGL", "TSLA"]):
            count = executor.cancel_all_orders()

        # Returns the count from get_open_orders
        assert count == 1

    def test_cancel_all_orders_handles_error(self, executor, mock_trading_client):
        """Test cancel_all_orders handles API errors."""
        mock_trading_client.cancel_orders.side_effect = Exception("API Error")

        count = executor.cancel_all_orders()

        assert count == 0


class TestExecutionResultDataclass:
    """Tests for ExecutionResult dataclass."""

    def test_execution_result_defaults(self):
        """Test ExecutionResult default values."""
        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.HOLD,
            strategy_used=StrategyType.DEFENSIVE
        )
        risk_result = RiskValidationResult(valid=True, message="OK")

        result = ExecutionResult(
            success=True,
            order_id=None,
            message="Test",
            decision=decision,
            risk_validation=risk_result
        )

        assert result.filled_price is None
        assert result.filled_quantity is None
        assert result.timestamp is not None

    def test_execution_result_with_all_fields(self):
        """Test ExecutionResult with all fields populated."""
        decision = TradingDecision(
            timestamp=datetime.now(),
            action=ActionType.BUY,
            symbol="GOOGL",
            quantity=10,
            strategy_used=StrategyType.MOMENTUM
        )
        risk_result = RiskValidationResult(valid=True, message="OK")
        ts = datetime(2024, 1, 15, 10, 30, 0)

        result = ExecutionResult(
            success=True,
            order_id="order-123",
            message="Order placed",
            decision=decision,
            risk_validation=risk_result,
            filled_price=185.50,
            filled_quantity=10,
            timestamp=ts
        )

        assert result.success is True
        assert result.order_id == "order-123"
        assert result.filled_price == 185.50
        assert result.filled_quantity == 10
        assert result.timestamp == ts
