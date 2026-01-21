"""Order execution handling for Alpaca trading."""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from alpaca.trading.client import TradingClient
from alpaca.trading.models import Order, Position
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus

from agents.base_agent import TradingDecision, ActionType
from risk.risk_manager import RiskManager, RiskValidationResult
from config.settings import SYMBOLS

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of order execution attempt."""

    success: bool
    order_id: Optional[str]
    message: str
    decision: TradingDecision
    risk_validation: RiskValidationResult
    filled_price: Optional[float] = None
    filled_quantity: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


class OrderExecutor:
    """
    Executes trading decisions after risk validation.

    Handles order placement, bracket orders (entry + stop-loss + take-profit),
    and position closing.
    """

    def __init__(self, trading_client: TradingClient, risk_manager: RiskManager, agent_name: str):
        self.client = trading_client
        self.risk_manager = risk_manager
        self.agent_name = agent_name

    async def execute_decision(
        self, decision: TradingDecision, current_price: float
    ) -> ExecutionResult:
        """
        Execute a trading decision with full risk validation.

        Args:
            decision: The trading decision from AI agent
            current_price: Current market price for the symbol

        Returns:
            ExecutionResult with success status and details
        """
        # First check if trading is allowed (time-based)
        time_check = self.risk_manager.check_trading_allowed()
        if not time_check.valid:
            return ExecutionResult(
                success=False,
                order_id=None,
                message=f"Trading blocked: {time_check.message}",
                decision=decision,
                risk_validation=time_check,
            )

        # Validate the decision against risk limits
        risk_result = self.risk_manager.validate_decision(decision, current_price)

        if not risk_result.valid:
            return ExecutionResult(
                success=False,
                order_id=None,
                message=f"Risk validation failed: {risk_result.message}",
                decision=decision,
                risk_validation=risk_result,
            )

        # Use adjusted quantity if provided, default to decision.quantity
        quantity = risk_result.adjusted_quantity if risk_result.adjusted_quantity is not None else decision.quantity

        # Execute based on action type
        if decision.action == ActionType.HOLD:
            return ExecutionResult(
                success=True,
                order_id=None,
                message="HOLD - no action taken",
                decision=decision,
                risk_validation=risk_result,
            )

        elif decision.action == ActionType.CLOSE:
            return await self._close_position(decision, risk_result)

        elif decision.action in [ActionType.BUY, ActionType.SELL]:
            if quantity is None:
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    message="No quantity specified for order",
                    decision=decision,
                    risk_validation=risk_result,
                )
            return await self._place_order(decision, quantity, current_price, risk_result)

        return ExecutionResult(
            success=False,
            order_id=None,
            message=f"Unknown action type: {decision.action}",
            decision=decision,
            risk_validation=risk_result,
        )

    async def _place_order(
        self,
        decision: TradingDecision,
        quantity: int,
        current_price: float,
        risk_result: RiskValidationResult,
    ) -> ExecutionResult:
        """Place a buy or sell order with optional bracket (stop-loss + take-profit)."""
        try:
            side = OrderSide.BUY if decision.action == ActionType.BUY else OrderSide.SELL

            # Use bracket order if both stop-loss and take-profit are specified
            if decision.stop_loss and decision.take_profit:
                order_request = MarketOrderRequest(
                    symbol=decision.symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=decision.take_profit),
                    stop_loss=StopLossRequest(stop_price=decision.stop_loss),
                )
            elif decision.stop_loss:
                # Use OTO (one-triggers-other) for just stop-loss
                order_request = MarketOrderRequest(
                    symbol=decision.symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.OTO,
                    stop_loss=StopLossRequest(stop_price=decision.stop_loss),
                )
            else:
                # Simple market order (should not happen due to risk validation)
                order_request = MarketOrderRequest(
                    symbol=decision.symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )

            # Submit the order
            order = self.client.submit_order(order_request)

            # Handle Order object properly - extract id if present
            order_id: Optional[str] = None
            if hasattr(order, 'id') and order.id is not None:
                order_id = str(order.id)

            return ExecutionResult(
                success=True,
                order_id=order_id,
                message=f"{decision.action.value.upper()} order placed for {quantity} shares of {decision.symbol}",
                decision=decision,
                risk_validation=risk_result,
                filled_quantity=quantity,
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                order_id=None,
                message=f"Order execution failed: {str(e)}",
                decision=decision,
                risk_validation=risk_result,
            )

    async def _close_position(
        self, decision: TradingDecision, risk_result: RiskValidationResult
    ) -> ExecutionResult:
        """Close an existing position."""
        if not decision.symbol:
            return ExecutionResult(
                success=False,
                order_id=None,
                message="No symbol specified for CLOSE action",
                decision=decision,
                risk_validation=risk_result,
            )

        try:
            # Verify position exists
            positions = self.client.get_all_positions()
            position: Optional[Any] = None
            for p in positions:
                # Check for required attributes instead of strict type check
                if hasattr(p, 'symbol') and hasattr(p, 'qty') and p.symbol == decision.symbol:
                    position = p
                    break

            if not position:
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    message=f"No position found for {decision.symbol}",
                    decision=decision,
                    risk_validation=risk_result,
                )

            # Cancel any pending orders for this symbol first
            # This releases shares held by bracket orders (stop-loss/take-profit)
            try:
                open_orders = self.client.get_orders(
                    GetOrdersRequest(
                        status=QueryOrderStatus.OPEN,
                        symbols=[decision.symbol]
                    )
                )
                print(f"[{self.agent_name}] Found {len(open_orders)} open orders for {decision.symbol}")
                for order in open_orders:
                    try:
                        self.client.cancel_order_by_id(order.id)
                        print(f"[{self.agent_name}] Cancelled pending order {order.id} for {decision.symbol}")
                    except Exception as cancel_err:
                        print(f"[{self.agent_name}] Failed to cancel order {order.id}: {cancel_err}")
            except Exception as e:
                print(f"[{self.agent_name}] Error fetching open orders: {e}")

            # Close the position (shares should now be free)
            order = self.client.close_position(decision.symbol)

            # Extract order ID if available
            order_id: Optional[str] = None
            if hasattr(order, 'id') and order.id is not None:
                order_id = str(order.id)

            # Get quantity from position
            position_qty = int(float(position.qty)) if position.qty is not None else 0

            return ExecutionResult(
                success=True,
                order_id=order_id,
                message=f"Position closed for {decision.symbol} ({position_qty} shares)",
                decision=decision,
                risk_validation=risk_result,
                filled_quantity=position_qty,
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                order_id=None,
                message=f"Failed to close position: {str(e)}",
                decision=decision,
                risk_validation=risk_result,
            )

    def get_open_orders(self) -> list[dict[str, Any]]:
        """Get all open orders for monitoring."""
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = self.client.get_orders(request)

        result: list[dict[str, Any]] = []
        for o in orders:
            # Type guard: skip if not an Order object (API may return str on error)
            if not isinstance(o, Order):
                continue
            # Check for required attributes
            if not hasattr(o, 'symbol') or not hasattr(o, 'id'):
                continue
            if o.symbol not in SYMBOLS:
                continue
            result.append({
                "order_id": str(o.id) if o.id is not None else "",
                "symbol": o.symbol,
                "side": o.side.value if o.side is not None else "",
                "quantity": int(float(o.qty)) if o.qty is not None else 0,
                "filled": int(float(o.filled_qty)) if o.filled_qty is not None else 0,
                "type": o.type.value if o.type is not None else "",
                "status": o.status.value if o.status is not None else "",
                "created_at": o.created_at.isoformat() if o.created_at is not None else "",
            })
        return result

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        try:
            # Get count of open orders before cancellation
            count = len(self.get_open_orders())
            self.client.cancel_orders()
            return count
        except Exception:
            return 0
