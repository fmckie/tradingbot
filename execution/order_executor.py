"""Order execution handling for Alpaca trading."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

from agents.base_agent import TradingDecision, ActionType
from risk.risk_manager import RiskManager, RiskValidationResult
from config.settings import SYMBOLS


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
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


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

        # Use adjusted quantity if provided
        quantity = risk_result.adjusted_quantity or decision.quantity

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

            return ExecutionResult(
                success=True,
                order_id=str(order.id),
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
            position = next((p for p in positions if p.symbol == decision.symbol), None)

            if not position:
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    message=f"No position found for {decision.symbol}",
                    decision=decision,
                    risk_validation=risk_result,
                )

            # Close the position
            order = self.client.close_position(decision.symbol)

            return ExecutionResult(
                success=True,
                order_id=str(order.id) if hasattr(order, "id") else None,
                message=f"Position closed for {decision.symbol} ({int(position.qty)} shares)",
                decision=decision,
                risk_validation=risk_result,
                filled_quantity=int(position.qty),
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                order_id=None,
                message=f"Failed to close position: {str(e)}",
                decision=decision,
                risk_validation=risk_result,
            )

    def get_open_orders(self) -> list[dict]:
        """Get all open orders for monitoring."""
        orders = self.client.get_orders(status="open")

        return [
            {
                "order_id": str(o.id),
                "symbol": o.symbol,
                "side": o.side.value,
                "quantity": int(o.qty),
                "filled": int(o.filled_qty) if o.filled_qty else 0,
                "type": o.type.value,
                "status": o.status.value,
                "created_at": o.created_at.isoformat(),
            }
            for o in orders
            if o.symbol in SYMBOLS
        ]

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        try:
            self.client.cancel_orders()
            return len(self.get_open_orders())
        except Exception:
            return 0
