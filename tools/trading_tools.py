"""Trading tools that AI agents can call to execute trades."""
from typing import Any
from dataclasses import dataclass
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLimitOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

from config.settings import SYMBOLS, RISK_LIMITS


@dataclass
class OrderResult:
    """Result of an order placement attempt."""

    success: bool
    order_id: str | None
    message: str
    details: dict


# Tool schemas for Claude/Grok function calling
TRADING_TOOLS_SCHEMA = [
    {
        "name": "get_positions",
        "description": "Get all current open positions including quantity, average entry price, current price, and unrealized P&L. Use this to understand your current exposure.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_account",
        "description": "Get account information including balance, buying power, equity, and daily P&L. Use this to understand available capital.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_orders",
        "description": "Get recent orders (filled, pending, canceled). Use this to review your trading history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Filter by order status",
                    "enum": ["open", "closed", "all"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of orders to retrieve",
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            "required": [],
        },
    },
    {
        "name": "place_order",
        "description": "Place a buy or sell order. STOP LOSS IS REQUIRED. You must specify a stop_loss price for risk management.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol - must be GOOGL or TSLA",
                    "enum": SYMBOLS,
                },
                "side": {
                    "type": "string",
                    "description": "Buy or sell",
                    "enum": ["buy", "sell"],
                },
                "quantity": {
                    "type": "integer",
                    "description": "Number of shares",
                    "minimum": 1,
                },
                "order_type": {
                    "type": "string",
                    "description": "Order type",
                    "enum": ["market", "limit"],
                },
                "limit_price": {
                    "type": "number",
                    "description": "Limit price (required for limit orders)",
                },
                "stop_loss": {
                    "type": "number",
                    "description": "REQUIRED: Stop loss price for risk management",
                },
                "take_profit": {
                    "type": "number",
                    "description": "Optional take profit price",
                },
            },
            "required": ["symbol", "side", "quantity", "order_type", "stop_loss"],
        },
    },
    {
        "name": "close_position",
        "description": "Close an existing position completely. Use this to exit a trade.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol to close - must be GOOGL or TSLA",
                    "enum": SYMBOLS,
                }
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "cancel_order",
        "description": "Cancel a pending order by order ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to cancel",
                }
            },
            "required": ["order_id"],
        },
    },
]


class TradingTools:
    """Handles trading tool calls from AI agents."""

    def __init__(self, trading_client: TradingClient, agent_name: str):
        self.client = trading_client
        self.agent_name = agent_name

    def execute(self, tool_name: str, parameters: dict) -> dict[str, Any]:
        """Execute a trading tool and return the result."""
        handlers = {
            "get_positions": self._get_positions,
            "get_account": self._get_account,
            "get_orders": self._get_orders,
            "place_order": self._place_order,
            "close_position": self._close_position,
            "cancel_order": self._cancel_order,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return handler(parameters)
        except Exception as e:
            return {"error": str(e)}

    def _get_positions(self, params: dict) -> dict:
        """Get current positions."""
        positions = self.client.get_all_positions()

        position_list = []
        total_value = 0.0
        total_unrealized_pnl = 0.0

        for pos in positions:
            if pos.symbol in SYMBOLS:  # Only show allowed symbols
                pos_value = float(pos.market_value)
                unrealized = float(pos.unrealized_pl)
                total_value += pos_value
                total_unrealized_pnl += unrealized

                position_list.append(
                    {
                        "symbol": pos.symbol,
                        "quantity": int(pos.qty),
                        "side": "long" if int(pos.qty) > 0 else "short",
                        "avg_entry_price": float(pos.avg_entry_price),
                        "current_price": float(pos.current_price),
                        "market_value": pos_value,
                        "unrealized_pnl": unrealized,
                        "unrealized_pnl_percent": float(pos.unrealized_plpc) * 100,
                        "change_today_percent": float(pos.change_today) * 100,
                    }
                )

        return {
            "positions": position_list,
            "total_positions": len(position_list),
            "total_market_value": total_value,
            "total_unrealized_pnl": total_unrealized_pnl,
        }

    def _get_account(self, params: dict) -> dict:
        """Get account information."""
        account = self.client.get_account()

        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "last_equity": float(account.last_equity),
            "daily_pnl": float(account.equity) - float(account.last_equity),
            "daily_pnl_percent": (float(account.equity) - float(account.last_equity))
            / float(account.last_equity)
            * 100
            if float(account.last_equity) > 0
            else 0.0,
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked,
        }

    def _get_orders(self, params: dict) -> dict:
        """Get recent orders."""
        status = params.get("status", "all")
        limit = min(params.get("limit", 20), 100)

        # Map status to Alpaca filter
        if status == "open":
            orders = self.client.get_orders(status="open", limit=limit)
        elif status == "closed":
            orders = self.client.get_orders(status="closed", limit=limit)
        else:
            orders = self.client.get_orders(limit=limit)

        order_list = []
        for order in orders:
            if order.symbol in SYMBOLS:
                order_list.append(
                    {
                        "order_id": str(order.id),
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "type": order.type.value,
                        "quantity": int(order.qty),
                        "filled_quantity": int(order.filled_qty) if order.filled_qty else 0,
                        "limit_price": float(order.limit_price) if order.limit_price else None,
                        "stop_price": float(order.stop_price) if order.stop_price else None,
                        "filled_avg_price": float(order.filled_avg_price)
                        if order.filled_avg_price
                        else None,
                        "status": order.status.value,
                        "created_at": order.created_at.isoformat(),
                        "filled_at": order.filled_at.isoformat() if order.filled_at else None,
                    }
                )

        return {"orders": order_list, "total_orders": len(order_list)}

    def _place_order(self, params: dict) -> dict:
        """Place a new order with validation."""
        symbol = params.get("symbol")
        side = params.get("side")
        quantity = params.get("quantity")
        order_type = params.get("order_type")
        limit_price = params.get("limit_price")
        stop_loss = params.get("stop_loss")
        take_profit = params.get("take_profit")

        # Validation
        if symbol not in SYMBOLS:
            return {"error": f"Invalid symbol. Must be one of {SYMBOLS}"}

        if not stop_loss and RISK_LIMITS.require_stop_loss:
            return {"error": "Stop loss is REQUIRED for all orders"}

        # Get current account and validate risk
        account = self.client.get_account()
        equity = float(account.equity)

        # Get current price for validation
        try:
            positions = self.client.get_all_positions()
            current_positions = {p.symbol: p for p in positions if p.symbol in SYMBOLS}
        except Exception:
            current_positions = {}

        # Check daily loss limit
        daily_pnl = float(account.equity) - float(account.last_equity)
        if daily_pnl < -equity * RISK_LIMITS.daily_loss_limit:
            return {
                "error": f"Daily loss limit ({RISK_LIMITS.daily_loss_limit*100}%) reached. No more trading today.",
                "daily_pnl": daily_pnl,
            }

        # Check max positions
        if symbol not in current_positions and len(current_positions) >= RISK_LIMITS.max_positions:
            return {
                "error": f"Max positions ({RISK_LIMITS.max_positions}) reached. Close a position first."
            }

        # Place the main order
        try:
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

            if order_type == "market":
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                )
            else:
                if not limit_price:
                    return {"error": "Limit price required for limit orders"}
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    limit_price=limit_price,
                    time_in_force=TimeInForce.DAY,
                )

            order = self.client.submit_order(order_request)

            result = {
                "success": True,
                "order_id": str(order.id),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "status": order.status.value,
                "message": f"Order placed successfully for {quantity} shares of {symbol}",
            }

            if limit_price:
                result["limit_price"] = limit_price

            # Note: In a real system, we'd place bracket orders or OCO orders
            # For simplicity, we're tracking stop_loss/take_profit separately
            result["stop_loss"] = stop_loss
            result["take_profit"] = take_profit

            return result

        except Exception as e:
            return {"error": f"Order placement failed: {str(e)}"}

    def _close_position(self, params: dict) -> dict:
        """Close an existing position."""
        symbol = params.get("symbol")

        if symbol not in SYMBOLS:
            return {"error": f"Invalid symbol. Must be one of {SYMBOLS}"}

        try:
            # Check if position exists
            positions = self.client.get_all_positions()
            position = next((p for p in positions if p.symbol == symbol), None)

            if not position:
                return {"error": f"No open position found for {symbol}"}

            # Close the position
            self.client.close_position(symbol)

            return {
                "success": True,
                "symbol": symbol,
                "closed_quantity": int(position.qty),
                "realized_pnl": float(position.unrealized_pl),
                "message": f"Position closed for {symbol}",
            }

        except Exception as e:
            return {"error": f"Failed to close position: {str(e)}"}

    def _cancel_order(self, params: dict) -> dict:
        """Cancel a pending order."""
        order_id = params.get("order_id")

        try:
            self.client.cancel_order_by_id(order_id)
            return {
                "success": True,
                "order_id": order_id,
                "message": "Order cancelled successfully",
            }
        except Exception as e:
            return {"error": f"Failed to cancel order: {str(e)}"}
