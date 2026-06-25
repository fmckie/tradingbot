"""Risk manager enforcing hard limits on trading decisions."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from alpaca.trading.client import TradingClient

from agents.base_agent import ActionType, TradingDecision
from config.settings import RISK_LIMITS, SYMBOLS


@dataclass
class RiskValidationResult:
    """Result of risk validation check."""

    valid: bool
    message: str
    adjusted_quantity: int | None = None
    violations: list[str] = field(default_factory=list)


class RiskManager:
    """
    Enforces hard risk limits on all trading decisions.

    These limits cannot be overridden by AI agents.
    """

    def __init__(self, trading_client: TradingClient, agent_name: str):
        self.client = trading_client
        self.agent_name = agent_name
        self.daily_pnl_start: float | None = None
        self.last_reset_date: date | None = None

    def validate_decision(
        self, decision: TradingDecision, current_price: float
    ) -> RiskValidationResult:
        """
        Validate a trading decision against all risk limits.

        Returns validation result with any violations or adjustments.
        """
        violations = []

        # HOLD and CLOSE actions are always valid
        if decision.action == ActionType.HOLD:
            return RiskValidationResult(
                valid=True, message="HOLD action - no validation needed"
            )

        if decision.action == ActionType.CLOSE:
            return RiskValidationResult(valid=True, message="CLOSE action - validated")

        # For BUY/SELL, validate thoroughly
        if decision.action in [ActionType.BUY, ActionType.SELL]:
            # 1. Validate symbol
            if decision.symbol not in SYMBOLS:
                violations.append(
                    f"Invalid symbol: {decision.symbol}. Must be {SYMBOLS}"
                )

            # 2. Validate stop-loss exists
            if RISK_LIMITS.require_stop_loss and not decision.stop_loss:
                violations.append("Stop-loss is REQUIRED for all trades")

            # 3. Validate stop-loss distance
            if decision.stop_loss and current_price > 0:
                stop_distance = abs(current_price - decision.stop_loss) / current_price

                if stop_distance < RISK_LIMITS.min_stop_distance:
                    violations.append(
                        f"Stop-loss too tight: {stop_distance * 100:.2f}% "
                        f"(min {RISK_LIMITS.min_stop_distance * 100:.1f}%)"
                    )

                if stop_distance > RISK_LIMITS.max_stop_distance:
                    violations.append(
                        f"Stop-loss too wide: {stop_distance * 100:.2f}% "
                        f"(max {RISK_LIMITS.max_stop_distance * 100:.1f}%)"
                    )

            # 4. Validate quantity exists
            if not decision.quantity or decision.quantity <= 0:
                violations.append("Invalid quantity - must be positive")

            # Get account info for remaining validations
            account = self.client.get_account()
            if isinstance(account, dict):
                equity = float(account.get("equity", 0))
            else:
                equity = float(account.equity or 0)

            # 5. Check daily loss limit
            daily_loss_check = self._check_daily_loss_limit(equity)
            if not daily_loss_check.valid:
                violations.append(daily_loss_check.message)

            # 6. Check max positions
            positions = self.client.get_all_positions()
            current_positions = [
                p for p in positions if hasattr(p, "symbol") and p.symbol in SYMBOLS
            ]

            if len(current_positions) >= RISK_LIMITS.max_positions:
                # Check if this is for a symbol we already hold
                held_symbols = [p.symbol for p in current_positions]
                if decision.symbol not in held_symbols:
                    violations.append(
                        f"Max positions ({RISK_LIMITS.max_positions}) reached. "
                        "Close a position first."
                    )

            # 7. Calculate and validate position size for risk
            if decision.quantity and decision.stop_loss and current_price > 0:
                # Risk per share
                risk_per_share = abs(current_price - decision.stop_loss)
                total_risk = decision.quantity * risk_per_share
                risk_percent = total_risk / equity

                if risk_percent > RISK_LIMITS.max_risk_per_trade:
                    # Calculate max quantity that fits within risk limit
                    max_risk_amount = equity * RISK_LIMITS.max_risk_per_trade
                    adjusted_qty = int(max_risk_amount / risk_per_share)

                    violations.append(
                        f"Risk too high: {risk_percent * 100:.2f}% "
                        f"(max {RISK_LIMITS.max_risk_per_trade * 100:.0f}%). "
                        f"Adjusted quantity from {decision.quantity} to {adjusted_qty}"
                    )

                    # Return with adjusted quantity if possible
                    if (
                        adjusted_qty > 0 and not violations[:-1]
                    ):  # Only the quantity violation
                        return RiskValidationResult(
                            valid=True,
                            message=(
                                f"Quantity adjusted to {adjusted_qty} "
                                f"to meet risk limits"
                            ),
                            adjusted_quantity=adjusted_qty,
                            violations=[violations[-1]],
                        )

            # 8. Check total exposure
            if decision.quantity and current_price > 0:
                order_value = decision.quantity * current_price
                current_exposure = sum(
                    float(getattr(p, "market_value", 0) or 0) for p in current_positions
                )
                total_exposure = current_exposure + order_value
                exposure_percent = total_exposure / equity

                if exposure_percent > RISK_LIMITS.max_exposure:
                    violations.append(
                        f"Would exceed max exposure: {exposure_percent * 100:.1f}% "
                        f"(max {RISK_LIMITS.max_exposure * 100:.0f}%)"
                    )

        # Return result
        if violations:
            return RiskValidationResult(
                valid=False,
                message="; ".join(violations),
                violations=violations,
            )

        return RiskValidationResult(
            valid=True,
            message="All risk checks passed",
        )

    def _check_daily_loss_limit(self, current_equity: float) -> RiskValidationResult:
        """Check if daily loss limit has been reached."""
        today = datetime.now().date()

        # Reset daily tracking if new day
        if self.last_reset_date != today:
            self.daily_pnl_start = current_equity
            self.last_reset_date = today

        if self.daily_pnl_start:
            daily_pnl = current_equity - self.daily_pnl_start
            daily_pnl_percent = daily_pnl / self.daily_pnl_start

            if daily_pnl_percent < -RISK_LIMITS.daily_loss_limit:
                return RiskValidationResult(
                    valid=False,
                    message=f"Daily loss limit reached: {daily_pnl_percent * 100:.2f}% "
                    f"(limit: -{RISK_LIMITS.daily_loss_limit * 100:.0f}%)",
                )

        return RiskValidationResult(valid=True, message="Within daily loss limit")

    def check_trading_allowed(self) -> RiskValidationResult:
        """Check if trading is currently allowed based on time."""
        from datetime import timedelta

        import pytz

        from config.settings import TRADING_HOURS

        now = datetime.now(pytz.timezone("America/New_York"))
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

        # Check if market is open
        if not (market_open <= now <= market_close):
            return RiskValidationResult(valid=False, message="Market is closed")

        # Check buffer periods (use timedelta to avoid negative minute issues)
        buffer_delta = timedelta(minutes=TRADING_HOURS.buffer_minutes)
        buffer_start = market_open + buffer_delta
        buffer_end = market_close - buffer_delta

        if now < buffer_start:
            return RiskValidationResult(
                valid=False,
                message=(
                    f"No trading in first {TRADING_HOURS.buffer_minutes} "
                    f"minutes of session"
                ),
            )

        if now > buffer_end:
            return RiskValidationResult(
                valid=False,
                message=(
                    f"No trading in last {TRADING_HOURS.buffer_minutes} "
                    f"minutes of session"
                ),
            )

        return RiskValidationResult(valid=True, message="Trading allowed")

    def calculate_position_size(
        self,
        current_price: float,
        stop_loss: float,
        max_risk_percent: float | None = None,
    ) -> int:
        """
        Calculate maximum position size based on risk limits.

        Args:
            current_price: Current stock price
            stop_loss: Planned stop-loss price
            max_risk_percent: Max risk as decimal (default: use RISK_LIMITS)

        Returns:
            Maximum number of shares that can be traded
        """
        if max_risk_percent is None:
            max_risk_percent = RISK_LIMITS.max_risk_per_trade

        account = self.client.get_account()
        if isinstance(account, dict):
            equity = float(account.get("equity", 0))
        else:
            equity = float(account.equity or 0)

        risk_per_share = abs(current_price - stop_loss)
        if risk_per_share <= 0:
            return 0

        max_risk_amount = equity * max_risk_percent
        max_shares = int(max_risk_amount / risk_per_share)

        # Also check exposure limit
        max_exposure_shares = int((equity * RISK_LIMITS.max_exposure) / current_price)

        return min(max_shares, max_exposure_shares)

    def get_risk_status(self) -> dict[str, Any]:
        """Get current risk status for monitoring."""
        account = self.client.get_account()
        if isinstance(account, dict):
            equity = float(account.get("equity", 0))
        else:
            equity = float(account.equity or 0)
        positions = self.client.get_all_positions()

        # Filter to Position objects with symbols in our list
        valid_positions = [
            p for p in positions if hasattr(p, "symbol") and p.symbol in SYMBOLS
        ]
        current_exposure = sum(
            float(getattr(p, "market_value", 0) or 0) for p in valid_positions
        )
        exposure_percent = (current_exposure / equity * 100) if equity > 0 else 0

        # Daily P&L
        daily_pnl = 0.0
        daily_pnl_percent = 0.0
        if self.daily_pnl_start:
            daily_pnl = equity - self.daily_pnl_start
            daily_pnl_percent = (
                (daily_pnl / self.daily_pnl_start * 100)
                if self.daily_pnl_start > 0
                else 0
            )

        return {
            "agent": self.agent_name,
            "equity": equity,
            "current_exposure": current_exposure,
            "exposure_percent": round(exposure_percent, 1),
            "max_exposure_percent": RISK_LIMITS.max_exposure * 100,
            "positions_count": len(valid_positions),
            "max_positions": RISK_LIMITS.max_positions,
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_percent": round(daily_pnl_percent, 2),
            "daily_loss_limit_percent": -RISK_LIMITS.daily_loss_limit * 100,
            "at_daily_limit": daily_pnl_percent < -RISK_LIMITS.daily_loss_limit * 100,
        }
