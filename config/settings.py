"""Competition settings and risk limits (not strategy - AI decides that)."""

import os
from dataclasses import dataclass
from typing import Final

# Allowed symbols - only GOOGL and TSLA
SYMBOLS: Final[list[str]] = ["GOOGL", "TSLA"]

# PostgreSQL settings (Neon serverless)
POSTGRES_URL: Final[str] = os.getenv("DATABASE_URL", "")
POSTGRES_POOL_SIZE: Final[int] = int(os.getenv("POSTGRES_POOL_SIZE", "5"))

# Learning system settings
LEARNING_ENABLED: Final[bool] = os.getenv("LEARNING_ENABLED", "true").lower() == "true"
SIGNIFICANT_PNL_THRESHOLD: Final[float] = float(
    os.getenv("SIGNIFICANT_PNL_THRESHOLD", "50.0")
)
MAX_LEARNINGS_PER_RECALL: Final[int] = int(os.getenv("MAX_LEARNINGS_PER_RECALL", "10"))

# Starting capital for each account
STARTING_CAPITAL: Final[float] = 100_000.00

# Competition duration
COMPETITION_DAYS: Final[int] = 30


@dataclass(frozen=True)
class RiskLimits:
    """Hard risk limits enforced by system - AI cannot override."""

    # Max risk per trade as percentage of account equity
    max_risk_per_trade: float = 0.02  # 2%

    # Max total capital deployed at any time
    max_exposure: float = 0.50  # 50%

    # Max positions (1 per symbol max)
    max_positions: int = 2

    # Daily loss limit as percentage of account
    daily_loss_limit: float = 0.05  # 5%

    # Stop-loss required on all trades
    require_stop_loss: bool = True

    # Min distance for stop-loss from entry (percentage)
    min_stop_distance: float = 0.005  # 0.5%

    # Max distance for stop-loss from entry (percentage)
    max_stop_distance: float = 0.05  # 5%


RISK_LIMITS: Final[RiskLimits] = RiskLimits()


@dataclass(frozen=True)
class TradingHours:
    """Market hours for trading (Eastern Time)."""

    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16
    market_close_minute: int = 0

    # No trading in first/last 15 minutes
    buffer_minutes: int = 15

    # Decision interval in minutes
    decision_interval: int = 60  # Hourly


TRADING_HOURS: Final[TradingHours] = TradingHours()


# Logging settings
LOG_LEVEL: Final[str] = "INFO"
LOG_DIR: Final[str] = os.getenv("LOG_DIR", "logs")
DATABASE_PATH: Final[str] = os.getenv("DATABASE_PATH", "trading_competition.sqlite")
