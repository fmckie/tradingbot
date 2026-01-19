"""Configuration module for AI Trading Competition."""
from .alpaca_config import get_claude_client, get_grok_client
from .settings import RISK_LIMITS, TRADING_HOURS, SYMBOLS

__all__ = [
    "get_claude_client",
    "get_grok_client",
    "RISK_LIMITS",
    "TRADING_HOURS",
    "SYMBOLS",
]
