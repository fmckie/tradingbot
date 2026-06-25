"""Monitoring module for competition tracking."""

from typing import Any

from .logger import TradeLogger
from .scoreboard import AgentScore, Scoreboard

__all__ = [
    "Scoreboard",
    "AgentScore",
    "TradeLogger",
    "build_state",
    "run_dashboard",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the web dashboard entry points.

    Imported on demand so ``import monitoring`` (and ``python -m
    monitoring.dashboard``) don't eagerly pull in the HTTP server stack.
    """
    if name in ("build_state", "run_dashboard"):
        from . import dashboard

        return dashboard.build_state if name == "build_state" else dashboard.run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
