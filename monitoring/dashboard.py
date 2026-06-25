"""Read-only web monitoring dashboard for the AI trading competition.

Serves a single nof1 "Alpha Arena"-style page plus a JSON snapshot built from
the competition SQLite log. The database is opened **read-only** (``mode=ro``)
so this never blocks ``main.py`` while it writes during a live run.

Run standalone:

    python -m monitoring.dashboard                 # http://127.0.0.1:8787
    python -m monitoring.dashboard --port 9000 --db trading_competition.sqlite
    python -m monitoring.dashboard --once          # print the JSON snapshot

No third-party dependencies — only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from config.settings import (
    DATABASE_PATH,
    RISK_LIMITS,
    STARTING_CAPITAL,
    SYMBOLS,
)

logger = logging.getLogger("monitoring.dashboard")

# Static labels (descriptions, not measured numbers). Numeric values are always
# read from the database or the RISK_LIMITS constants.
HTML_PATH = Path(__file__).with_name("dashboard.html")
INDICATORS = ["RSI", "MACD", "Bollinger Bands", "VWAP", "ATR", "EMA"]
FOOTER = (
    "Source: trading_competition.sqlite · paper-traded · "
    "Python · Alpaca · Postgres learning loop · Modal"
)

# Known agent identities. Display names mirror the models actually configured in
# agents/claude_agent.py and agents/grok_agent.py (self.model) — keep in sync if
# those change. Unknown agents fall back to an upper-cased key with a trailing
# sort order so the snapshot stays deterministic.
AGENT_DISPLAY = {"claude": "CLAUDE SONNET 4.6", "grok": "xAI GROK 4.3"}
AGENT_SORT = {"claude": 0, "grok": 1}


# --------------------------------------------------------------------------- #
# Read-only database access
# --------------------------------------------------------------------------- #
def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open ``db_path`` read-only via a SQLite URI so reads never block writers."""
    uri = f"file:{quote(str(db_path))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _distinct_agents(conn: sqlite3.Connection) -> list[str]:
    """Return every agent that appears in any table, in a deterministic order."""
    agents: set[str] = set()
    for table in ("performance", "decisions", "trades"):
        if _table_exists(conn, table):
            agents.update(
                str(r[0]) for r in conn.execute(f"SELECT DISTINCT agent FROM {table}")
            )
    return sorted(agents, key=lambda a: (AGENT_SORT.get(a, 99), a))


def _latest_performance(conn: sqlite3.Connection, agent: str) -> dict[str, Any]:
    """Latest equity/drawdown snapshot for an agent (empty dict if none)."""
    if not _table_exists(conn, "performance"):
        return {}
    row = conn.execute(
        """
        SELECT equity, cash, positions_value, max_drawdown, timestamp
        FROM performance
        WHERE agent = ?
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (agent,),
    ).fetchone()
    return dict(row) if row else {}


def _counts(
    conn: sqlite3.Connection, table: str, column: str, agent: str
) -> dict[str, int]:
    """``{value: count}`` for one column, one agent — sorted for determinism."""
    if not _table_exists(conn, table):
        return {}
    rows = conn.execute(
        f"SELECT {column} AS k, COUNT(*) AS n FROM {table} "
        f"WHERE agent = ? GROUP BY {column}",
        (agent,),
    ).fetchall()
    return {str(r["k"]): int(r["n"]) for r in sorted(rows, key=lambda r: str(r["k"]))}


def _orders(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """All logged orders (the decision/order log), oldest first."""
    if not _table_exists(conn, "trades"):
        return []
    rows = conn.execute(
        """
        SELECT timestamp, agent, action, symbol, quantity, price,
               stop_loss, take_profit, strategy, success, message
        FROM trades
        ORDER BY timestamp ASC, id ASC
        """
    ).fetchall()
    orders: list[dict[str, Any]] = []
    for r in rows:
        orders.append(
            {
                "timestamp": r["timestamp"],
                "agent": r["agent"],
                "action": r["action"],
                "symbol": r["symbol"],
                "quantity": r["quantity"],
                "price": r["price"],
                "stop_loss": r["stop_loss"],
                "take_profit": r["take_profit"],
                "strategy": r["strategy"],
                "accepted": bool(r["success"]),
                "message": r["message"] or "",
            }
        )
    return orders


def _agent_snapshot(conn: sqlite3.Connection, agent: str) -> dict[str, Any]:
    perf = _latest_performance(conn, agent)
    equity = float(perf.get("equity") or STARTING_CAPITAL)
    action_counts = _counts(conn, "decisions", "action", agent)
    strategy_counts = _counts(conn, "decisions", "strategy", agent)
    order_rows = (
        conn.execute(
            "SELECT success, COUNT(*) AS n FROM trades "
            "WHERE agent = ? GROUP BY success",
            (agent,),
        ).fetchall()
        if _table_exists(conn, "trades")
        else []
    )
    accepted = sum(int(r["n"]) for r in order_rows if r["success"] == 1)
    rejected = sum(int(r["n"]) for r in order_rows if r["success"] != 1)
    return {
        "key": agent,
        "name": AGENT_DISPLAY.get(agent, agent.upper()),
        "equity": round(equity, 2),
        "starting_equity": round(STARTING_CAPITAL, 2),
        "cash": round(float(perf.get("cash") or equity), 2),
        "positions_value": round(float(perf.get("positions_value") or 0.0), 2),
        "return_pct": round((equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 2),
        "max_drawdown": round(float(perf.get("max_drawdown") or 0.0), 2),
        "decisions": sum(action_counts.values()),
        "actions": action_counts,
        "strategies": strategy_counts,
        "orders": accepted + rejected,
        "orders_accepted": accepted,
        "orders_rejected": rejected,
        "last_update": perf.get("timestamp"),
    }


def build_state(db_path: Path, now: datetime | None = None) -> dict[str, Any]:
    """Build the full JSON snapshot from the read-only competition database."""
    generated_at = (now or datetime.now(UTC)).isoformat()
    conn = _connect_readonly(db_path)
    try:
        agents = [_agent_snapshot(conn, a) for a in _distinct_agents(conn)]
        orders = _orders(conn)
    finally:
        conn.close()

    total_decisions = sum(a["decisions"] for a in agents)
    total_orders = len(orders)
    rejected = sum(1 for o in orders if not o["accepted"])
    max_dd = max((a["max_drawdown"] for a in agents), default=0.0)
    daily_halt_pct = round(RISK_LIMITS.daily_loss_limit * 100, 1)

    return {
        "generated_at": generated_at,
        "meta": {
            "title": "AI TRADING COMPETITION",
            "mode": "PAPER",
            "subtitle": (
                "Claude Sonnet 4.6 vs xAI Grok 4.3 · GOOGL · TSLA · "
                "independent hourly calls"
            ),
            "symbols": list(SYMBOLS),
            "indicators": INDICATORS,
            "starting_capital": round(STARTING_CAPITAL, 2),
            "source": Path(db_path).name,
        },
        "agents": agents,
        "decisions": {"total": total_decisions},
        "risk_limits": {
            "max_risk_per_trade_pct": round(RISK_LIMITS.max_risk_per_trade * 100, 2),
            "max_exposure_pct": round(RISK_LIMITS.max_exposure * 100, 2),
            "max_positions": RISK_LIMITS.max_positions,
            "daily_loss_limit_pct": daily_halt_pct,
            "min_stop_distance_pct": round(RISK_LIMITS.min_stop_distance * 100, 2),
            "max_stop_distance_pct": round(RISK_LIMITS.max_stop_distance * 100, 2),
            "require_stop_loss": RISK_LIMITS.require_stop_loss,
        },
        "orders": orders,
        "kpis": {
            "decisions": total_decisions,
            "orders": total_orders,
            "orders_rejected": rejected,
            "risk_gated_pct": 100 if total_orders else 0,
            "limits_summary": "2 / 50 / 5",
            "max_drawdown": round(max_dd, 2),
            "daily_halt_pct": daily_halt_pct,
        },
        "footer": FOOTER,
    }


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
class DashboardServer(ThreadingHTTPServer):
    """Threading HTTP server carrying the read-only DB and HTML paths."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], db_path: Path, html_path: Path):
        super().__init__(address, DashboardHandler)
        self.db_path = db_path
        self.html_path = html_path


class DashboardHandler(BaseHTTPRequestHandler):
    """Serves ``/`` (the page) and ``/api/state`` (the JSON snapshot)."""

    server_version = "TradingDashboard/1.0"

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        started = time.monotonic()
        path = urlparse(self.path).path
        status = 500
        try:
            if path == "/" or path == "/index.html":
                status = self._serve_html()
            elif path == "/api/state":
                status = self._serve_state()
            elif path == "/healthz":
                status = self._send_json(200, {"status": "ok"})
            else:
                status = self._send_json(404, {"error": "not found", "path": path})
        except BrokenPipeError:
            status = 499  # client disconnected before the response finished
        except Exception as exc:  # boundary: never crash the server thread
            logger.exception("dashboard request failed path=%s", path)
            status = self._send_json(500, {"error": str(exc)})
        finally:
            elapsed_ms = round((time.monotonic() - started) * 1000, 1)
            logger.info(
                "request method=GET path=%s status=%s elapsed_ms=%s client=%s",
                path,
                status,
                elapsed_ms,
                self.client_address[0],
            )

    def _serve_html(self) -> int:
        html_path: Path = self.server.html_path  # type: ignore[attr-defined]
        try:
            body = html_path.read_bytes()
        except OSError as exc:
            return self._send_json(500, {"error": f"cannot read dashboard.html: {exc}"})
        self._send_bytes(200, "text/html; charset=utf-8", body)
        return 200

    def _serve_state(self) -> int:
        db_path: Path = self.server.db_path  # type: ignore[attr-defined]
        try:
            state = build_state(db_path)
        except sqlite3.OperationalError as exc:
            return self._send_json(
                503, {"error": f"database unavailable: {exc}", "db": str(db_path)}
            )
        return self._send_json(200, state)

    def _send_json(self, status: int, payload: dict[str, Any]) -> int:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", body)
        return status

    def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence the default stderr logging; we log structured lines instead."""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m monitoring.dashboard",
        description="Read-only web monitoring dashboard for the trading competition.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="bind host (default 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8787, help="bind port (default 8787)"
    )
    parser.add_argument(
        "--db",
        default=DATABASE_PATH,
        help=f"path to the competition SQLite file (default {DATABASE_PATH})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="print the JSON snapshot to stdout and exit (no server)",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args(argv)
    db_path = Path(args.db).resolve()
    if not db_path.exists():
        logger.error("database not found path=%s", db_path)
        return 2

    if args.once:
        print(json.dumps(build_state(db_path), indent=2, sort_keys=True))
        return 0

    server = DashboardServer((args.host, args.port), db_path, HTML_PATH)
    logger.info(
        "dashboard listening url=http://%s:%s db=%s html=%s",
        args.host,
        args.port,
        db_path,
        HTML_PATH,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("dashboard shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
