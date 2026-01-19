"""Trade and decision logging for the competition."""
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import sqlite3
from pathlib import Path

from config.settings import DATABASE_PATH, LOG_DIR
from agents.base_agent import TradingDecision, ActionType, StrategyType
from execution.order_executor import ExecutionResult


@dataclass
class TradeLog:
    """Log entry for a trade."""

    timestamp: str
    agent: str
    action: str
    symbol: Optional[str]
    quantity: Optional[int]
    price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    strategy: str
    reasoning: str
    success: bool
    message: str
    order_id: Optional[str]


@dataclass
class DecisionLog:
    """Log entry for a decision (including HOLDs)."""

    timestamp: str
    agent: str
    action: str
    strategy: str
    reasoning: str
    tool_calls: list
    market_context: dict


class TradeLogger:
    """
    Logs all trading decisions and results for analysis.

    Uses SQLite for persistent storage and JSON for detailed logs.
    """

    def __init__(self):
        self.db_path = DATABASE_PATH
        self.log_dir = Path(LOG_DIR)
        self.log_dir.mkdir(exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Trades table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                agent TEXT NOT NULL,
                action TEXT NOT NULL,
                symbol TEXT,
                quantity INTEGER,
                price REAL,
                stop_loss REAL,
                take_profit REAL,
                strategy TEXT,
                reasoning TEXT,
                success INTEGER,
                message TEXT,
                order_id TEXT
            )
        """
        )

        # Decisions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                agent TEXT NOT NULL,
                action TEXT NOT NULL,
                strategy TEXT,
                reasoning TEXT,
                tool_calls TEXT,
                market_context TEXT
            )
        """
        )

        # Performance snapshots table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                agent TEXT NOT NULL,
                equity REAL,
                cash REAL,
                positions_value REAL,
                unrealized_pnl REAL,
                daily_pnl REAL,
                total_trades INTEGER,
                win_rate REAL,
                max_drawdown REAL
            )
        """
        )

        conn.commit()
        conn.close()

    def log_decision(
        self,
        agent: str,
        decision: TradingDecision,
        market_context: dict,
    ):
        """Log a trading decision (including HOLDs)."""
        log_entry = DecisionLog(
            timestamp=decision.timestamp.isoformat(),
            agent=agent,
            action=decision.action.value,
            strategy=decision.strategy_used.value,
            reasoning=decision.reasoning,
            tool_calls=decision.tool_calls,
            market_context=market_context,
        )

        # Write to SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO decisions (timestamp, agent, action, strategy, reasoning, tool_calls, market_context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                log_entry.timestamp,
                log_entry.agent,
                log_entry.action,
                log_entry.strategy,
                log_entry.reasoning,
                json.dumps(log_entry.tool_calls),
                json.dumps(log_entry.market_context),
            ),
        )

        conn.commit()
        conn.close()

        # Also write to daily JSON log
        self._write_json_log(agent, "decision", asdict(log_entry))

    def log_trade(
        self,
        agent: str,
        decision: TradingDecision,
        result: ExecutionResult,
        price: Optional[float] = None,
    ):
        """Log a trade execution result."""
        log_entry = TradeLog(
            timestamp=datetime.now().isoformat(),
            agent=agent,
            action=decision.action.value,
            symbol=decision.symbol,
            quantity=result.filled_quantity or decision.quantity,
            price=result.filled_price or price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            strategy=decision.strategy_used.value,
            reasoning=decision.reasoning[:500],  # Truncate for DB
            success=result.success,
            message=result.message,
            order_id=result.order_id,
        )

        # Write to SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO trades
            (timestamp, agent, action, symbol, quantity, price, stop_loss, take_profit,
             strategy, reasoning, success, message, order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                log_entry.timestamp,
                log_entry.agent,
                log_entry.action,
                log_entry.symbol,
                log_entry.quantity,
                log_entry.price,
                log_entry.stop_loss,
                log_entry.take_profit,
                log_entry.strategy,
                log_entry.reasoning,
                1 if log_entry.success else 0,
                log_entry.message,
                log_entry.order_id,
            ),
        )

        conn.commit()
        conn.close()

        # Also write to daily JSON log
        self._write_json_log(agent, "trade", asdict(log_entry))

    def log_performance(
        self,
        agent: str,
        equity: float,
        cash: float,
        positions_value: float,
        unrealized_pnl: float,
        daily_pnl: float,
        total_trades: int,
        win_rate: float,
        max_drawdown: float,
    ):
        """Log performance snapshot."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO performance
            (timestamp, agent, equity, cash, positions_value, unrealized_pnl,
             daily_pnl, total_trades, win_rate, max_drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                agent,
                equity,
                cash,
                positions_value,
                unrealized_pnl,
                daily_pnl,
                total_trades,
                win_rate,
                max_drawdown,
            ),
        )

        conn.commit()
        conn.close()

    def _write_json_log(self, agent: str, log_type: str, data: dict):
        """Write to daily JSON log file."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"{agent}_{log_type}_{date_str}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(data) + "\n")

    def get_agent_trades(self, agent: str, limit: int = 50) -> list[dict]:
        """Get recent trades for an agent."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM trades
            WHERE agent = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (agent, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_agent_decisions(self, agent: str, limit: int = 50) -> list[dict]:
        """Get recent decisions for an agent."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM decisions
            WHERE agent = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (agent, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_strategy_breakdown(self, agent: str) -> dict[str, int]:
        """Get strategy usage breakdown for an agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT strategy, COUNT(*) as count
            FROM decisions
            WHERE agent = ?
            GROUP BY strategy
            ORDER BY count DESC
        """,
            (agent,),
        )

        rows = cursor.fetchall()
        conn.close()

        return {row[0]: row[1] for row in rows}

    def get_performance_history(self, agent: str, days: int = 30) -> list[dict]:
        """Get performance history for an agent."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM performance
            WHERE agent = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (agent, days * 24),  # Approximate hourly snapshots
        )

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in reversed(rows)]

    def get_competition_summary(self) -> dict:
        """Get overall competition summary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get trade counts
        cursor.execute(
            """
            SELECT agent,
                   COUNT(*) as total_trades,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_trades
            FROM trades
            GROUP BY agent
        """
        )
        trade_stats = {row[0]: {"total": row[1], "successful": row[2]} for row in cursor.fetchall()}

        # Get strategy breakdown
        cursor.execute(
            """
            SELECT agent, strategy, COUNT(*) as count
            FROM decisions
            GROUP BY agent, strategy
        """
        )
        strategy_stats = {}
        for row in cursor.fetchall():
            agent, strategy, count = row
            if agent not in strategy_stats:
                strategy_stats[agent] = {}
            strategy_stats[agent][strategy] = count

        # Get latest performance
        cursor.execute(
            """
            SELECT agent, equity, max_drawdown, win_rate
            FROM performance
            WHERE timestamp = (SELECT MAX(timestamp) FROM performance WHERE agent = performance.agent)
        """
        )
        perf_stats = {
            row[0]: {"equity": row[1], "max_drawdown": row[2], "win_rate": row[3]}
            for row in cursor.fetchall()
        }

        conn.close()

        return {
            "trades": trade_stats,
            "strategies": strategy_stats,
            "performance": perf_stats,
        }
