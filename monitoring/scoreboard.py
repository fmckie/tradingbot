"""Live scoreboard for AI trading competition."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import math
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich import box

from config.settings import STARTING_CAPITAL, COMPETITION_DAYS


@dataclass
class AgentScore:
    """Score and metrics for a single agent."""

    name: str
    starting_equity: float = STARTING_CAPITAL
    current_equity: float = STARTING_CAPITAL
    cash: float = STARTING_CAPITAL
    positions_value: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    peak_equity: float = STARTING_CAPITAL
    max_drawdown: float = 0.0

    # Daily tracking
    daily_pnl: float = 0.0
    daily_trades: int = 0

    # Strategy tracking
    strategies_used: dict = field(default_factory=dict)
    current_strategy: str = "defensive"
    last_decision_reasoning: str = ""

    # Timestamps
    last_update: Optional[datetime] = None
    competition_start: datetime = field(default_factory=datetime.now)

    @property
    def total_return_percent(self) -> float:
        """Total return as percentage."""
        if self.starting_equity == 0:
            return 0.0
        return (self.current_equity - self.starting_equity) / self.starting_equity * 100

    @property
    def win_rate(self) -> float:
        """Win rate as percentage."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades * 100

    @property
    def loss_rate(self) -> float:
        """Loss rate as percentage."""
        if self.total_trades == 0:
            return 0.0
        return self.losing_trades / self.total_trades * 100

    @property
    def sharpe_ratio(self) -> float:
        """Simplified Sharpe ratio estimate (annualized)."""
        # This is a simplified calculation
        if self.total_trades < 5:
            return 0.0

        # Approximate daily returns
        days_elapsed = max(1, (datetime.now() - self.competition_start).days)
        daily_return = self.total_return_percent / days_elapsed

        # Assume 2% risk-free rate annually, ~0.0055% daily
        risk_free_daily = 0.0055
        excess_return = daily_return - risk_free_daily

        # Estimate volatility from drawdown (simplified)
        if self.max_drawdown == 0:
            return 0.0

        volatility_estimate = self.max_drawdown / 2  # Rough estimate

        if volatility_estimate == 0:
            return 0.0

        # Annualize
        sharpe = (excess_return / volatility_estimate) * math.sqrt(252)
        return round(sharpe, 2)

    def update_equity(self, equity: float, cash: float, positions_value: float):
        """Update equity values and track peak/drawdown."""
        self.current_equity = equity
        self.cash = cash
        self.positions_value = positions_value
        self.last_update = datetime.now()

        # Track peak and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity

        drawdown = (self.peak_equity - equity) / self.peak_equity * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def record_trade(self, profit: float, strategy: str):
        """Record a completed trade."""
        self.total_trades += 1
        self.daily_trades += 1

        if profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        self.realized_pnl += profit
        self.strategies_used[strategy] = self.strategies_used.get(strategy, 0) + 1

    def reset_daily(self):
        """Reset daily counters."""
        self.daily_pnl = 0.0
        self.daily_trades = 0


class Scoreboard:
    """Live scoreboard display for the competition."""

    def __init__(self):
        self.console = Console()
        self.scores: dict[str, AgentScore] = {}
        self.competition_day = 1
        self.competition_start = datetime.now()

    def register_agent(self, name: str):
        """Register an agent for tracking."""
        self.scores[name] = AgentScore(name=name, competition_start=self.competition_start)

    def update_agent(
        self,
        name: str,
        equity: float,
        cash: float,
        positions_value: float,
        unrealized_pnl: float = 0.0,
        strategy: str = "defensive",
        reasoning: str = "",
    ):
        """Update an agent's score."""
        if name not in self.scores:
            self.register_agent(name)

        score = self.scores[name]
        score.update_equity(equity, cash, positions_value)
        score.unrealized_pnl = unrealized_pnl
        score.current_strategy = strategy
        score.last_decision_reasoning = reasoning

    def record_trade(self, name: str, profit: float, strategy: str):
        """Record a completed trade for an agent."""
        if name in self.scores:
            self.scores[name].record_trade(profit, strategy)

    def get_leader(self) -> Optional[str]:
        """Get the current leader by equity."""
        if not self.scores:
            return None

        return max(self.scores.keys(), key=lambda n: self.scores[n].current_equity)

    def get_lead_amount(self) -> float:
        """Get the lead amount between first and second place."""
        if len(self.scores) < 2:
            return 0.0

        equities = sorted([s.current_equity for s in self.scores.values()], reverse=True)
        return equities[0] - equities[1]

    def render(self) -> str:
        """Render the scoreboard as a rich panel."""
        # Calculate competition day
        days_elapsed = (datetime.now() - self.competition_start).days + 1
        self.competition_day = min(days_elapsed, COMPETITION_DAYS)

        # Main scoreboard table
        table = Table(
            title=f"AI TRADING COMPETITION - DAY {self.competition_day}/{COMPETITION_DAYS}",
            box=box.DOUBLE_EDGE,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Metric", style="dim")
        for name in sorted(self.scores.keys()):
            table.add_column(name.upper(), justify="right")

        if self.scores:
            claude = self.scores.get("claude", AgentScore("claude"))
            grok = self.scores.get("grok", AgentScore("grok"))

            # Balance row
            table.add_row(
                "Balance",
                f"${claude.current_equity:,.2f}",
                f"${grok.current_equity:,.2f}",
            )

            # Return row
            table.add_row(
                "Return",
                f"{claude.total_return_percent:+.2f}%",
                f"{grok.total_return_percent:+.2f}%",
            )

            # Strategy row
            table.add_row(
                "Strategy",
                claude.current_strategy.title(),
                grok.current_strategy.title(),
            )

            # Trades row
            table.add_row(
                "Trades",
                str(claude.total_trades),
                str(grok.total_trades),
            )

            # Win rate row
            table.add_row(
                "Win Rate",
                f"{claude.win_rate:.0f}%",
                f"{grok.win_rate:.0f}%",
            )

            # Sharpe row
            table.add_row(
                "Sharpe",
                f"{claude.sharpe_ratio:.2f}",
                f"{grok.sharpe_ratio:.2f}",
            )

            # Max DD row
            table.add_row(
                "Max DD",
                f"-{claude.max_drawdown:.1f}%",
                f"-{grok.max_drawdown:.1f}%",
            )

        # Leader panel
        leader = self.get_leader()
        lead_amount = self.get_lead_amount()
        leader_text = (
            f"LEADER: {leader.upper()} (+${lead_amount:,.2f})"
            if leader
            else "NO LEADER YET"
        )

        return Panel(
            table,
            title="[bold yellow]LIVE SCOREBOARD[/bold yellow]",
            subtitle=f"[green]{leader_text}[/green]",
            border_style="yellow",
        )

    def render_decisions(self) -> str:
        """Render recent decisions panel."""
        decisions_text = ""

        for name, score in sorted(self.scores.items()):
            reasoning = score.last_decision_reasoning[:200] + "..." \
                if len(score.last_decision_reasoning) > 200 \
                else score.last_decision_reasoning

            decisions_text += f"[bold]{name.upper()}[/bold]: {reasoning}\n\n"

        return Panel(
            decisions_text or "No decisions yet",
            title="[bold cyan]RECENT DECISIONS[/bold cyan]",
            border_style="cyan",
        )

    def render_strategy_analysis(self) -> str:
        """Render strategy analysis panel."""
        analysis_text = ""

        for name, score in sorted(self.scores.items()):
            if score.strategies_used:
                most_used = max(score.strategies_used, key=score.strategies_used.get)
                usage = score.strategies_used.get(most_used, 0)
                analysis_text += f"[bold]{name.upper()}[/bold]: Primarily {most_used} ({usage} uses)\n"
            else:
                analysis_text += f"[bold]{name.upper()}[/bold]: No trades yet\n"

        return Panel(
            analysis_text or "No strategy data yet",
            title="[bold magenta]STRATEGY ANALYSIS[/bold magenta]",
            border_style="magenta",
        )

    def display(self):
        """Display the full scoreboard."""
        self.console.clear()
        self.console.print(self.render())
        self.console.print(self.render_decisions())
        self.console.print(self.render_strategy_analysis())

    def get_summary(self) -> dict:
        """Get scoreboard summary as dictionary for logging."""
        return {
            "competition_day": self.competition_day,
            "leader": self.get_leader(),
            "lead_amount": self.get_lead_amount(),
            "agents": {
                name: {
                    "equity": score.current_equity,
                    "return_percent": score.total_return_percent,
                    "trades": score.total_trades,
                    "win_rate": score.win_rate,
                    "sharpe": score.sharpe_ratio,
                    "max_drawdown": score.max_drawdown,
                    "strategy": score.current_strategy,
                }
                for name, score in self.scores.items()
            },
        }
