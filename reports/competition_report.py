"""Generate competition reports from the learning system data."""
import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Optional
from pathlib import Path

from config.settings import STARTING_CAPITAL, POSTGRES_URL


class CompetitionReporter:
    """
    Generates readable reports from competition data.

    Uses PostgreSQL learning system data to create:
    - End-of-competition summary
    - Agent journey narratives
    - Strategy analysis
    - Key learnings compilation
    """

    def __init__(self):
        self.output_dir = Path("reports/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_full_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a complete competition report.

        Returns the report as a string and optionally writes to file.
        """
        if not POSTGRES_URL:
            return "Error: PostgreSQL not configured. Cannot generate report."

        try:
            from database.learning_store import LearningStore
            from database.postgres_client import PostgresClient

            # Ensure connection is available
            await PostgresClient.health_check()

            report_parts = []

            # Header
            report_parts.append(self._generate_header())

            # Scoreboard
            scoreboard = await self._generate_scoreboard()
            report_parts.append(scoreboard)

            # Agent journeys
            for agent_name in ["claude", "grok"]:
                journey = await self._generate_agent_journey(agent_name)
                report_parts.append(journey)

            # Key insights
            insights = await self._generate_key_insights()
            report_parts.append(insights)

            # Footer
            report_parts.append(self._generate_footer())

            report = "\n".join(report_parts)

            # Write to file if specified
            if output_file:
                output_path = self.output_dir / output_file
                output_path.write_text(report)
                print(f"Report written to: {output_path}")
            else:
                # Default filename
                date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
                output_path = self.output_dir / f"competition_report_{date_str}.txt"
                output_path.write_text(report)
                print(f"Report written to: {output_path}")

            return report

        except Exception as e:
            return f"Error generating report: {e}"

    def _generate_header(self) -> str:
        """Generate report header."""
        now = datetime.now()
        return f"""
================================================================================
                    TRADING COMPETITION RESULTS
                    Generated: {now.strftime("%Y-%m-%d %H:%M")}
================================================================================

Competition: Claude (claude-opus-4-5-20250514) vs Grok (grok-4)
Starting Capital: ${STARTING_CAPITAL:,.2f} per account
Symbols: GOOGL, TSLA
"""

    async def _generate_scoreboard(self) -> str:
        """Generate the competition scoreboard."""
        from database.learning_store import LearningStore

        scores = await LearningStore.get_all_agents_latest_scores()

        if not scores:
            return """
SCOREBOARD
----------
No competition data available yet.
"""

        # Sort by ending equity (winner first)
        scores.sort(key=lambda x: x.ending_equity or 0, reverse=True)

        winner = scores[0].agent_name.upper() if scores else "TBD"

        lines = [
            "\n" + "=" * 60,
            "                      SCOREBOARD",
            "=" * 60,
            f"\nWINNER: {winner}",
            "",
            f"{'Agent':<12} {'Final Equity':<15} {'Total P&L':<12} {'Win Rate':<10}",
            "-" * 60
        ]

        for score in scores:
            equity = float(score.ending_equity or 0)
            pnl = equity - STARTING_CAPITAL
            win_rate = 0.0
            if score.trades_count and score.trades_count > 0:
                win_rate = (score.wins / score.trades_count) * 100

            lines.append(
                f"{score.agent_name.capitalize():<12} "
                f"${equity:>12,.2f} "
                f"${pnl:>+10,.2f} "
                f"{win_rate:>8.1f}%"
            )

        lines.append("-" * 60)
        return "\n".join(lines)

    async def _generate_agent_journey(self, agent_name: str) -> str:
        """Generate the journey narrative for an agent."""
        from database.learning_store import LearningStore

        lines = [
            "\n" + "=" * 60,
            f"                {agent_name.upper()}'S JOURNEY",
            "=" * 60
        ]

        # Get top learnings
        top_learnings = await LearningStore.get_top_learnings(agent_name, limit=5)

        lines.append("\nTOP 5 LEARNINGS THAT WORKED:")
        if top_learnings:
            for i, learning in enumerate(top_learnings, 1):
                success_rate = learning.success_rate
                lines.append(
                    f"\n{i}. [{learning.category.upper()}] {learning.pattern[:80]}"
                )
                lines.append(f"   Insight: {learning.insight[:100]}...")
                lines.append(
                    f"   Track record: {learning.success_count} wins, "
                    f"{learning.failure_count} losses ({success_rate:.0f}% success)"
                )
        else:
            lines.append("   No learnings recorded yet.")

        # Get recent episodes with outcomes
        episodes = await LearningStore.get_recent_episodes(agent_name, limit=20)
        completed = [e for e in episodes if e.outcome_status != "pending"]

        lines.append("\n\nDECISIONS THAT MADE THE DIFFERENCE:")
        significant = [
            e for e in completed
            if e.outcome_pnl and abs(float(e.outcome_pnl)) >= 100
        ]
        if significant:
            for ep in significant[:5]:
                decision = ep.decision_made
                pnl = float(ep.outcome_pnl or 0)
                timestamp = ep.timestamp.strftime("%Y-%m-%d %H:%M") if ep.timestamp else "N/A"
                lines.append(
                    f"\n- [{timestamp}] {decision.get('action', 'N/A').upper()} "
                    f"{decision.get('symbol', 'N/A')}"
                )
                lines.append(f"  Strategy: {decision.get('strategy', 'unknown')}")
                lines.append(f"  Outcome: {ep.outcome_status} (${pnl:+.2f})")
                reasoning = decision.get('reasoning', '')[:100]
                if reasoning:
                    lines.append(f"  Reasoning: {reasoning}...")
        else:
            lines.append("   No significant trades recorded yet.")

        # Strategy breakdown
        strategy_counts = {}
        strategy_wins = {}
        for ep in completed:
            strategy = ep.decision_made.get('strategy', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            if ep.outcome_status == 'win':
                strategy_wins[strategy] = strategy_wins.get(strategy, 0) + 1

        lines.append("\n\nSTRATEGY BREAKDOWN:")
        if strategy_counts:
            for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
                wins = strategy_wins.get(strategy, 0)
                win_rate = (wins / count * 100) if count > 0 else 0
                lines.append(f"  - {strategy.upper()}: {count} trades, {win_rate:.0f}% win rate")
        else:
            lines.append("   No strategy data available.")

        return "\n".join(lines)

    async def _generate_key_insights(self) -> str:
        """Generate key insights section."""
        from database.learning_store import LearningStore

        lines = [
            "\n" + "=" * 60,
            "                    KEY INSIGHTS",
            "=" * 60
        ]

        # Aggregate data across both agents
        all_learnings = []
        for agent_name in ["claude", "grok"]:
            learnings = await LearningStore.get_top_learnings(agent_name, limit=10)
            all_learnings.extend(learnings)

        if all_learnings:
            # Find most successful patterns
            successful = sorted(
                all_learnings,
                key=lambda x: x.success_count - x.failure_count,
                reverse=True
            )[:3]

            lines.append("\nMOST RELIABLE PATTERNS:")
            for learning in successful:
                lines.append(
                    f"  - [{learning.agent_name.upper()}] {learning.pattern[:60]}"
                )
                lines.append(
                    f"    ({learning.success_count} wins, {learning.failure_count} losses)"
                )

            # Count categories
            category_counts = {}
            for l in all_learnings:
                category_counts[l.category] = category_counts.get(l.category, 0) + 1

            lines.append("\n\nLEARNING CATEGORIES:")
            for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  - {cat.upper()}: {count} learnings")

            # Tag frequency
            tag_counts = {}
            for l in all_learnings:
                for tag in l.tags:
                    tag_counts[tag.upper()] = tag_counts.get(tag.upper(), 0) + 1

            lines.append("\n\nMOST COMMON TAGS:")
            top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:10]
            for tag, count in top_tags:
                lines.append(f"  - {tag}: {count}")

        else:
            lines.append("\nNo insights available yet. Run more trading cycles to generate data.")

        return "\n".join(lines)

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""

================================================================================
                           END OF REPORT
================================================================================

This report was generated from the PostgreSQL learning system.
Data includes all episodes, reflections, and distilled learnings from the
competition period.

For questions or issues, check the trading_competition.sqlite for SQLite logs
or query the PostgreSQL database directly for detailed learning data.
"""


async def generate_report_cli():
    """CLI entry point for report generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate trading competition report")
    parser.add_argument(
        "--output", "-o",
        help="Output filename (default: auto-generated timestamp)",
        default=None
    )
    args = parser.parse_args()

    reporter = CompetitionReporter()
    report = await reporter.generate_full_report(args.output)
    print(report)


if __name__ == "__main__":
    asyncio.run(generate_report_cli())
