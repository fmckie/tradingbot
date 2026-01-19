"""CRUD operations for the learning system (episodes, reflections, learnings)."""
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Any
from enum import Enum

from .postgres_client import PostgresClient


class OutcomeStatus(Enum):
    """Status of a trading episode outcome."""
    PENDING = "pending"
    WIN = "win"
    LOSS = "loss"
    HOLD = "hold"
    BREAKEVEN = "breakeven"


@dataclass
class Episode:
    """A single decision cycle episode."""
    id: Optional[int] = None
    agent_name: str = ""
    timestamp: Optional[datetime] = None
    market_regime: str = ""
    symbols_context: dict = field(default_factory=dict)
    account_state: dict = field(default_factory=dict)
    decision_made: dict = field(default_factory=dict)
    outcome_pnl: Optional[Decimal] = None
    outcome_status: str = OutcomeStatus.PENDING.value
    created_at: Optional[datetime] = None


@dataclass
class Reflection:
    """Post-outcome analysis of an episode."""
    id: Optional[int] = None
    episode_id: Optional[int] = None
    agent_name: str = ""
    what_worked: str = ""
    what_failed: str = ""
    lesson_learned: str = ""
    next_time_will: str = ""
    confidence_adjustment: Optional[Decimal] = None
    tags: list[str] = field(default_factory=list)
    created_at: Optional[datetime] = None


@dataclass
class Learning:
    """A distilled rule or pattern from experience."""
    id: Optional[int] = None
    agent_name: str = ""
    category: str = ""  # strategy/indicator/timing/risk
    pattern: str = ""
    insight: str = ""
    success_count: int = 0
    failure_count: int = 0
    last_validated: Optional[datetime] = None
    is_active: bool = True
    tags: list[str] = field(default_factory=list)
    created_at: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total * 100


@dataclass
class CompetitionScore:
    """Daily competition score for an agent."""
    id: Optional[int] = None
    agent_name: str = ""
    date: Optional[date] = None
    starting_equity: Optional[Decimal] = None
    ending_equity: Optional[Decimal] = None
    daily_pnl: Optional[Decimal] = None
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    strategies_used: dict = field(default_factory=dict)
    top_learning_id: Optional[int] = None


class LearningStore:
    """
    Storage layer for the learning system.

    Handles CRUD operations for episodes, reflections, and learnings.
    Uses tag-based search for cost efficiency (no embeddings).
    """

    # ==================== Episodes ====================

    @staticmethod
    async def create_episode(episode: Episode) -> int:
        """Create a new episode and return its ID."""
        query = """
            INSERT INTO episodes
            (agent_name, timestamp, market_regime, symbols_context,
             account_state, decision_made, outcome_pnl, outcome_status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        """
        return await PostgresClient.fetchval(
            query,
            episode.agent_name,
            episode.timestamp or datetime.now(),
            episode.market_regime,
            json.dumps(episode.symbols_context),
            json.dumps(episode.account_state),
            json.dumps(episode.decision_made),
            episode.outcome_pnl,
            episode.outcome_status
        )

    @staticmethod
    async def get_episode(episode_id: int) -> Optional[Episode]:
        """Get an episode by ID."""
        query = "SELECT * FROM episodes WHERE id = $1"
        row = await PostgresClient.fetchrow(query, episode_id)
        if row:
            return Episode(
                id=row["id"],
                agent_name=row["agent_name"],
                timestamp=row["timestamp"],
                market_regime=row["market_regime"],
                symbols_context=json.loads(row["symbols_context"]) if row["symbols_context"] else {},
                account_state=json.loads(row["account_state"]) if row["account_state"] else {},
                decision_made=json.loads(row["decision_made"]) if row["decision_made"] else {},
                outcome_pnl=row["outcome_pnl"],
                outcome_status=row["outcome_status"],
                created_at=row["created_at"]
            )
        return None

    @staticmethod
    async def update_episode_outcome(
        episode_id: int,
        outcome_pnl: Decimal,
        outcome_status: str
    ) -> None:
        """Update an episode's outcome after trade closes."""
        query = """
            UPDATE episodes
            SET outcome_pnl = $1, outcome_status = $2
            WHERE id = $3
        """
        await PostgresClient.execute(query, outcome_pnl, outcome_status, episode_id)

    @staticmethod
    async def get_recent_episodes(
        agent_name: str,
        limit: int = 20
    ) -> list[Episode]:
        """Get recent episodes for an agent."""
        query = """
            SELECT * FROM episodes
            WHERE agent_name = $1
            ORDER BY timestamp DESC
            LIMIT $2
        """
        rows = await PostgresClient.fetch(query, agent_name, limit)
        return [
            Episode(
                id=row["id"],
                agent_name=row["agent_name"],
                timestamp=row["timestamp"],
                market_regime=row["market_regime"],
                symbols_context=json.loads(row["symbols_context"]) if row["symbols_context"] else {},
                account_state=json.loads(row["account_state"]) if row["account_state"] else {},
                decision_made=json.loads(row["decision_made"]) if row["decision_made"] else {},
                outcome_pnl=row["outcome_pnl"],
                outcome_status=row["outcome_status"],
                created_at=row["created_at"]
            )
            for row in rows
        ]

    @staticmethod
    async def get_episodes_by_regime(
        agent_name: str,
        market_regime: str,
        limit: int = 10
    ) -> list[Episode]:
        """Get episodes matching a market regime."""
        query = """
            SELECT * FROM episodes
            WHERE agent_name = $1 AND market_regime ILIKE $2
            ORDER BY timestamp DESC
            LIMIT $3
        """
        rows = await PostgresClient.fetch(query, agent_name, f"%{market_regime}%", limit)
        return [
            Episode(
                id=row["id"],
                agent_name=row["agent_name"],
                timestamp=row["timestamp"],
                market_regime=row["market_regime"],
                symbols_context=json.loads(row["symbols_context"]) if row["symbols_context"] else {},
                account_state=json.loads(row["account_state"]) if row["account_state"] else {},
                decision_made=json.loads(row["decision_made"]) if row["decision_made"] else {},
                outcome_pnl=row["outcome_pnl"],
                outcome_status=row["outcome_status"],
                created_at=row["created_at"]
            )
            for row in rows
        ]

    # ==================== Reflections ====================

    @staticmethod
    async def create_reflection(reflection: Reflection) -> int:
        """Create a new reflection and return its ID."""
        query = """
            INSERT INTO reflections
            (episode_id, agent_name, what_worked, what_failed,
             lesson_learned, next_time_will, confidence_adjustment, tags)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        """
        return await PostgresClient.fetchval(
            query,
            reflection.episode_id,
            reflection.agent_name,
            reflection.what_worked,
            reflection.what_failed,
            reflection.lesson_learned,
            reflection.next_time_will,
            reflection.confidence_adjustment,
            reflection.tags
        )

    @staticmethod
    async def get_reflection(reflection_id: int) -> Optional[Reflection]:
        """Get a reflection by ID."""
        query = "SELECT * FROM reflections WHERE id = $1"
        row = await PostgresClient.fetchrow(query, reflection_id)
        if row:
            return Reflection(
                id=row["id"],
                episode_id=row["episode_id"],
                agent_name=row["agent_name"],
                what_worked=row["what_worked"],
                what_failed=row["what_failed"],
                lesson_learned=row["lesson_learned"],
                next_time_will=row["next_time_will"],
                confidence_adjustment=row["confidence_adjustment"],
                tags=row["tags"] or [],
                created_at=row["created_at"]
            )
        return None

    @staticmethod
    async def get_reflections_by_tags(
        agent_name: str,
        tags: list[str],
        limit: int = 10
    ) -> list[Reflection]:
        """Get reflections matching any of the given tags."""
        query = """
            SELECT * FROM reflections
            WHERE agent_name = $1 AND tags && $2::text[]
            ORDER BY created_at DESC
            LIMIT $3
        """
        rows = await PostgresClient.fetch(query, agent_name, tags, limit)
        return [
            Reflection(
                id=row["id"],
                episode_id=row["episode_id"],
                agent_name=row["agent_name"],
                what_worked=row["what_worked"],
                what_failed=row["what_failed"],
                lesson_learned=row["lesson_learned"],
                next_time_will=row["next_time_will"],
                confidence_adjustment=row["confidence_adjustment"],
                tags=row["tags"] or [],
                created_at=row["created_at"]
            )
            for row in rows
        ]

    # ==================== Learnings ====================

    @staticmethod
    async def create_learning(learning: Learning) -> int:
        """Create a new learning and return its ID."""
        query = """
            INSERT INTO learnings
            (agent_name, category, pattern, insight, success_count,
             failure_count, last_validated, is_active, tags)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
        """
        return await PostgresClient.fetchval(
            query,
            learning.agent_name,
            learning.category,
            learning.pattern,
            learning.insight,
            learning.success_count,
            learning.failure_count,
            learning.last_validated,
            learning.is_active,
            learning.tags
        )

    @staticmethod
    async def get_learning(learning_id: int) -> Optional[Learning]:
        """Get a learning by ID."""
        query = "SELECT * FROM learnings WHERE id = $1"
        row = await PostgresClient.fetchrow(query, learning_id)
        if row:
            return Learning(
                id=row["id"],
                agent_name=row["agent_name"],
                category=row["category"],
                pattern=row["pattern"],
                insight=row["insight"],
                success_count=row["success_count"],
                failure_count=row["failure_count"],
                last_validated=row["last_validated"],
                is_active=row["is_active"],
                tags=row["tags"] or [],
                created_at=row["created_at"]
            )
        return None

    @staticmethod
    async def get_learnings_by_tags(
        agent_name: str,
        tags: list[str],
        limit: int = 10
    ) -> list[Learning]:
        """Get active learnings matching any of the given tags."""
        query = """
            SELECT * FROM learnings
            WHERE agent_name = $1 AND is_active = true AND tags && $2::text[]
            ORDER BY (success_count - failure_count) DESC, created_at DESC
            LIMIT $3
        """
        rows = await PostgresClient.fetch(query, agent_name, tags, limit)
        return [
            Learning(
                id=row["id"],
                agent_name=row["agent_name"],
                category=row["category"],
                pattern=row["pattern"],
                insight=row["insight"],
                success_count=row["success_count"],
                failure_count=row["failure_count"],
                last_validated=row["last_validated"],
                is_active=row["is_active"],
                tags=row["tags"] or [],
                created_at=row["created_at"]
            )
            for row in rows
        ]

    @staticmethod
    async def get_learnings_by_category(
        agent_name: str,
        category: str,
        limit: int = 10
    ) -> list[Learning]:
        """Get active learnings by category."""
        query = """
            SELECT * FROM learnings
            WHERE agent_name = $1 AND category = $2 AND is_active = true
            ORDER BY (success_count - failure_count) DESC
            LIMIT $3
        """
        rows = await PostgresClient.fetch(query, agent_name, category, limit)
        return [
            Learning(
                id=row["id"],
                agent_name=row["agent_name"],
                category=row["category"],
                pattern=row["pattern"],
                insight=row["insight"],
                success_count=row["success_count"],
                failure_count=row["failure_count"],
                last_validated=row["last_validated"],
                is_active=row["is_active"],
                tags=row["tags"] or [],
                created_at=row["created_at"]
            )
            for row in rows
        ]

    @staticmethod
    async def get_top_learnings(
        agent_name: str,
        limit: int = 10
    ) -> list[Learning]:
        """Get top performing active learnings."""
        query = """
            SELECT * FROM learnings
            WHERE agent_name = $1 AND is_active = true
            ORDER BY (success_count - failure_count) DESC, success_count DESC
            LIMIT $2
        """
        rows = await PostgresClient.fetch(query, agent_name, limit)
        return [
            Learning(
                id=row["id"],
                agent_name=row["agent_name"],
                category=row["category"],
                pattern=row["pattern"],
                insight=row["insight"],
                success_count=row["success_count"],
                failure_count=row["failure_count"],
                last_validated=row["last_validated"],
                is_active=row["is_active"],
                tags=row["tags"] or [],
                created_at=row["created_at"]
            )
            for row in rows
        ]

    @staticmethod
    async def increment_learning_success(learning_id: int) -> None:
        """Increment success count for a learning."""
        query = """
            UPDATE learnings
            SET success_count = success_count + 1, last_validated = NOW()
            WHERE id = $1
        """
        await PostgresClient.execute(query, learning_id)

    @staticmethod
    async def increment_learning_failure(learning_id: int) -> None:
        """Increment failure count for a learning."""
        query = """
            UPDATE learnings
            SET failure_count = failure_count + 1, last_validated = NOW()
            WHERE id = $1
        """
        await PostgresClient.execute(query, learning_id)

    @staticmethod
    async def deactivate_learning(learning_id: int) -> None:
        """Deactivate a learning that is no longer valid."""
        query = "UPDATE learnings SET is_active = false WHERE id = $1"
        await PostgresClient.execute(query, learning_id)

    @staticmethod
    async def find_similar_learning(
        agent_name: str,
        pattern: str,
        tags: list[str]
    ) -> Optional[Learning]:
        """Find an existing learning with similar pattern and tags."""
        query = """
            SELECT * FROM learnings
            WHERE agent_name = $1
              AND pattern ILIKE $2
              AND tags && $3::text[]
              AND is_active = true
            LIMIT 1
        """
        row = await PostgresClient.fetchrow(query, agent_name, f"%{pattern}%", tags)
        if row:
            return Learning(
                id=row["id"],
                agent_name=row["agent_name"],
                category=row["category"],
                pattern=row["pattern"],
                insight=row["insight"],
                success_count=row["success_count"],
                failure_count=row["failure_count"],
                last_validated=row["last_validated"],
                is_active=row["is_active"],
                tags=row["tags"] or [],
                created_at=row["created_at"]
            )
        return None

    # ==================== Competition Scores ====================

    @staticmethod
    async def upsert_daily_score(score: CompetitionScore) -> int:
        """Insert or update daily competition score."""
        query = """
            INSERT INTO competition_scores
            (agent_name, date, starting_equity, ending_equity, daily_pnl,
             trades_count, wins, losses, strategies_used, top_learning_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (agent_name, date)
            DO UPDATE SET
                ending_equity = EXCLUDED.ending_equity,
                daily_pnl = EXCLUDED.daily_pnl,
                trades_count = EXCLUDED.trades_count,
                wins = EXCLUDED.wins,
                losses = EXCLUDED.losses,
                strategies_used = EXCLUDED.strategies_used,
                top_learning_id = EXCLUDED.top_learning_id
            RETURNING id
        """
        return await PostgresClient.fetchval(
            query,
            score.agent_name,
            score.date or date.today(),
            score.starting_equity,
            score.ending_equity,
            score.daily_pnl,
            score.trades_count,
            score.wins,
            score.losses,
            json.dumps(score.strategies_used),
            score.top_learning_id
        )

    @staticmethod
    async def get_competition_history(
        agent_name: str,
        days: int = 30
    ) -> list[CompetitionScore]:
        """Get competition score history."""
        query = """
            SELECT * FROM competition_scores
            WHERE agent_name = $1
            ORDER BY date DESC
            LIMIT $2
        """
        rows = await PostgresClient.fetch(query, agent_name, days)
        return [
            CompetitionScore(
                id=row["id"],
                agent_name=row["agent_name"],
                date=row["date"],
                starting_equity=row["starting_equity"],
                ending_equity=row["ending_equity"],
                daily_pnl=row["daily_pnl"],
                trades_count=row["trades_count"],
                wins=row["wins"],
                losses=row["losses"],
                strategies_used=json.loads(row["strategies_used"]) if row["strategies_used"] else {},
                top_learning_id=row["top_learning_id"]
            )
            for row in rows
        ]

    @staticmethod
    async def get_all_agents_latest_scores() -> list[CompetitionScore]:
        """Get latest scores for all agents."""
        query = """
            SELECT DISTINCT ON (agent_name) *
            FROM competition_scores
            ORDER BY agent_name, date DESC
        """
        rows = await PostgresClient.fetch(query)
        return [
            CompetitionScore(
                id=row["id"],
                agent_name=row["agent_name"],
                date=row["date"],
                starting_equity=row["starting_equity"],
                ending_equity=row["ending_equity"],
                daily_pnl=row["daily_pnl"],
                trades_count=row["trades_count"],
                wins=row["wins"],
                losses=row["losses"],
                strategies_used=json.loads(row["strategies_used"]) if row["strategies_used"] else {},
                top_learning_id=row["top_learning_id"]
            )
            for row in rows
        ]
