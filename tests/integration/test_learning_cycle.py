"""Integration tests for the learning system flow.

Tests the complete learning cycle:
- Episode creation
- Trade outcome recording
- Reflection generation
- Learning storage and recall
- Learning deduplication
- Success/failure count updates
"""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from database.learning_store import (
    Episode,
    Reflection,
    Learning,
    OutcomeStatus,
    LearningStore,
)
from agents.base_agent import MarketContext


class TestEpisodeCreationFlow:
    """Test episode creation during decision making."""

    @pytest.mark.asyncio
    async def test_episode_creation_captures_market_context(self, in_memory_store):
        """Episode should capture full market context at decision time."""
        episode = Episode(
            agent_name="claude",
            timestamp=datetime.now(),
            market_regime="bullish - strong momentum",
            symbols_context={
                "GOOGL": {"price": 150.00, "rsi": 65.0, "macd": 0.5},
                "TSLA": {"price": 250.00, "rsi": 55.0, "macd": 0.2},
            },
            account_state={
                "equity": 100000.00,
                "cash": 80000.00,
                "positions_value": 20000.00,
            },
            decision_made={
                "action": "BUY",
                "symbol": "GOOGL",
                "quantity": 20,
                "strategy": "momentum",
                "reasoning": "RSI showing strong momentum",
            },
            outcome_status=OutcomeStatus.PENDING.value,
        )

        episode_id = await in_memory_store.create_episode(episode)

        # Verify episode was created
        assert episode_id is not None
        assert episode_id > 0

        # Retrieve and verify
        retrieved = await in_memory_store.get_episode(episode_id)
        assert retrieved is not None
        assert retrieved.agent_name == "claude"
        assert retrieved.market_regime == "bullish - strong momentum"
        assert "GOOGL" in retrieved.symbols_context
        assert retrieved.decision_made["action"] == "BUY"
        assert retrieved.outcome_status == OutcomeStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_episode_timestamps_are_recorded(self, in_memory_store):
        """Episode should have proper timestamps."""
        episode = Episode(
            agent_name="claude",
            timestamp=datetime.now(),
            market_regime="bullish",
            symbols_context={},
            account_state={},
            decision_made={"action": "HOLD"},
            outcome_status=OutcomeStatus.PENDING.value,
        )
        episode_id = await in_memory_store.create_episode(episode)

        retrieved = await in_memory_store.get_episode(episode_id)
        assert retrieved.created_at is not None
        # Just verify it's a datetime, avoid timezone comparison issues
        assert isinstance(retrieved.created_at, datetime)


class TestTradeOutcomeRecording:
    """Test recording trade outcomes."""

    @pytest.mark.asyncio
    async def test_win_outcome_updates_episode(self, in_memory_store):
        """Winning trade outcome should update episode correctly."""
        # Create episode
        episode = Episode(
            agent_name="claude",
            timestamp=datetime.now(),
            market_regime="bullish",
            symbols_context={"GOOGL": {"price": 150.00}},
            account_state={"equity": 100000.00},
            decision_made={
                "action": "BUY",
                "symbol": "GOOGL",
                "quantity": 10,
            },
            outcome_status=OutcomeStatus.PENDING.value,
        )
        episode_id = await in_memory_store.create_episode(episode)

        # Record winning outcome
        outcome_pnl = Decimal("250.00")  # Profit
        await in_memory_store.update_episode_outcome(
            episode_id, outcome_pnl, OutcomeStatus.WIN.value
        )

        # Verify update
        updated = await in_memory_store.get_episode(episode_id)
        assert updated.outcome_pnl == outcome_pnl
        assert updated.outcome_status == OutcomeStatus.WIN.value

    @pytest.mark.asyncio
    async def test_loss_outcome_updates_episode(self, in_memory_store):
        """Losing trade outcome should update episode correctly."""
        episode = Episode(
            agent_name="claude",
            timestamp=datetime.now(),
            market_regime="bullish",
            symbols_context={},
            account_state={},
            decision_made={"action": "BUY", "symbol": "GOOGL"},
            outcome_status=OutcomeStatus.PENDING.value,
        )
        episode_id = await in_memory_store.create_episode(episode)

        # Record losing outcome
        outcome_pnl = Decimal("-150.00")  # Loss
        await in_memory_store.update_episode_outcome(
            episode_id, outcome_pnl, OutcomeStatus.LOSS.value
        )

        updated = await in_memory_store.get_episode(episode_id)
        assert updated.outcome_pnl == outcome_pnl
        assert updated.outcome_status == OutcomeStatus.LOSS.value

    @pytest.mark.asyncio
    async def test_breakeven_outcome_updates_episode(self, in_memory_store):
        """Breakeven trade outcome should update episode correctly."""
        episode = Episode(
            agent_name="claude",
            timestamp=datetime.now(),
            market_regime="mixed",
            symbols_context={},
            account_state={},
            decision_made={"action": "BUY", "symbol": "TSLA"},
            outcome_status=OutcomeStatus.PENDING.value,
        )
        episode_id = await in_memory_store.create_episode(episode)

        # Record breakeven
        await in_memory_store.update_episode_outcome(
            episode_id, Decimal("0.00"), OutcomeStatus.BREAKEVEN.value
        )

        updated = await in_memory_store.get_episode(episode_id)
        assert updated.outcome_pnl == Decimal("0.00")
        assert updated.outcome_status == OutcomeStatus.BREAKEVEN.value

    @pytest.mark.asyncio
    async def test_hold_outcome_updates_episode(self, in_memory_store):
        """HOLD decision outcome should update episode correctly."""
        episode = Episode(
            agent_name="claude",
            timestamp=datetime.now(),
            market_regime="uncertain",
            symbols_context={},
            account_state={},
            decision_made={"action": "HOLD"},
            outcome_status=OutcomeStatus.PENDING.value,
        )
        episode_id = await in_memory_store.create_episode(episode)

        # HOLD doesn't have P&L
        await in_memory_store.update_episode_outcome(
            episode_id, Decimal("0.00"), OutcomeStatus.HOLD.value
        )

        updated = await in_memory_store.get_episode(episode_id)
        assert updated.outcome_status == OutcomeStatus.HOLD.value


class TestReflectionCreationFlow:
    """Test reflection creation after trade outcomes."""

    @pytest.mark.asyncio
    async def test_reflection_created_for_completed_episode(self, in_memory_store):
        """Reflection should be created after episode outcome."""
        # Create and complete episode
        episode = Episode(
            agent_name="claude",
            timestamp=datetime.now(),
            market_regime="bullish",
            symbols_context={"GOOGL": {"price": 150.00, "rsi": 70}},
            account_state={"equity": 100000.00},
            decision_made={
                "action": "BUY",
                "symbol": "GOOGL",
                "quantity": 10,
                "strategy": "momentum",
            },
            outcome_pnl=Decimal("200.00"),
            outcome_status=OutcomeStatus.WIN.value,
        )
        episode_id = await in_memory_store.create_episode(episode)

        # Create reflection
        reflection = Reflection(
            episode_id=episode_id,
            agent_name="claude",
            what_worked="RSI correctly indicated overbought bounce opportunity",
            what_failed="Entry could have been better timed",
            lesson_learned="High RSI in strong trend can still produce winners",
            next_time_will="Wait for slight pullback before entry",
            confidence_adjustment=Decimal("0.05"),
            tags=["GOOGL", "momentum", "rsi", "bullish"],
        )
        reflection_id = await in_memory_store.create_reflection(reflection)

        # Verify
        retrieved = await in_memory_store.get_reflection(reflection_id)
        assert retrieved is not None
        assert retrieved.episode_id == episode_id
        assert retrieved.what_worked is not None
        assert "GOOGL" in retrieved.tags

    @pytest.mark.asyncio
    async def test_reflection_captures_failure_analysis(self, in_memory_store):
        """Reflection should capture failure analysis for losing trades."""
        episode = Episode(
            agent_name="claude",
            timestamp=datetime.now(),
            market_regime="bearish",
            symbols_context={"TSLA": {"price": 250.00, "rsi": 30}},
            account_state={"equity": 100000.00},
            decision_made={
                "action": "BUY",
                "symbol": "TSLA",
                "strategy": "mean_reversion",
            },
            outcome_pnl=Decimal("-300.00"),
            outcome_status=OutcomeStatus.LOSS.value,
        )
        episode_id = await in_memory_store.create_episode(episode)

        reflection = Reflection(
            episode_id=episode_id,
            agent_name="claude",
            what_worked="",  # Nothing worked
            what_failed="Bought into falling knife - trend was too strong",
            lesson_learned="Don't catch falling knives in strong bearish trend",
            next_time_will="Wait for trend reversal confirmation",
            confidence_adjustment=Decimal("-0.10"),  # Negative adjustment
            tags=["TSLA", "mean_reversion", "bearish", "falling_knife"],
        )
        reflection_id = await in_memory_store.create_reflection(reflection)

        retrieved = await in_memory_store.get_reflection(reflection_id)
        assert retrieved.what_failed != ""
        assert retrieved.confidence_adjustment == Decimal("-0.10")


class TestLearningStorageFlow:
    """Test distilling reflections into learnings."""

    @pytest.mark.asyncio
    async def test_learning_created_from_reflection(self, in_memory_store):
        """Learning should be created from reflection insights."""
        learning = Learning(
            agent_name="claude",
            category="indicator",
            pattern="RSI above 70 in bullish market",
            insight="High RSI in strong uptrend signals continuation, not reversal",
            success_count=1,
            failure_count=0,
            is_active=True,
            tags=["rsi", "bullish", "momentum"],
        )
        learning_id = await in_memory_store.create_learning(learning)

        retrieved = await in_memory_store.get_learning(learning_id)
        assert retrieved is not None
        assert retrieved.category == "indicator"
        assert retrieved.is_active is True
        assert retrieved.success_count == 1

    @pytest.mark.asyncio
    async def test_learning_success_rate_calculation(self, in_memory_store):
        """Learning success rate should be calculated correctly."""
        learning = Learning(
            agent_name="claude",
            category="strategy",
            pattern="Morning breakout trade",
            insight="Trade breakouts in first hour",
            success_count=8,
            failure_count=2,
            is_active=True,
            tags=["breakout", "timing"],
        )
        learning_id = await in_memory_store.create_learning(learning)

        retrieved = await in_memory_store.get_learning(learning_id)
        assert retrieved.success_rate == 80.0  # 8/10 = 80%

    @pytest.mark.asyncio
    async def test_learning_categories(self, in_memory_store):
        """Learnings should support different categories."""
        categories = ["strategy", "indicator", "timing", "risk"]

        for category in categories:
            learning = Learning(
                agent_name="claude",
                category=category,
                pattern=f"Pattern for {category}",
                insight=f"Insight for {category}",
                success_count=1,
                failure_count=0,
                is_active=True,
                tags=[category],
            )
            learning_id = await in_memory_store.create_learning(learning)
            retrieved = await in_memory_store.get_learning(learning_id)
            assert retrieved.category == category


class TestLearningRecallBasedOnMarketRegime:
    """Test learning recall based on market conditions."""

    @pytest.mark.asyncio
    async def test_recall_learnings_by_regime_tags(
        self, in_memory_store, sample_learnings_list
    ):
        """Learnings should be recalled based on market regime tags."""
        # Store learnings
        for learning in sample_learnings_list:
            await in_memory_store.create_learning(learning)

        # Query for bullish regime
        bullish_learnings = await in_memory_store.get_learnings_by_tags(
            agent_name="claude",
            tags=["bullish"],
            limit=10
        )

        # Should find the RSI oversold bounce learning
        assert len(bullish_learnings) >= 1
        patterns = [l.pattern for l in bullish_learnings]
        assert any("RSI" in p for p in patterns)

    @pytest.mark.asyncio
    async def test_recall_learnings_by_symbol_tags(
        self, in_memory_store, sample_learnings_list
    ):
        """Learnings should be recalled based on symbol tags."""
        for learning in sample_learnings_list:
            await in_memory_store.create_learning(learning)

        # Query for GOOGL
        googl_learnings = await in_memory_store.get_learnings_by_tags(
            agent_name="claude",
            tags=["GOOGL"],
            limit=10
        )

        assert len(googl_learnings) >= 1
        # All returned should have GOOGL in tags
        for learning in googl_learnings:
            assert "GOOGL" in learning.tags

    @pytest.mark.asyncio
    async def test_recall_top_performing_learnings(
        self, in_memory_store, sample_learnings_list
    ):
        """Top learnings should be sorted by success - failure."""
        for learning in sample_learnings_list:
            await in_memory_store.create_learning(learning)

        top_learnings = await in_memory_store.get_top_learnings(
            agent_name="claude",
            limit=10
        )

        # Should be sorted by (success - failure) descending
        for i in range(len(top_learnings) - 1):
            current_score = top_learnings[i].success_count - top_learnings[i].failure_count
            next_score = top_learnings[i+1].success_count - top_learnings[i+1].failure_count
            assert current_score >= next_score

    @pytest.mark.asyncio
    async def test_inactive_learnings_not_recalled(self, in_memory_store):
        """Inactive learnings should not be returned in queries."""
        # Create active learning
        active = Learning(
            agent_name="claude",
            category="strategy",
            pattern="Active pattern",
            insight="Active insight",
            success_count=5,
            failure_count=0,
            is_active=True,
            tags=["test"],
        )
        active_id = await in_memory_store.create_learning(active)

        # Create inactive learning
        inactive = Learning(
            agent_name="claude",
            category="strategy",
            pattern="Inactive pattern",
            insight="Inactive insight",
            success_count=10,  # Higher score but inactive
            failure_count=0,
            is_active=False,
            tags=["test"],
        )
        await in_memory_store.create_learning(inactive)

        # Query
        results = await in_memory_store.get_learnings_by_tags(
            agent_name="claude",
            tags=["test"],
            limit=10
        )

        # Should only return active
        assert len(results) == 1
        assert results[0].pattern == "Active pattern"


class TestLearningDeduplication:
    """Test learning deduplication to avoid redundant entries."""

    @pytest.mark.asyncio
    async def test_find_similar_learning(self, in_memory_store):
        """Should find similar existing learning before creating new."""
        # Create existing learning
        existing = Learning(
            agent_name="claude",
            category="indicator",
            pattern="RSI oversold bounce",
            insight="Buy when RSI drops below 30",
            success_count=3,
            failure_count=1,
            is_active=True,
            tags=["rsi", "oversold", "bullish"],
        )
        await in_memory_store.create_learning(existing)

        # Try to find similar
        similar = await in_memory_store.find_similar_learning(
            agent_name="claude",
            pattern="RSI oversold",  # Partial match
            tags=["rsi", "oversold"]
        )

        assert similar is not None
        assert similar.pattern == "RSI oversold bounce"

    @pytest.mark.asyncio
    async def test_no_duplicate_when_similar_exists(self, in_memory_store):
        """Should update existing learning instead of creating duplicate."""
        # Create initial learning
        learning = Learning(
            agent_name="claude",
            category="strategy",
            pattern="Morning momentum trade",
            insight="First hour momentum tends to continue",
            success_count=2,
            failure_count=0,
            is_active=True,
            tags=["momentum", "timing"],
        )
        learning_id = await in_memory_store.create_learning(learning)

        # Find and update instead of creating new
        similar = await in_memory_store.find_similar_learning(
            agent_name="claude",
            pattern="Morning momentum",
            tags=["momentum"]
        )

        if similar:
            # Update existing
            await in_memory_store.increment_learning_success(similar.id)
        else:
            # Create new (shouldn't happen in this test)
            pytest.fail("Should have found similar learning")

        # Verify update
        updated = await in_memory_store.get_learning(learning_id)
        assert updated.success_count == 3  # Incremented

    @pytest.mark.asyncio
    async def test_agent_isolation_in_deduplication(self, in_memory_store):
        """Similar learning search should be isolated per agent."""
        # Claude's learning
        claude_learning = Learning(
            agent_name="claude",
            category="strategy",
            pattern="RSI strategy",
            insight="Claude's RSI insight",
            success_count=5,
            failure_count=0,
            is_active=True,
            tags=["rsi"],
        )
        await in_memory_store.create_learning(claude_learning)

        # Grok's similar learning
        grok_learning = Learning(
            agent_name="grok",
            category="strategy",
            pattern="RSI strategy",
            insight="Grok's RSI insight",
            success_count=3,
            failure_count=0,
            is_active=True,
            tags=["rsi"],
        )
        await in_memory_store.create_learning(grok_learning)

        # Search for Grok should not find Claude's
        grok_similar = await in_memory_store.find_similar_learning(
            agent_name="grok",
            pattern="RSI",
            tags=["rsi"]
        )

        assert grok_similar is not None
        assert grok_similar.agent_name == "grok"


class TestSuccessFailureCountUpdates:
    """Test updating success/failure counts on learnings."""

    @pytest.mark.asyncio
    async def test_increment_success_count(self, in_memory_store):
        """Success count should increment correctly."""
        learning = Learning(
            agent_name="claude",
            category="strategy",
            pattern="Test pattern",
            insight="Test insight",
            success_count=5,
            failure_count=2,
            is_active=True,
            tags=["test"],
        )
        learning_id = await in_memory_store.create_learning(learning)

        await in_memory_store.increment_learning_success(learning_id)

        updated = await in_memory_store.get_learning(learning_id)
        assert updated.success_count == 6
        assert updated.failure_count == 2  # Unchanged

    @pytest.mark.asyncio
    async def test_increment_failure_count(self, in_memory_store):
        """Failure count should increment correctly."""
        learning = Learning(
            agent_name="claude",
            category="strategy",
            pattern="Test pattern",
            insight="Test insight",
            success_count=5,
            failure_count=2,
            is_active=True,
            tags=["test"],
        )
        learning_id = await in_memory_store.create_learning(learning)

        await in_memory_store.increment_learning_failure(learning_id)

        updated = await in_memory_store.get_learning(learning_id)
        assert updated.success_count == 5  # Unchanged
        assert updated.failure_count == 3

    @pytest.mark.asyncio
    async def test_last_validated_updates_on_increment(self, in_memory_store):
        """Last validated timestamp should update when count changes."""
        learning = Learning(
            agent_name="claude",
            category="strategy",
            pattern="Test pattern",
            insight="Test insight",
            success_count=0,
            failure_count=0,
            last_validated=None,
            is_active=True,
            tags=["test"],
        )
        learning_id = await in_memory_store.create_learning(learning)

        await in_memory_store.increment_learning_success(learning_id)

        updated = await in_memory_store.get_learning(learning_id)
        assert updated.last_validated is not None
        # Just verify it's a datetime, avoid timezone comparison issues
        assert isinstance(updated.last_validated, datetime)

    @pytest.mark.asyncio
    async def test_success_rate_changes_with_counts(self, in_memory_store):
        """Success rate should update as counts change."""
        learning = Learning(
            agent_name="claude",
            category="strategy",
            pattern="Test pattern",
            insight="Test insight",
            success_count=4,
            failure_count=1,  # 80% initial
            is_active=True,
            tags=["test"],
        )
        learning_id = await in_memory_store.create_learning(learning)

        # Initial rate
        initial = await in_memory_store.get_learning(learning_id)
        assert initial.success_rate == 80.0

        # Add failure
        await in_memory_store.increment_learning_failure(learning_id)

        # New rate should be lower
        updated = await in_memory_store.get_learning(learning_id)
        # 4 success, 2 failures = 66.67%
        assert abs(updated.success_rate - 66.67) < 0.1


class TestLearningDeactivation:
    """Test deactivating learnings that are no longer valid."""

    @pytest.mark.asyncio
    async def test_deactivate_learning(self, in_memory_store):
        """Deactivated learning should not appear in queries."""
        learning = Learning(
            agent_name="claude",
            category="strategy",
            pattern="Outdated pattern",
            insight="Outdated insight",
            success_count=10,
            failure_count=0,
            is_active=True,
            tags=["test"],
        )
        learning_id = await in_memory_store.create_learning(learning)

        # Verify it's active
        active_results = await in_memory_store.get_learnings_by_tags(
            agent_name="claude", tags=["test"], limit=10
        )
        assert len(active_results) == 1

        # Deactivate
        await in_memory_store.deactivate_learning(learning_id)

        # Should not appear in queries
        results = await in_memory_store.get_learnings_by_tags(
            agent_name="claude", tags=["test"], limit=10
        )
        assert len(results) == 0

        # But should still exist (soft delete)
        still_exists = await in_memory_store.get_learning(learning_id)
        assert still_exists is not None
        assert still_exists.is_active is False


class TestCompleteLearningCycle:
    """Test the complete learning cycle flow."""

    @pytest.mark.asyncio
    async def test_full_learning_cycle(self, in_memory_store):
        """Test complete cycle: Episode -> Outcome -> Reflection -> Learning -> Recall."""
        # 1. Create episode (decision made)
        episode = Episode(
            agent_name="claude",
            timestamp=datetime.now(),
            market_regime="bullish - strong",
            symbols_context={
                "GOOGL": {"price": 150.00, "rsi": 65.0, "trend": "up"}
            },
            account_state={"equity": 100000.00},
            decision_made={
                "action": "BUY",
                "symbol": "GOOGL",
                "quantity": 20,
                "strategy": "momentum",
                "reasoning": "Strong uptrend with RSI confirmation",
            },
            outcome_status=OutcomeStatus.PENDING.value,
        )
        episode_id = await in_memory_store.create_episode(episode)

        # 2. Record outcome (trade closed)
        await in_memory_store.update_episode_outcome(
            episode_id, Decimal("350.00"), OutcomeStatus.WIN.value
        )

        # 3. Create reflection (analyze what happened)
        reflection = Reflection(
            episode_id=episode_id,
            agent_name="claude",
            what_worked="RSI above 60 in bullish market correctly predicted continuation",
            what_failed="Could have held longer for more profit",
            lesson_learned="RSI 60-70 in uptrend is a continuation signal, not overbought",
            next_time_will="Hold position longer when trend is strong",
            confidence_adjustment=Decimal("0.05"),
            tags=["GOOGL", "momentum", "rsi", "bullish", "continuation"],
        )
        await in_memory_store.create_reflection(reflection)

        # 4. Distill into learning
        learning = Learning(
            agent_name="claude",
            category="indicator",
            pattern="RSI 60-70 in bullish trend",
            insight="RSI in 60-70 range during strong uptrend indicates continuation",
            success_count=1,
            failure_count=0,
            is_active=True,
            tags=["rsi", "bullish", "continuation", "GOOGL"],
        )
        learning_id = await in_memory_store.create_learning(learning)

        # 5. Recall learning for similar context
        recalled = await in_memory_store.get_learnings_by_tags(
            agent_name="claude",
            tags=["bullish", "rsi"],
            limit=5
        )

        # Verify cycle completed
        assert len(recalled) >= 1
        found_learning = next(
            (l for l in recalled if "RSI 60-70" in l.pattern), None
        )
        assert found_learning is not None
        assert found_learning.success_count == 1

    @pytest.mark.asyncio
    async def test_learning_accumulates_over_episodes(self, in_memory_store):
        """Learning should accumulate evidence over multiple similar episodes."""
        # Create initial learning
        learning = Learning(
            agent_name="claude",
            category="timing",
            pattern="Morning momentum trade (9:45-10:30 ET)",
            insight="First hour after open shows strong momentum continuation",
            success_count=0,
            failure_count=0,
            is_active=True,
            tags=["timing", "momentum", "morning"],
        )
        learning_id = await in_memory_store.create_learning(learning)

        # Simulate 5 successful trades using this pattern
        for i in range(5):
            episode = Episode(
                agent_name="claude",
                timestamp=datetime.now(),
                market_regime="bullish",
                symbols_context={},
                account_state={},
                decision_made={"action": "BUY", "strategy": "momentum"},
                outcome_pnl=Decimal("100.00"),
                outcome_status=OutcomeStatus.WIN.value,
            )
            await in_memory_store.create_episode(episode)
            await in_memory_store.increment_learning_success(learning_id)

        # Simulate 1 failed trade
        await in_memory_store.increment_learning_failure(learning_id)

        # Check accumulated stats
        final = await in_memory_store.get_learning(learning_id)
        assert final.success_count == 5
        assert final.failure_count == 1
        assert abs(final.success_rate - 83.33) < 0.1  # ~83.33%

    @pytest.mark.asyncio
    async def test_multiple_agents_have_separate_learnings(self, in_memory_store):
        """Claude and Grok should have completely separate learnings."""
        # Claude's learning
        claude_learning = Learning(
            agent_name="claude",
            category="strategy",
            pattern="Claude's momentum strategy",
            insight="Claude's approach to momentum",
            success_count=10,
            failure_count=2,
            is_active=True,
            tags=["momentum"],
        )
        await in_memory_store.create_learning(claude_learning)

        # Grok's learning
        grok_learning = Learning(
            agent_name="grok",
            category="strategy",
            pattern="Grok's momentum strategy",
            insight="Grok's approach to momentum",
            success_count=8,
            failure_count=4,
            is_active=True,
            tags=["momentum"],
        )
        await in_memory_store.create_learning(grok_learning)

        # Query each agent
        claude_results = await in_memory_store.get_learnings_by_tags(
            agent_name="claude", tags=["momentum"], limit=10
        )
        grok_results = await in_memory_store.get_learnings_by_tags(
            agent_name="grok", tags=["momentum"], limit=10
        )

        # Should be completely separate
        assert len(claude_results) == 1
        assert len(grok_results) == 1
        assert claude_results[0].agent_name == "claude"
        assert grok_results[0].agent_name == "grok"
        assert claude_results[0].success_count == 10
        assert grok_results[0].success_count == 8
