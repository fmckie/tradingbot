#!/usr/bin/env python3
"""Test script for the PostgreSQL learning system."""
import asyncio
import sys
from datetime import datetime, date
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()

from config.settings import POSTGRES_URL, LEARNING_ENABLED


async def test_learning_system():
    """Run comprehensive tests on the learning system."""
    print("=" * 60)
    print("TRADING BOT LEARNING SYSTEM - TEST SUITE")
    print("=" * 60)

    # Check config
    print("\n1. CONFIGURATION CHECK")
    print("-" * 40)
    if not POSTGRES_URL:
        print("   [FAIL] DATABASE_URL not configured")
        return False
    print(f"   [OK] DATABASE_URL: {POSTGRES_URL[:50]}...")
    print(f"   [OK] LEARNING_ENABLED: {LEARNING_ENABLED}")

    # Import after loading env
    from database.postgres_client import PostgresClient, init_database
    from database.learning_store import (
        LearningStore, Episode, Reflection, Learning,
        CompetitionScore, OutcomeStatus
    )

    # Test connection
    print("\n2. DATABASE CONNECTION")
    print("-" * 40)
    try:
        healthy = await PostgresClient.health_check()
        if healthy:
            print("   [OK] Connected to PostgreSQL")
        else:
            print("   [FAIL] Connection health check failed")
            return False
    except Exception as e:
        print(f"   [FAIL] Connection error: {e}")
        return False

    # Initialize schema
    print("\n3. SCHEMA INITIALIZATION")
    print("-" * 40)
    try:
        await init_database()
        print("   [OK] Schema initialized successfully")
    except Exception as e:
        print(f"   [FAIL] Schema init error: {e}")
        return False

    # Verify tables exist
    print("\n4. TABLE VERIFICATION")
    print("-" * 40)
    tables = ["episodes", "reflections", "learnings", "competition_scores"]
    for table in tables:
        try:
            count = await PostgresClient.fetchval(f"SELECT COUNT(*) FROM {table}")
            print(f"   [OK] {table}: {count} rows")
        except Exception as e:
            print(f"   [FAIL] {table}: {e}")
            return False

    # Test Episode CRUD
    print("\n5. EPISODE CRUD TEST")
    print("-" * 40)
    try:
        episode = Episode(
            agent_name="test_agent",
            timestamp=datetime.now(),
            market_regime="bullish - test",
            symbols_context={"GOOGL": {"price": 150.0, "rsi": 55.0}},
            account_state={"equity": 100000.0, "cash": 50000.0},
            decision_made={
                "action": "buy",
                "symbol": "GOOGL",
                "quantity": 10,
                "strategy": "momentum"
            },
            outcome_status=OutcomeStatus.PENDING.value
        )
        episode_id = await LearningStore.create_episode(episode)
        print(f"   [OK] Created episode: id={episode_id}")

        # Retrieve
        retrieved = await LearningStore.get_episode(episode_id)
        if retrieved and retrieved.agent_name == "test_agent":
            print(f"   [OK] Retrieved episode: {retrieved.market_regime}")
        else:
            print("   [FAIL] Episode retrieval mismatch")
            return False

        # Update outcome
        await LearningStore.update_episode_outcome(
            episode_id, Decimal("125.50"), OutcomeStatus.WIN.value
        )
        updated = await LearningStore.get_episode(episode_id)
        if updated.outcome_status == "win":
            print(f"   [OK] Updated outcome: {updated.outcome_status}, P&L: ${updated.outcome_pnl}")
        else:
            print("   [FAIL] Outcome update failed")
            return False

    except Exception as e:
        print(f"   [FAIL] Episode CRUD error: {e}")
        return False

    # Test Reflection CRUD
    print("\n6. REFLECTION CRUD TEST")
    print("-" * 40)
    try:
        reflection = Reflection(
            episode_id=episode_id,
            agent_name="test_agent",
            what_worked="RSI signal was accurate",
            what_failed="Entered slightly too early",
            lesson_learned="Wait for RSI to cross back above 30 for confirmation",
            next_time_will="Use RSI + MACD confirmation together",
            confidence_adjustment=Decimal("0.05"),
            tags=["RSI", "GOOGL", "MOMENTUM", "TEST"]
        )
        reflection_id = await LearningStore.create_reflection(reflection)
        print(f"   [OK] Created reflection: id={reflection_id}")

        # Retrieve by tags
        reflections = await LearningStore.get_reflections_by_tags(
            "test_agent", ["RSI", "TEST"], limit=5
        )
        if reflections:
            print(f"   [OK] Retrieved {len(reflections)} reflection(s) by tags")
        else:
            print("   [FAIL] Tag-based retrieval failed")
            return False

    except Exception as e:
        print(f"   [FAIL] Reflection CRUD error: {e}")
        return False

    # Test Learning CRUD
    print("\n7. LEARNING CRUD TEST")
    print("-" * 40)
    try:
        learning = Learning(
            agent_name="test_agent",
            category="indicator",
            pattern="When RSI < 30 and MACD histogram turning positive",
            insight="High probability reversal setup, but wait for confirmation candle",
            success_count=5,
            failure_count=2,
            is_active=True,
            tags=["RSI", "MACD", "REVERSAL", "GOOGL", "TEST"]
        )
        learning_id = await LearningStore.create_learning(learning)
        print(f"   [OK] Created learning: id={learning_id}")

        # Retrieve by tags
        learnings = await LearningStore.get_learnings_by_tags(
            "test_agent", ["RSI", "GOOGL"], limit=5
        )
        if learnings:
            print(f"   [OK] Retrieved {len(learnings)} learning(s) by tags")
            for l in learnings:
                print(f"       - {l.pattern[:50]}... ({l.success_rate:.0f}% success)")
        else:
            print("   [FAIL] Learning retrieval failed")
            return False

        # Test increment
        await LearningStore.increment_learning_success(learning_id)
        updated_learning = await LearningStore.get_learning(learning_id)
        if updated_learning.success_count == 6:
            print(f"   [OK] Incremented success count: {updated_learning.success_count}")
        else:
            print("   [FAIL] Increment failed")
            return False

        # Get top learnings
        top = await LearningStore.get_top_learnings("test_agent", limit=3)
        print(f"   [OK] Top learnings: {len(top)} retrieved")

    except Exception as e:
        print(f"   [FAIL] Learning CRUD error: {e}")
        return False

    # Test Competition Score
    print("\n8. COMPETITION SCORE TEST")
    print("-" * 40)
    try:
        score = CompetitionScore(
            agent_name="test_agent",
            date=date.today(),
            starting_equity=Decimal("100000.00"),
            ending_equity=Decimal("100250.50"),
            daily_pnl=Decimal("250.50"),
            trades_count=5,
            wins=3,
            losses=2,
            strategies_used={"momentum": 3, "mean_reversion": 2}
        )
        score_id = await LearningStore.upsert_daily_score(score)
        print(f"   [OK] Upserted daily score: id={score_id}")

        # Get history
        history = await LearningStore.get_competition_history("test_agent", days=7)
        print(f"   [OK] Competition history: {len(history)} day(s)")

    except Exception as e:
        print(f"   [FAIL] Competition score error: {e}")
        return False

    # Cleanup test data
    print("\n9. CLEANUP")
    print("-" * 40)
    try:
        await PostgresClient.execute(
            "DELETE FROM reflections WHERE agent_name = $1", "test_agent"
        )
        await PostgresClient.execute(
            "DELETE FROM learnings WHERE agent_name = $1", "test_agent"
        )
        await PostgresClient.execute(
            "DELETE FROM episodes WHERE agent_name = $1", "test_agent"
        )
        await PostgresClient.execute(
            "DELETE FROM competition_scores WHERE agent_name = $1", "test_agent"
        )
        print("   [OK] Test data cleaned up")
    except Exception as e:
        print(f"   [WARN] Cleanup error: {e}")

    # Close connection
    await PostgresClient.close()
    print("   [OK] Connection closed")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_learning_system())
    sys.exit(0 if success else 1)
