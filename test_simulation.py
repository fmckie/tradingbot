#!/usr/bin/env python3
"""
Simulation test for the learning system - works without live market data.

Tests the full flow:
1. Agent recalls past learnings
2. Agent makes decision with mock market data
3. Episode is created
4. Outcome is processed
5. Reflection is generated
6. Learning is distilled
"""
import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()

from config.settings import POSTGRES_URL, LEARNING_ENABLED


# Mock market data
MOCK_MARKET_DATA = {
    "GOOGL": {
        "price": 178.50,
        "bid": 178.45,
        "ask": 178.55,
        "daily_open": 176.20,
        "daily_high": 179.80,
        "daily_low": 175.90,
        "daily_close": 178.50,
        "daily_volume": 15_000_000,
        "daily_change_percent": 1.3,
        "prev_close": 176.20,
        "rsi": 62.5,
        "macd_histogram": 0.45,
        "bollinger_percent_b": 0.72,
        "atr": 3.2,
        "vwap": 177.80,
        "above_vwap": True,
        "ema_9": 177.20,
        "ema_21": 175.50,
        "trend": "bullish"
    },
    "TSLA": {
        "price": 248.30,
        "bid": 248.20,
        "ask": 248.40,
        "daily_open": 245.00,
        "daily_high": 252.00,
        "daily_low": 244.50,
        "daily_close": 248.30,
        "daily_volume": 45_000_000,
        "daily_change_percent": 1.35,
        "prev_close": 245.00,
        "rsi": 58.2,
        "macd_histogram": 0.82,
        "bollinger_percent_b": 0.65,
        "atr": 8.5,
        "vwap": 247.50,
        "above_vwap": True,
        "ema_9": 246.80,
        "ema_21": 243.20,
        "trend": "bullish"
    }
}

MOCK_ACCOUNT = {
    "equity": 100000.00,
    "cash": 75000.00,
    "buying_power": 150000.00,
    "portfolio_value": 100000.00,
    "daily_pnl": 250.00,
    "daily_pnl_percent": 0.25
}

MOCK_POSITIONS = [
    {
        "symbol": "GOOGL",
        "quantity": 50,
        "avg_entry_price": 175.00,
        "current_price": 178.50,
        "market_value": 8925.00,
        "unrealized_pnl": 175.00,
        "unrealized_pnl_percent": 2.0
    }
]


async def run_simulation():
    """Run a full simulation of the learning system."""
    print("=" * 60)
    print("TRADING BOT LEARNING SYSTEM - SIMULATION TEST")
    print("(No live market data required)")
    print("=" * 60)

    if not POSTGRES_URL:
        print("\n[ERROR] DATABASE_URL not configured")
        return False

    # Import after env loaded
    from database.postgres_client import PostgresClient, init_database
    from database.learning_store import (
        LearningStore, Episode, Reflection, Learning,
        CompetitionScore, OutcomeStatus
    )
    from agents.base_agent import MarketContext

    # Initialize database
    print("\n1. INITIALIZING DATABASE")
    print("-" * 40)
    await init_database()
    print("   [OK] Database ready")

    # Seed some initial learnings for agents to recall
    print("\n2. SEEDING INITIAL LEARNINGS")
    print("-" * 40)

    seed_learnings = [
        Learning(
            agent_name="claude",
            category="indicator",
            pattern="RSI above 60 with price above VWAP indicates momentum",
            insight="Strong bullish signal when combined with positive MACD",
            success_count=8,
            failure_count=2,
            is_active=True,
            tags=["RSI", "VWAP", "MOMENTUM", "BULLISH", "GOOGL", "TSLA"]
        ),
        Learning(
            agent_name="claude",
            category="strategy",
            pattern="Trend following works best in clear directional markets",
            insight="Wait for EMA9 > EMA21 confirmation before entering",
            success_count=12,
            failure_count=4,
            is_active=True,
            tags=["TREND", "EMA", "STRATEGY"]
        ),
        Learning(
            agent_name="grok",
            category="indicator",
            pattern="MACD histogram positive with RSI between 50-70",
            insight="Good entry zone for momentum trades",
            success_count=6,
            failure_count=3,
            is_active=True,
            tags=["MACD", "RSI", "MOMENTUM", "GOOGL", "TSLA"]
        ),
        Learning(
            agent_name="grok",
            category="risk",
            pattern="Position size should account for ATR",
            insight="Use ATR to set stop-loss distance, reduces whipsaws",
            success_count=10,
            failure_count=1,
            is_active=True,
            tags=["ATR", "RISK", "STOP_LOSS"]
        ),
    ]

    for learning in seed_learnings:
        try:
            await LearningStore.create_learning(learning)
            print(f"   [OK] Seeded: {learning.pattern[:50]}...")
        except Exception as e:
            # Might already exist from previous run
            print(f"   [SKIP] {learning.pattern[:30]}... (may exist)")

    # Create mock market context
    print("\n3. CREATING MOCK MARKET CONTEXT")
    print("-" * 40)

    context = MarketContext(
        timestamp=datetime.now(),
        symbols=MOCK_MARKET_DATA,
        account=MOCK_ACCOUNT,
        positions=MOCK_POSITIONS,
        recent_trades=[],
        market_condition="bullish - both stocks trending up"
    )
    print(f"   Market: {context.market_condition}")
    print(f"   GOOGL: ${MOCK_MARKET_DATA['GOOGL']['price']} (RSI: {MOCK_MARKET_DATA['GOOGL']['rsi']})")
    print(f"   TSLA: ${MOCK_MARKET_DATA['TSLA']['price']} (RSI: {MOCK_MARKET_DATA['TSLA']['rsi']})")

    # Test recall for both agents
    print("\n4. TESTING LEARNING RECALL")
    print("-" * 40)

    # We need to create a minimal agent to test recall
    # Import the actual agents
    try:
        from agents.claude_agent import ClaudeAgent
        from agents.grok_agent import GrokAgent

        # Create agents with empty tools (we won't execute trades)
        claude = ClaudeAgent(tools={})
        grok = GrokAgent(tools={})

        # Test recall
        claude_learnings = await claude.recall_learnings(context)
        if claude_learnings:
            print(f"   [OK] Claude recalled learnings:")
            # Print first few lines
            for line in claude_learnings.split("\n")[2:6]:
                if line.strip():
                    print(f"       {line.strip()[:70]}")
        else:
            print("   [INFO] Claude has no learnings to recall yet")

        grok_learnings = await grok.recall_learnings(context)
        if grok_learnings:
            print(f"   [OK] Grok recalled learnings:")
            for line in grok_learnings.split("\n")[2:6]:
                if line.strip():
                    print(f"       {line.strip()[:70]}")
        else:
            print("   [INFO] Grok has no learnings to recall yet")

    except Exception as e:
        print(f"   [WARN] Could not test agent recall: {e}")

    # Simulate a complete episode cycle
    print("\n5. SIMULATING COMPLETE EPISODE CYCLE")
    print("-" * 40)

    # Create episode (as if agent decided to BUY TSLA)
    episode = Episode(
        agent_name="claude",
        timestamp=datetime.now(),
        market_regime="bullish",
        symbols_context=MOCK_MARKET_DATA,
        account_state=MOCK_ACCOUNT,
        decision_made={
            "action": "buy",
            "symbol": "TSLA",
            "quantity": 20,
            "strategy": "momentum",
            "reasoning": "RSI at 58 with positive MACD and price above VWAP suggests continued momentum. EMA9 > EMA21 confirms bullish trend.",
            "confidence": 0.75
        },
        outcome_status=OutcomeStatus.PENDING.value
    )
    episode_id = await LearningStore.create_episode(episode)
    print(f"   [OK] Created episode: id={episode_id}")
    print(f"       Decision: BUY 20 TSLA @ $248.30 (momentum)")

    # Simulate trade outcome (profit)
    await asyncio.sleep(0.5)  # Small delay to simulate time passing

    outcome_pnl = Decimal("156.40")  # Simulated profit
    outcome_status = OutcomeStatus.WIN.value

    await LearningStore.update_episode_outcome(episode_id, outcome_pnl, outcome_status)
    print(f"   [OK] Trade outcome: WIN (+${outcome_pnl})")

    # Generate reflection
    print("\n6. GENERATING REFLECTION")
    print("-" * 40)

    try:
        reflection_data = await claude.generate_reflection(
            episode_id=episode_id,
            decision_made=episode.decision_made,
            market_context=MOCK_MARKET_DATA,
            outcome_pnl=float(outcome_pnl),
            outcome_status=outcome_status
        )

        if reflection_data:
            print("   [OK] Claude generated reflection:")
            if reflection_data.get("what_worked"):
                print(f"       What worked: {reflection_data['what_worked'][:60]}...")
            if reflection_data.get("lesson_learned"):
                print(f"       Lesson: {reflection_data['lesson_learned'][:60]}...")
            if reflection_data.get("tags"):
                print(f"       Tags: {reflection_data['tags'][:5]}")

            # Store reflection
            reflection = Reflection(
                episode_id=episode_id,
                agent_name="claude",
                what_worked=reflection_data.get("what_worked", ""),
                what_failed=reflection_data.get("what_failed", ""),
                lesson_learned=reflection_data.get("lesson_learned", ""),
                next_time_will=reflection_data.get("next_time_will", ""),
                confidence_adjustment=Decimal("0.05"),
                tags=reflection_data.get("tags", [])
            )
            reflection_id = await LearningStore.create_reflection(reflection)
            print(f"   [OK] Stored reflection: id={reflection_id}")
        else:
            print("   [SKIP] No reflection generated (API key may be missing)")

    except Exception as e:
        print(f"   [WARN] Reflection generation failed: {e}")
        print("         (This is OK if ANTHROPIC_API_KEY is not set)")

    # Check final state
    print("\n7. VERIFYING FINAL STATE")
    print("-" * 40)

    episodes = await LearningStore.get_recent_episodes("claude", limit=5)
    print(f"   Claude episodes: {len(episodes)}")

    learnings = await LearningStore.get_top_learnings("claude", limit=5)
    print(f"   Claude learnings: {len(learnings)}")

    grok_learnings = await LearningStore.get_top_learnings("grok", limit=5)
    print(f"   Grok learnings: {len(grok_learnings)}")

    # Generate a mini report
    print("\n8. MINI COMPETITION REPORT")
    print("-" * 40)

    for agent in ["claude", "grok"]:
        agent_learnings = await LearningStore.get_top_learnings(agent, limit=3)
        print(f"\n   {agent.upper()}'s Top Learnings:")
        for i, l in enumerate(agent_learnings, 1):
            print(f"   {i}. [{l.category}] {l.pattern[:50]}...")
            print(f"      Success rate: {l.success_rate:.0f}% ({l.success_count}W/{l.failure_count}L)")

    # Cleanup option
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE!")
    print("=" * 60)
    print("\nData persisted in PostgreSQL. To clean up test data, run:")
    print("  DELETE FROM reflections WHERE agent_name IN ('claude', 'grok');")
    print("  DELETE FROM episodes WHERE agent_name IN ('claude', 'grok');")
    print("  -- Keep learnings for actual competition")

    await PostgresClient.close()
    return True


if __name__ == "__main__":
    asyncio.run(run_simulation())
