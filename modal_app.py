"""
Modal deployment for AI Trading Competition.

Runs the hourly trading cycle on a schedule during market hours.
Deploy with: modal deploy modal_app.py
"""
import modal
from datetime import datetime
import pytz

# Create Modal app
app = modal.App("ai-trading-competition")

# Create persistent volume for SQLite logs
volume = modal.Volume.from_name("trading-logs", create_if_missing=True)

# Define the image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "alpaca-py>=0.21.0",
        "anthropic>=0.18.0",
        "httpx>=0.27.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "ta>=0.11.0",
        "python-dotenv>=1.0.0",
        "schedule>=1.2.0",
        "rich>=13.0.0",
        "sqlite-utils>=3.35",
        "pytz>=2024.1",
        "asyncpg>=0.29.0",  # PostgreSQL learning system
    )
    .add_local_dir(".", "/app")
)


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("trading-secrets")],
    # Run every hour from 14:00-21:00 UTC Mon-Fri
    # This covers 9AM-4PM ET in both standard and daylight time
    schedule=modal.Cron("0 14-21 * * 1-5"),
    timeout=300,
)
def run_hourly_cycle():
    """Run the hourly trading decision cycle."""
    import asyncio
    import os
    import sys

    # Add app directory to path
    sys.path.insert(0, "/app")

    # Set working directory
    os.chdir("/app")

    # Use persistent volume for database
    os.environ["DATABASE_PATH"] = "/data/trading_competition.sqlite"

    from main import TradingCompetition
    from config.settings import TRADING_HOURS

    # Check if within trading window (with buffer)
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)

    # Calculate effective trading window
    trade_start_hour = TRADING_HOURS.market_open_hour
    trade_start_minute = TRADING_HOURS.market_open_minute + TRADING_HOURS.buffer_minutes
    trade_end_hour = TRADING_HOURS.market_close_hour
    trade_end_minute = TRADING_HOURS.market_close_minute - TRADING_HOURS.buffer_minutes

    # Handle minute overflow
    if trade_start_minute >= 60:
        trade_start_hour += 1
        trade_start_minute -= 60
    if trade_end_minute < 0:
        trade_end_hour -= 1
        trade_end_minute += 60

    # Create time boundaries
    trade_start = now.replace(
        hour=trade_start_hour, minute=trade_start_minute, second=0, microsecond=0
    )
    trade_end = now.replace(
        hour=trade_end_hour, minute=trade_end_minute, second=0, microsecond=0
    )

    # Check if it's a weekday and within trading window
    if now.weekday() >= 5:
        print(f"[SKIP] Weekend: {now.strftime('%A, %Y-%m-%d %H:%M ET')}")
        return {"status": "skipped", "reason": "weekend"}

    if now < trade_start:
        print(f"[SKIP] Before trading window: {now.strftime('%H:%M ET')} (opens at {trade_start.strftime('%H:%M ET')})")
        return {"status": "skipped", "reason": "before_market"}

    if now > trade_end:
        print(f"[SKIP] After trading window: {now.strftime('%H:%M ET')} (closed at {trade_end.strftime('%H:%M ET')})")
        return {"status": "skipped", "reason": "after_market"}

    print(f"[RUN] Trading cycle at {now.strftime('%Y-%m-%d %H:%M ET')}")

    async def _run_with_db_init():
        # Initialize PostgreSQL schema (idempotent)
        try:
            from database.postgres_client import init_database
            await init_database()
        except Exception as e:
            print(f"[WARN] Learning system unavailable: {e}")

        # Run the competition cycle
        competition = TradingCompetition()
        await competition.run_hourly_cycle()

    asyncio.run(_run_with_db_init())

    # Commit volume changes (persist SQLite)
    volume.commit()

    print("[DONE] Cycle completed successfully")
    return {"status": "completed", "timestamp": now.isoformat()}


@app.local_entrypoint()
def main(dry: bool = False):
    """Local entrypoint for manual testing.

    Args:
        dry: If True, run dry test (bypasses market hours). Otherwise run hourly cycle.
    """
    if dry:
        print("Running DRY RUN on Modal (logs will appear in dashboard)...")
        result = run_dry.remote()
    else:
        print("Running trading cycle on Modal (logs will appear in dashboard)...")
        result = run_hourly_cycle.remote()
    print(f"Result: {result}")


# For testing without schedule
@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("trading-secrets")],
    timeout=300,
)
def run_once():
    """Run a single cycle without schedule check - for testing."""
    import asyncio
    import os
    import sys

    sys.path.insert(0, "/app")
    os.chdir("/app")
    os.environ["DATABASE_PATH"] = "/data/trading_competition.sqlite"

    from main import TradingCompetition

    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    print(f"[TEST] Running test cycle at {now.strftime('%Y-%m-%d %H:%M ET')}")

    async def _run_with_db_init():
        # Initialize PostgreSQL schema (idempotent)
        try:
            from database.postgres_client import init_database
            await init_database()
        except Exception as e:
            print(f"[WARN] Learning system unavailable: {e}")

        # Run the competition cycle
        competition = TradingCompetition()
        await competition.run_hourly_cycle()

    asyncio.run(_run_with_db_init())

    volume.commit()
    return {"status": "test_completed", "timestamp": now.isoformat()}


# Dry run - bypasses market hours check for testing
@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("trading-secrets")],
    timeout=600,  # 10 min timeout for full cycle
)
def run_dry():
    """Run a full trading cycle ignoring market hours - for testing."""
    import asyncio
    import os
    import sys

    sys.path.insert(0, "/app")
    os.chdir("/app")
    os.environ["DATABASE_PATH"] = "/data/trading_competition.sqlite"

    from main import TradingCompetition

    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    print(f"[DRY RUN] Running full cycle at {now.strftime('%Y-%m-%d %H:%M ET')}")
    print("[DRY RUN] Market hours check DISABLED")

    async def _run_with_db_init():
        # Initialize PostgreSQL schema (idempotent)
        try:
            from database.postgres_client import init_database
            await init_database()
        except Exception as e:
            print(f"[WARN] Learning system unavailable: {e}")

        # Run with market check disabled
        competition = TradingCompetition(skip_market_check=True)
        await competition.run_hourly_cycle()

    asyncio.run(_run_with_db_init())

    volume.commit()
    return {"status": "dry_run_completed", "timestamp": now.isoformat()}
