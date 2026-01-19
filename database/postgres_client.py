"""PostgreSQL client with connection pooling for Neon serverless."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Optional
import asyncpg
from asyncpg import Pool, Connection

from config.settings import POSTGRES_URL, POSTGRES_POOL_SIZE


class PostgresClient:
    """
    PostgreSQL client with async connection pooling.

    Designed for Neon serverless PostgreSQL - handles cold starts gracefully.
    """

    _instance: Optional["PostgresClient"] = None
    _pool: Optional[Pool] = None

    def __new__(cls) -> "PostgresClient":
        """Singleton pattern for connection pool reuse."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def get_pool(cls) -> Pool:
        """Get or create the connection pool."""
        if cls._pool is None or cls._pool._closed:
            if not POSTGRES_URL:
                raise ValueError(
                    "DATABASE_URL not set. Please configure your Neon PostgreSQL connection string."
                )

            cls._pool = await asyncpg.create_pool(
                POSTGRES_URL,
                min_size=1,
                max_size=POSTGRES_POOL_SIZE,
                command_timeout=30,
                # Neon-specific: handle cold starts
                server_settings={
                    "application_name": "tradingbot-learning"
                }
            )
        return cls._pool

    @classmethod
    @asynccontextmanager
    async def connection(cls) -> AsyncGenerator[Connection, None]:
        """Get a connection from the pool."""
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            yield conn

    @classmethod
    async def execute(cls, query: str, *args: Any) -> str:
        """Execute a query that doesn't return results."""
        async with cls.connection() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def fetch(cls, query: str, *args: Any) -> list[asyncpg.Record]:
        """Execute a query and fetch all results."""
        async with cls.connection() as conn:
            return await conn.fetch(query, *args)

    @classmethod
    async def fetchrow(cls, query: str, *args: Any) -> Optional[asyncpg.Record]:
        """Execute a query and fetch one result."""
        async with cls.connection() as conn:
            return await conn.fetchrow(query, *args)

    @classmethod
    async def fetchval(cls, query: str, *args: Any) -> Any:
        """Execute a query and fetch a single value."""
        async with cls.connection() as conn:
            return await conn.fetchval(query, *args)

    @classmethod
    async def init_schema(cls) -> None:
        """Initialize the database schema from schema.sql."""
        schema_path = Path(__file__).parent / "schema.sql"

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        schema_sql = schema_path.read_text()

        async with cls.connection() as conn:
            await conn.execute(schema_sql)

    @classmethod
    async def close(cls) -> None:
        """Close the connection pool."""
        if cls._pool is not None and not cls._pool._closed:
            await cls._pool.close()
            cls._pool = None

    @classmethod
    async def health_check(cls) -> bool:
        """Check if the database connection is healthy."""
        try:
            result = await cls.fetchval("SELECT 1")
            return result == 1
        except Exception:
            return False


async def init_database() -> None:
    """Initialize the database (call on startup)."""
    if not POSTGRES_URL:
        print("Warning: DATABASE_URL not set - PostgreSQL learning system disabled")
        return

    try:
        await PostgresClient.init_schema()
        print("PostgreSQL schema initialized successfully")
    except Exception as e:
        print(f"Failed to initialize PostgreSQL schema: {e}")
        raise
