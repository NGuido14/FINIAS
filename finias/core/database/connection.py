from __future__ import annotations
from typing import Any, Optional
import asyncpg
import logging

from finias.core.config.settings import get_settings

logger = logging.getLogger("finias.database")


class DatabasePool:
    """
    Manages the async PostgreSQL connection pool.

    Usage:
        db = DatabasePool()
        await db.initialize()

        async with db.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM market_data WHERE symbol = $1", "SPY")

        await db.close()
    """

    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None

    async def initialize(self, min_size: int = 5, max_size: int = 20) -> None:
        """Create the connection pool."""
        settings = get_settings()
        self._pool = await asyncpg.create_pool(
            dsn=settings.postgres_dsn,
            min_size=min_size,
            max_size=max_size,
            command_timeout=30,
        )
        logger.info(f"Database pool initialized (min={min_size}, max={max_size})")

    def acquire(self):
        """Acquire a connection from the pool. Use as async context manager."""
        if self._pool is None:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")
        return self._pool.acquire()

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a query that doesn't return rows."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """Execute a query and return all rows."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Optional[asyncpg.Record]:
        """Execute a query and return one row."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Execute a query and return a single value."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Database pool closed")

    @property
    def is_initialized(self) -> bool:
        return self._pool is not None


# Singleton
_db_pool = None

async def get_db() -> DatabasePool:
    global _db_pool
    if _db_pool is None:
        _db_pool = DatabasePool()
        await _db_pool.initialize()
    return _db_pool
