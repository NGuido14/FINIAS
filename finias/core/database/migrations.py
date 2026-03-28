from __future__ import annotations
import os
import logging
from pathlib import Path

from finias.core.database.connection import DatabasePool

logger = logging.getLogger("finias.migrations")

SCHEMAS_DIR = Path(__file__).parent / "schemas"


async def run_migrations(db: DatabasePool) -> None:
    """
    Apply any unapplied migrations.

    Reads .sql files from the schemas directory in alphabetical order.
    Each file is applied in a transaction. The schema_migrations table
    tracks which have been applied.
    """
    # Ensure migrations table exists
    await db.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(20) PRIMARY KEY,
            applied_at TIMESTAMPTZ DEFAULT NOW(),
            description TEXT
        )
    """)

    # Get applied migrations
    rows = await db.fetch("SELECT version FROM schema_migrations ORDER BY version")
    applied = {row["version"] for row in rows}

    # Find and apply new migrations
    sql_files = sorted(SCHEMAS_DIR.glob("v*.sql"))

    for sql_file in sql_files:
        version = sql_file.stem  # e.g., "v001_initial"
        if version in applied:
            continue

        logger.info(f"Applying migration: {version}")
        sql = sql_file.read_text()

        async with db.acquire() as conn:
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (version, description) VALUES ($1, $2)",
                    version, sql_file.name
                )

        logger.info(f"Migration applied: {version}")

    logger.info("All migrations up to date")
