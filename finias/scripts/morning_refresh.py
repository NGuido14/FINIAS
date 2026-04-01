"""
FINIAS Morning Macro Refresh

Runs the full Macro Strategist pipeline once and caches the result
in Redis for fast retrieval by the Director and downstream agents.

Usage:
    python -m finias.scripts.morning_refresh

Schedule with cron for daily operation:
    30 6 * * 1-5 cd /path/to/FINIAS && .venv/bin/python -m finias.scripts.morning_refresh >> logs/refresh.log 2>&1
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone

from finias.core.config.settings import get_settings
from finias.core.database.connection import DatabasePool
from finias.core.database.migrations import run_migrations
from finias.core.state.redis_state import RedisState
from finias.core.agents.models import AgentQuery
from finias.data.providers.polygon_client import PolygonClient
from finias.data.providers.fred_client import FredClient
from finias.data.cache.market_cache import MarketDataCache
from finias.agents.macro_strategist.agent import MacroStrategist
from finias.agents.macro_strategist.prompts.refresh import MORNING_REFRESH_PROMPT


logger = logging.getLogger("finias.scripts.morning_refresh")


async def main():
    """Run the morning macro refresh."""
    logging.basicConfig(
        level=getattr(logging, get_settings().log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler("finias_refresh.log"),
            logging.StreamHandler(sys.stdout),
        ]
    )

    start_time = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("  FINIAS MORNING MACRO REFRESH")
    logger.info(f"  Started: {start_time.isoformat()}")
    logger.info("=" * 60)

    # Initialize infrastructure
    db = DatabasePool()
    await db.initialize()
    await run_migrations(db)

    state = RedisState()
    await state.initialize()

    polygon = PolygonClient()
    fred = FredClient()
    cache = MarketDataCache(db=db, state=state, polygon=polygon, fred=fred)

    macro = MacroStrategist(cache=cache, state=state)

    try:
        # Run the full macro pipeline with comprehensive morning prompt
        logger.info("Running full macro pipeline...")
        query = AgentQuery(
            asking_agent="morning_refresh",
            target_agent="macro_strategist",
            question=MORNING_REFRESH_PROMPT.format(question=""),
            require_fresh_data=True,
        )

        opinion = await macro.query(query)

        # Log results
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(f"Refresh complete in {elapsed:.1f}s")
        logger.info(f"  Direction: {opinion.direction.value}")
        logger.info(f"  Confidence: {opinion.confidence.value}")
        logger.info(f"  Regime: {opinion.regime.value if opinion.regime else 'N/A'}")
        logger.info(f"  Key findings: {len(opinion.key_findings)}")
        logger.info(f"  Risks: {len(opinion.risks_to_view)}")

        # Verify Redis cache was updated
        cached = await state.get_regime()
        if cached:
            updated_at = cached.get("_updated_at", "unknown")
            logger.info(f"  Redis cache updated: {updated_at}")
        else:
            logger.error("  WARNING: Redis cache NOT updated!")

        # Mark data as fresh
        await state.mark_data_fresh("macro_refresh")

        logger.info("Morning refresh complete. Regime cached in Redis.")
        logger.info(f"  Total elapsed: {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"Morning refresh FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        await polygon.close()
        await fred.close()
        await state.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
