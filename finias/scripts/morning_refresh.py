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
        # Fetch live prices and store in Redis (shared infrastructure)
        logger.info("Fetching live market prices...")
        try:
            from finias.data.providers.price_feed import fetch_live_prices, store_live_prices
            live_prices = await fetch_live_prices()
            await store_live_prices(state, live_prices)
            fetched = {k: v for k, v in live_prices.items()
                       if k not in ("fetched_at", "source", "error") and v is not None}
            logger.info(f"  Live prices: {', '.join(k + '=' + str(v) for k, v in fetched.items())}")
        except Exception as e:
            logger.warning(f"  Live price feed unavailable: {e}")

        # Fetch CFTC COT positioning data (weekly — checks for new data)
        logger.info("Checking CFTC COT positioning data...")
        try:
            from finias.data.providers.cot_client import fetch_and_store_cot_data
            cot_result = await fetch_and_store_cot_data(db)
            if cot_result.get("error"):
                logger.warning(f"  COT fetch issue: {cot_result['error']}")
            elif cot_result.get("new_data"):
                logger.info(f"  COT: {cot_result['new_records']} new records "
                           f"(latest: {cot_result['latest_date']})")
            else:
                logger.info(f"  COT: No new data (latest: {cot_result.get('latest_date', 'none')})")
        except Exception as e:
            logger.warning(f"  COT positioning unavailable: {e}")

        # === Data Quality Check + Auto-Backfill ===
        logger.info("Running data quality validation...")
        try:
            from finias.data.validation.fred_quality import detect_fred_gaps, validate_all_fred
            from finias.data.validation.quality import CONSECUTIVE_CRITICAL

            # Step 1: Detect gaps in critical series
            critical_gaps = await detect_fred_gaps(db, critical_only=True)

            if critical_gaps:
                logger.warning(f"Found {len(critical_gaps)} gap(s) in critical FRED series")

                # Step 2: Attempt auto-backfill from FRED
                for gap in critical_gaps:
                    try:
                        from datetime import date as _date
                        gap_start = _date.fromisoformat(gap["from_date"])
                        gap_end = _date.fromisoformat(gap["to_date"])
                        obs = await fred.get_series(
                            gap["series_id"],
                            observation_start=gap_start,
                            observation_end=gap_end,
                        )
                        if obs:
                            for o in obs:
                                await db.execute(
                                    "INSERT INTO economic_indicators (series_id, obs_date, value, source) "
                                    "VALUES ($1, $2, $3, 'fred_backfill') "
                                    "ON CONFLICT (series_id, obs_date) DO UPDATE SET value = $3",
                                    gap["series_id"],
                                    _date.fromisoformat(o["date"]),
                                    o["value"],
                                )
                            logger.info(f"  Auto-backfilled {gap['series_id']}: {len(obs)} observation(s)")
                        else:
                            logger.warning(
                                f"  {gap['series_id']}: FRED has no data for gap "
                                f"({gap['from_date']} → {gap['to_date']}). "
                                f"Affects: {gap['computation_affected']}"
                            )
                    except Exception as e:
                        logger.warning(f"  {gap['series_id']}: backfill failed: {e}")

                # Step 3: Re-check after backfill
                remaining_gaps = await detect_fred_gaps(db, critical_only=True)
                if remaining_gaps:
                    logger.warning(
                        f"  {len(remaining_gaps)} gap(s) remain after backfill "
                        f"(source data unavailable)"
                    )
                else:
                    logger.info("  All critical gaps resolved by auto-backfill")
            else:
                logger.info("  No gaps in critical FRED series")

        except Exception as e:
            logger.warning(f"  Data quality check failed (non-blocking): {e}")

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
