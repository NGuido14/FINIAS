"""
Macro Strategist data ingestion.

Handles fetching and refreshing all data needed by the Macro Strategist.
Can be run as a standalone refresh task or called by the agent itself.
"""

from __future__ import annotations
from datetime import date, timedelta
import logging

from finias.data.cache.market_cache import MarketDataCache
from finias.data.providers.fred_client import MACRO_SERIES

logger = logging.getLogger("finias.agent.macro_strategist.ingestion")


# Symbols needed by the Macro Strategist
REQUIRED_SYMBOLS = [
    # Index ETFs
    "SPY", "QQQ", "IWM", "DIA", "TLT", "HYG", "LQD", "GLD", "SLV", "USO",
    # Sector ETFs
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLC", "XLY", "XLRE", "XLB",
    # International & equal-weight
    "RSP", "EEM", "SHY"
]

REQUIRED_FRED_SERIES = list(MACRO_SERIES.keys())


async def refresh_macro_data(cache: MarketDataCache, lookback_days: int = 365) -> dict:
    """
    Refresh all data needed by the Macro Strategist.

    Returns dict with status of each data fetch.
    """
    from_date = date.today() - timedelta(days=lookback_days)
    to_date = date.today()
    status = {"symbols": {}, "fred": {}}

    # Market data
    for symbol in REQUIRED_SYMBOLS:
        try:
            bars = await cache.get_daily_bars(symbol, from_date, to_date, force_refresh=True)
            status["symbols"][symbol] = {"ok": True, "count": len(bars)}
            logger.info(f"Refreshed {symbol}: {len(bars)} bars")
        except Exception as e:
            status["symbols"][symbol] = {"ok": False, "error": str(e)}
            logger.error(f"Failed to refresh {symbol}: {e}")

    # FRED data
    for series_id in REQUIRED_FRED_SERIES:
        try:
            obs = await cache.get_fred_series(series_id, from_date=from_date, force_refresh=True)
            status["fred"][series_id] = {"ok": True, "count": len(obs)}
            logger.info(f"Refreshed {series_id}: {len(obs)} observations")
        except Exception as e:
            status["fred"][series_id] = {"ok": False, "error": str(e)}
            logger.error(f"Failed to refresh {series_id}: {e}")

    # Populate the macro data matrix
    try:
        matrix_count = await cache.populate_macro_matrix()
        status["matrix"] = {"ok": True, "dates_populated": matrix_count}
        logger.info(f"Macro matrix populated: {matrix_count} dates")
    except Exception as e:
        status["matrix"] = {"ok": False, "error": str(e)}
        logger.error(f"Failed to populate macro matrix: {e}")

    return status
