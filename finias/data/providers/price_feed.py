"""
Live price feed — shared infrastructure for all FINIAS agents.

Fetches current prices from yfinance for instruments where FRED's
1-7 day lag creates material gaps. Stores to Redis under its own
key (prices:live), separate from the macro regime.

Architecture:
  - prices:live is SHARED — any agent can read it
  - Morning refresh populates it, agents refresh on-demand if stale
  - The macro agent reads SKEW for computation (only yfinance source)
  - The macro agent reads all prices for interpretation data notes
  - Future agents (Risk Officer, Trade Decision) call get_current_prices()
    for execution-time pricing

If yfinance fails, every field returns None and the system continues
with FRED-only data. No computation breaks except SKEW (which has
no FRED alternative and gracefully defaults to unknown).
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("finias.data.price_feed")

# yfinance ticker mapping — 7 instruments
LIVE_INSTRUMENTS = {
    "vix": "^VIX",
    "skew": "^SKEW",
    "wti": "CL=F",
    "brent": "BZ=F",
    "gold": "GC=F",
    "dxy": "DX-Y.NYB",
    "spx": "^GSPC",
}

# Redis key and TTL
PRICES_REDIS_KEY = "prices:live"
PRICES_TTL = 50400  # 14 hours — same as regime TTL


async def fetch_live_prices() -> dict:
    """
    Fetch current prices for key market instruments from yfinance.

    Returns a dict with instrument prices and metadata.
    All price fields are Optional — if any fetch fails, that field is None.

    This runs in a thread executor because yfinance is synchronous.
    """
    import asyncio

    def _fetch_sync() -> dict:
        result = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "yfinance",
        }

        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed — live prices unavailable")
            for key in LIVE_INSTRUMENTS:
                result[key] = None
            result["error"] = "yfinance not installed"
            return result

        for key, ticker in LIVE_INSTRUMENTS.items():
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="5d")
                if hist is not None and not hist.empty:
                    result[key] = round(float(hist["Close"].iloc[-1]), 2)
                else:
                    result[key] = None
                    logger.warning(f"No data returned for {ticker}")
            except Exception as e:
                result[key] = None
                logger.warning(f"Failed to fetch {ticker}: {e}")

        return result

    try:
        loop = asyncio.get_event_loop()
        prices = await loop.run_in_executor(None, _fetch_sync)

        fetched = {k: v for k, v in prices.items()
                   if k not in ("fetched_at", "source", "error") and v is not None}
        failed = {k for k in LIVE_INSTRUMENTS if prices.get(k) is None}
        logger.info(f"Live prices fetched: {len(fetched)}/{len(LIVE_INSTRUMENTS)} instruments")
        if failed:
            logger.warning(f"Live prices missing: {', '.join(failed)}")

        return prices

    except Exception as e:
        logger.error(f"Live price feed failed entirely: {e}")
        result = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "yfinance",
            "error": str(e),
        }
        for key in LIVE_INSTRUMENTS:
            result[key] = None
        return result


async def store_live_prices(state, prices: dict) -> None:
    """
    Store live prices in Redis under prices:live.

    Args:
        state: RedisState instance
        prices: dict from fetch_live_prices()
    """
    await state.client.setex(
        PRICES_REDIS_KEY,
        PRICES_TTL,
        json.dumps(prices, default=str),
    )
    logger.info("Live prices stored in Redis (prices:live)")


async def get_live_prices(state) -> Optional[dict]:
    """
    Read the current live prices from Redis.

    Returns the prices dict if available, None if not stored yet.
    Any agent can call this to read shared price data.
    """
    data = await state.client.get(PRICES_REDIS_KEY)
    if data:
        return json.loads(data)
    return None


async def get_current_prices(state, max_age_seconds: int = 300) -> dict:
    """
    Get the freshest available prices for decision-making.

    Checks if cached prices:live in Redis are within max_age_seconds.
    If fresh enough, returns cached. If stale, fetches fresh from yfinance
    and updates the cache.

    This is what downstream agents should call at DECISION TIME.
    The morning refresh's cached prices may be hours old — this function
    ensures freshness for execution-critical decisions.

    Args:
        state: RedisState instance
        max_age_seconds: Maximum acceptable age in seconds (default 5 min)

    Returns:
        dict with live prices (may have None values if fetch failed)
    """
    cached = await get_live_prices(state)

    if cached:
        fetched_at = cached.get("fetched_at")
        if fetched_at:
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(fetched_at)).total_seconds()
                if age < max_age_seconds:
                    return cached
            except (ValueError, TypeError):
                pass

    # Cache is stale or missing — fetch fresh
    fresh = await fetch_live_prices()
    await store_live_prices(state, fresh)
    return fresh
