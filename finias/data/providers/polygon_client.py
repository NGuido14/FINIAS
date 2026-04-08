from __future__ import annotations
from typing import Any, Optional
from datetime import date, datetime, timedelta, timezone
import asyncio
import logging
import aiohttp

from finias.core.config.settings import get_settings

logger = logging.getLogger("finias.data.polygon")


class PolygonClient:
    """
    Async Polygon.io REST API client with rate limiting.

    Rate limits (free tier): 5 calls/minute
    Rate limits (paid tier): varies, typically 100+/minute

    This client implements simple token-bucket rate limiting.
    All methods return parsed JSON dicts — no Polygon SDK dependency.
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_calls_per_minute: Optional[int] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.polygon_api_key
        self.max_calls_per_minute = max_calls_per_minute or settings.polygon_rate_limit
        self._semaphore = asyncio.Semaphore(self.max_calls_per_minute)
        self._session: Optional[aiohttp.ClientSession] = None
        self._call_times: list[float] = []

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self._session

    async def _rate_limited_request(self, url: str, params: Optional[dict] = None) -> dict[str, Any]:
        """Make a rate-limited GET request."""
        import time

        # Simple rate limiting: ensure we don't exceed max calls per minute
        now = time.monotonic()
        self._call_times = [t for t in self._call_times if now - t < 60]

        if len(self._call_times) >= self.max_calls_per_minute:
            wait_time = 60 - (now - self._call_times[0])
            if wait_time > 0:
                logger.debug(f"Rate limit: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        session = await self._get_session()
        async with session.get(url, params=params) as resp:
            self._call_times.append(time.monotonic())

            if resp.status == 429:
                logger.warning("Rate limited by Polygon. Waiting 60s.")
                await asyncio.sleep(60)
                return await self._rate_limited_request(url, params)

            resp.raise_for_status()
            return await resp.json()

    async def get_daily_bars(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        adjusted: bool = True
    ) -> list[dict[str, Any]]:
        """
        Get daily OHLCV bars for a symbol.

        Returns list of dicts with keys:
            o (open), h (high), l (low), c (close), v (volume),
            vw (vwap), n (num_trades), t (timestamp_ms)
        """
        url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}"
        params = {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000}
        data = await self._rate_limited_request(url, params)
        return data.get("results", [])

    async def get_snapshot(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get current snapshot for a single ticker (last price, today's change, etc.)."""
        url = f"{self.BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        try:
            data = await self._rate_limited_request(url)
            return data.get("ticker")
        except Exception as e:
            logger.error(f"Snapshot failed for {symbol}: {e}")
            return None

    async def get_market_indices(
        self,
        symbols: list[str],
        from_date: date,
        to_date: date
    ) -> dict[str, list[dict[str, Any]]]:
        """Get daily bars for multiple index/ETF symbols."""
        results = {}
        for symbol in symbols:
            bars = await self.get_daily_bars(symbol, from_date, to_date)
            results[symbol] = bars
        return results

    async def get_grouped_daily(self, trade_date: date) -> list[dict[str, Any]]:
        """
        Get daily bars for ALL tickers on a specific date.
        Useful for breadth calculations.
        """
        url = f"{self.BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{trade_date}"
        params = {"adjusted": "true"}
        data = await self._rate_limited_request(url, params)
        return data.get("results", [])

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
