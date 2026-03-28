from __future__ import annotations
from typing import Any, Optional
from datetime import date, datetime, timedelta, timezone
import logging

from finias.core.database.connection import DatabasePool
from finias.core.state.redis_state import RedisState
from finias.data.providers.polygon_client import PolygonClient
from finias.data.providers.fred_client import FredClient

logger = logging.getLogger("finias.data.cache")


class MarketDataCache:
    """
    Intelligent caching layer for market data.

    Flow: Agent requests data → Cache checks PostgreSQL → If missing,
    fetches from API → Stores in PostgreSQL → Returns to agent.

    Redis is used to track data freshness so we know when to refresh.
    """

    def __init__(
        self,
        db: DatabasePool,
        state: RedisState,
        polygon: PolygonClient,
        fred: FredClient,
    ):
        self.db = db
        self.state = state
        self.polygon = polygon
        self.fred = fred

    async def get_daily_bars(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        force_refresh: bool = False
    ) -> list[dict[str, Any]]:
        """
        Get daily OHLCV data, using cache when possible.

        1. Check PostgreSQL for existing data
        2. Identify missing date ranges
        3. Fetch missing data from Polygon
        4. Store in PostgreSQL
        5. Return complete dataset
        """
        if not force_refresh:
            # Try database first
            rows = await self.db.fetch(
                """
                SELECT trade_date, open, high, low, close, volume, vwap
                FROM market_data_daily
                WHERE symbol = $1 AND trade_date BETWEEN $2 AND $3
                ORDER BY trade_date ASC
                """,
                symbol, from_date, to_date
            )

            if rows:
                return [dict(r) for r in rows]

        # Fetch from Polygon
        logger.info(f"Fetching {symbol} bars from Polygon: {from_date} to {to_date}")
        bars = await self.polygon.get_daily_bars(symbol, from_date, to_date)

        # Store in database
        if bars:
            async with self.db.acquire() as conn:
                for bar in bars:
                    bar_date = datetime.fromtimestamp(
                        bar["t"] / 1000, tz=timezone.utc
                    ).date()
                    await conn.execute(
                        """
                        INSERT INTO market_data_daily
                            (symbol, trade_date, open, high, low, close, volume, vwap, num_trades)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (symbol, trade_date) DO UPDATE SET
                            open = EXCLUDED.open, high = EXCLUDED.high,
                            low = EXCLUDED.low, close = EXCLUDED.close,
                            volume = EXCLUDED.volume, vwap = EXCLUDED.vwap,
                            num_trades = EXCLUDED.num_trades
                        """,
                        symbol, bar_date,
                        bar.get("o"), bar.get("h"), bar.get("l"), bar.get("c"),
                        bar.get("v"), bar.get("vw"), bar.get("n")
                    )

            await self.state.mark_data_fresh(f"polygon:{symbol}")

        # Return from database (now populated)
        rows = await self.db.fetch(
            """
            SELECT trade_date, open, high, low, close, volume, vwap
            FROM market_data_daily
            WHERE symbol = $1 AND trade_date BETWEEN $2 AND $3
            ORDER BY trade_date ASC
            """,
            symbol, from_date, to_date
        )
        return [dict(r) for r in rows]

    async def get_fred_series(
        self,
        series_id: str,
        from_date: Optional[date] = None,
        force_refresh: bool = False
    ) -> list[dict[str, Any]]:
        """
        Get FRED economic data, using cache when possible.
        """
        if from_date is None:
            from_date = date.today() - timedelta(days=365)

        if not force_refresh:
            rows = await self.db.fetch(
                """
                SELECT obs_date, value
                FROM economic_indicators
                WHERE series_id = $1 AND obs_date >= $2
                ORDER BY obs_date ASC
                """,
                series_id, from_date
            )
            if rows:
                return [{"date": str(r["obs_date"]), "value": float(r["value"])} for r in rows]

        # Fetch from FRED
        logger.info(f"Fetching {series_id} from FRED")
        from finias.data.providers.fred_client import MACRO_SERIES
        series_name = MACRO_SERIES.get(series_id, series_id)

        observations = await self.fred.get_series(series_id, observation_start=from_date)

        # Store in database
        if observations:
            async with self.db.acquire() as conn:
                for obs in observations:
                    await conn.execute(
                        """
                        INSERT INTO economic_indicators
                            (series_id, obs_date, value, series_name, source)
                        VALUES ($1, $2, $3, $4, 'fred')
                        ON CONFLICT (series_id, obs_date) DO UPDATE SET
                            value = EXCLUDED.value
                        """,
                        series_id,
                        date.fromisoformat(obs["date"]),
                        obs["value"],
                        series_name
                    )

            await self.state.mark_data_fresh(f"fred:{series_id}")

        # Return from database
        rows = await self.db.fetch(
            """
            SELECT obs_date, value
            FROM economic_indicators
            WHERE series_id = $1 AND obs_date >= $2
            ORDER BY obs_date ASC
            """,
            series_id, from_date
        )
        return [{"date": str(r["obs_date"]), "value": float(r["value"])} for r in rows]
