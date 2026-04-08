from __future__ import annotations
from typing import Any, Optional
from datetime import date, datetime, timedelta, timezone
import logging

from finias.core.database.connection import DatabasePool
from finias.core.state.redis_state import RedisState
from finias.data.providers.polygon_client import PolygonClient
from finias.data.providers.fred_client import FredClient
from finias.data.cache.matrix_mapping import SERIES_TO_COLUMN

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
                # Staleness check: if latest bar is > 2 calendar days old, refetch
                # (2 calendar days accounts for weekends — Friday data checked on Sunday is OK)
                latest_bar_date = rows[-1]["trade_date"]
                days_stale = (date.today() - latest_bar_date).days
                if days_stale <= 1:
                    return [dict(r) for r in rows]
                else:
                    logger.info(
                        f"Polygon cache stale for {symbol}: latest bar {latest_bar_date} "
                        f"is {days_stale} days old. Refreshing from Polygon."
                    )
                    # Only fetch the missing days, not the entire history
                    fetch_from = latest_bar_date + timedelta(days=1)
                    logger.info(f"Fetching {symbol} bars from Polygon: {fetch_from} to {to_date}")
                    bars = await self.polygon.get_daily_bars(symbol, fetch_from, to_date)
                    if bars:
                        async with self.db.acquire() as conn:
                            for bar in bars:
                                bar_date = datetime.fromtimestamp(bar["t"] / 1000, tz=timezone.utc).date()
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
                    # Return full history from database (original from_date preserved)
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

        # Fetch from Polygon (only reached on force_refresh or no cached data at all)
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

        Staleness logic: if the most recent cached observation is older than
        the staleness threshold, refresh from FRED. Daily series refresh after
        2 days stale, weekly/monthly series after 8 days.
        """
        if from_date is None:
            from_date = date.today() - timedelta(days=365)

        # Staleness thresholds by update frequency
        # Daily series: most market indicators, yields, VIX
        daily_series = {
            "DGS2", "DGS5", "DGS10", "DGS30", "DTB3",
            "T10Y2Y", "T10Y3M", "VIXCLS", "DTWEXBGS",
            "DFEDTARU", "DFEDTARL", "RRPONTSYD",
            "T5YIE", "T10YIE", "T5YIFR",
            "DFII5", "DFII10", "THREEFYTP10",
            "DCOILWTICO", "BAMLH0A0HYM2",
        }
        # Weekly series: Fed balance sheet, claims, financial conditions
        weekly_series = {
            "WALCL", "TREAST", "WSHOMCB", "WTREGEN", "WRESBAL",
            "ICSA", "CCSA", "NFCI", "ANFCI", "TOTBKCR",
        }
        # Everything else is monthly or quarterly

        if series_id in daily_series:
            stale_days = 2
        elif series_id in weekly_series:
            stale_days = 8
        else:
            stale_days = 40  # Monthly data: ~30 days between releases + buffer

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
                # Check staleness: is the most recent observation too old?
                latest_date = rows[-1]["obs_date"]
                days_old = (date.today() - latest_date).days
                if days_old <= stale_days:
                    return [{"date": str(r["obs_date"]), "value": float(r["value"])} for r in rows]
                else:
                    logger.info(
                        f"Cache stale for {series_id}: latest obs is {days_old} days old "
                        f"(threshold: {stale_days}). Refreshing."
                    )

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

        # Return from database (now refreshed)
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

    async def populate_macro_matrix(self) -> int:
        """
        Populate the macro_data_matrix table from economic_indicators.

        Pivots row-per-observation data into one-row-per-date matrix format.
        Also computes net_liquidity (fed_total_assets - tga_balance - reverse_repo).

        Returns number of rows upserted.
        """
        # Get all dates that have any data
        date_rows = await self.db.fetch(
            "SELECT DISTINCT obs_date FROM economic_indicators ORDER BY obs_date ASC"
        )

        if not date_rows:
            return 0

        count = 0
        async with self.db.acquire() as conn:
            for date_row in date_rows:
                obs_date = date_row["obs_date"]

                # Get all series values for this date
                series_rows = await conn.fetch(
                    "SELECT series_id, value FROM economic_indicators WHERE obs_date = $1",
                    obs_date
                )

                if not series_rows:
                    continue

                # Build column-value pairs
                columns = ["obs_date"]
                values = [obs_date]
                placeholders = ["$1"]
                update_parts = []
                idx = 2

                for row in series_rows:
                    col_name = SERIES_TO_COLUMN.get(row["series_id"])
                    if col_name:
                        columns.append(col_name)
                        values.append(float(row["value"]))
                        placeholders.append(f"${idx}")
                        update_parts.append(f"{col_name} = EXCLUDED.{col_name}")
                        idx += 1

                if len(columns) <= 1:
                    continue

                # Upsert
                sql = f"""
                    INSERT INTO macro_data_matrix ({', '.join(columns)})
                    VALUES ({', '.join(placeholders)})
                    ON CONFLICT (obs_date) DO UPDATE SET
                        {', '.join(update_parts)},
                        updated_at = NOW()
                """
                await conn.execute(sql, *values)
                count += 1

            # Compute net_liquidity for all rows that have the components
            await conn.execute("""
                UPDATE macro_data_matrix
                SET net_liquidity = fed_total_assets - COALESCE(tga_balance, 0) - COALESCE(reverse_repo, 0) * 1000
                WHERE fed_total_assets IS NOT NULL
                  AND (tga_balance IS NOT NULL OR reverse_repo IS NOT NULL)
            """)

        logger.info(f"Populated macro_data_matrix: {count} dates")
        return count

    async def get_macro_matrix(
        self,
        from_date: date,
        to_date: date,
        columns: list[str] | None = None,
    ) -> list[dict]:
        """
        Get macro data from the matrix table.

        Returns list of dicts, one per date, with requested columns.
        All dates are aligned — no cross-series date mismatches possible.

        Args:
            from_date: Start date
            to_date: End date
            columns: Specific columns to fetch. None = all columns.

        Returns:
            List of dicts with obs_date + requested column values.
        """
        if columns:
            # Validate column names to prevent SQL injection
            valid_columns = set(SERIES_TO_COLUMN.values()) | {"net_liquidity", "obs_date"}
            safe_columns = [c for c in columns if c in valid_columns]
            col_str = "obs_date, " + ", ".join(safe_columns)
        else:
            col_str = "*"

        rows = await self.db.fetch(
            f"""
            SELECT {col_str}
            FROM macro_data_matrix
            WHERE obs_date BETWEEN $1 AND $2
            ORDER BY obs_date ASC
            """,
            from_date, to_date
        )

        return [dict(r) for r in rows]

    async def get_batch_daily_bars(
        self,
        symbols: list[str],
        from_date: date,
        to_date: date = None,
    ) -> dict[str, list[dict]]:
        """
        Batch-load OHLCV bars for multiple symbols in a single DB query.

        This is the primary data access method for agents processing many symbols
        (e.g., Technical Analyst scanning 500 symbols). Unlike get_daily_bars(),
        this does NOT trigger Polygon fetches for missing/stale data. It reads
        only what is in PostgreSQL. Data freshness is managed separately by
        the seeding/refresh scripts.

        Args:
            symbols: List of ticker symbols to load.
            from_date: Start date for data window.
            to_date: End date (defaults to today).

        Returns:
            Dict mapping symbol -> list of bar dicts, each with:
            {trade_date, open, high, low, close, volume, vwap}
            Bars are sorted chronologically (oldest first) per symbol.
            Symbols with no data in the range are omitted from the dict.
        """
        to_date = to_date or date.today()

        rows = await self.db.fetch(
            """
            SELECT symbol, trade_date, open, high, low, close, volume, vwap
            FROM market_data_daily
            WHERE symbol = ANY($1) AND trade_date BETWEEN $2 AND $3
            ORDER BY symbol, trade_date ASC
            """,
            symbols, from_date, to_date
        )

        # Partition by symbol
        result: dict[str, list[dict]] = {}
        for row in rows:
            sym = row["symbol"]
            if sym not in result:
                result[sym] = []
            result[sym].append(dict(row))

        loaded = len(result)
        total_bars = len(rows)
        missing = len(symbols) - loaded
        if missing > 0:
            logger.warning(
                f"Batch load: {loaded}/{len(symbols)} symbols loaded "
                f"({total_bars} total bars), {missing} symbols had no data"
            )
        else:
            logger.info(
                f"Batch load: {loaded} symbols, {total_bars} total bars"
            )

        return result

    async def get_universe_staleness(
        self,
        symbols: list[str],
    ) -> dict[str, dict]:
        """
        Check data freshness for a list of symbols.

        Returns dict mapping symbol -> {latest_date, days_stale, bar_count}.
        Symbols with no data are included with latest_date=None.
        """
        rows = await self.db.fetch(
            """
            SELECT symbol, MAX(trade_date) as latest, COUNT(*) as bar_count
            FROM market_data_daily
            WHERE symbol = ANY($1)
            GROUP BY symbol
            """,
            symbols,
        )

        staleness = {}
        found = set()
        for row in rows:
            sym = row["symbol"]
            found.add(sym)
            latest = row["latest"]
            staleness[sym] = {
                "latest_date": latest,
                "days_stale": (date.today() - latest).days if latest else None,
                "bar_count": row["bar_count"],
            }

        for sym in symbols:
            if sym not in found:
                staleness[sym] = {
                    "latest_date": None,
                    "days_stale": None,
                    "bar_count": 0,
                }

        return staleness

    def extract_series_from_matrix(
        self,
        matrix_rows: list[dict],
        column_name: str,
    ) -> list[dict]:
        """
        Extract a single series from matrix rows in the format computation
        modules expect: [{"date": str, "value": float}, ...].

        Filters out NULL values (dates where this series has no observation).
        """
        series = []
        for row in matrix_rows:
            val = row.get(column_name)
            if val is not None:
                series.append({
                    "date": str(row["obs_date"]),
                    "value": float(val),
                })
        return series
