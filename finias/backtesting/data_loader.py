"""
Historical data loader for backtesting.

Fetches extended historical data (5yr Polygon, 10yr FRED) and provides
date-filtered views with publication lag enforcement.

Publication lag prevents look-ahead bias — the most critical backtesting sin.
On simulation date T, monthly CPI with obs_date January 1 is NOT available
until ~February 14 (45-day lag). The backtest must respect this.
"""

from __future__ import annotations
from datetime import date, timedelta
from typing import Optional
import logging

from finias.core.database.connection import DatabasePool
from finias.data.providers.fred_client import FredClient, MACRO_SERIES
from finias.data.providers.polygon_client import PolygonClient

logger = logging.getLogger("finias.backtesting.data_loader")


# Publication lag by series type (days after obs_date until data is available)
# Conservative estimates — actual release dates vary but these prevent look-ahead
PUBLICATION_LAG = {
    # Daily market data — available next business day
    "daily_market": 1,     # Polygon bars
    # Daily FRED (yields, VIX, dollar) — available same day or next day
    "daily_fred": 0,
    # Daily FRED (reverse repo) — available next day
    "daily_fred_lagged": 1,
    # Weekly FRED (Fed balance sheet, claims, NFCI) — ~7 day lag
    "weekly": 7,
    # Monthly FRED (CPI, PCE, unemployment, etc.) — ~30-45 day lag
    "monthly": 45,
    # Quarterly / infrequent — ~90 day lag
    "quarterly": 90,
}

# Map each FRED series to its publication lag category
SERIES_LAG_CATEGORY = {
    # Daily (0-day lag)
    "DGS2": "daily_fred", "DGS5": "daily_fred", "DGS10": "daily_fred",
    "DGS30": "daily_fred", "DTB3": "daily_fred", "VIXCLS": "daily_fred",
    "VXVCLS": "daily_fred", "T10Y2Y": "daily_fred", "T10Y3M": "daily_fred",
    "BAMLH0A0HYM2": "daily_fred", "DFEDTARU": "daily_fred",
    "DFEDTARL": "daily_fred", "DTWEXBGS": "daily_fred",
    "T5YIE": "daily_fred", "T10YIE": "daily_fred",
    "DFII5": "daily_fred", "DFII10": "daily_fred",
    "THREEFYTP10": "daily_fred", "DCOILWTICO": "daily_fred",
    "DCOILBRENTEU": "daily_fred", "T5YIFR": "daily_fred",

    # Daily with 1-day lag
    "RRPONTSYD": "daily_fred_lagged",

    # Weekly (7-day lag)
    "WALCL": "weekly", "TREAST": "weekly", "WSHOMCB": "weekly",
    "WTREGEN": "weekly", "WRESBAL": "weekly", "NFCI": "weekly",
    "ANFCI": "weekly", "STLFSI4": "weekly", "ICSA": "weekly",
    "CCSA": "weekly",

    # Monthly (45-day lag)
    "FEDFUNDS": "monthly", "CPIAUCSL": "monthly", "CPILFESL": "monthly",
    "CUSR0000SEHC": "monthly", "CUSR0000SAS": "monthly",
    "PCEPI": "monthly", "PCEPILFE": "monthly",
    "STICKCPIM159SFRBATL": "monthly", "FLEXCPIM159SFRBATL": "monthly",
    "PCETRIM12M159SFRBDAL": "monthly", "PPIACO": "monthly",
    "CES0500000003": "monthly", "UNRATE": "monthly", "PAYEMS": "monthly",
    "U6RATE": "monthly", "CIVPART": "monthly", "LNS11300060": "monthly",
    "TEMPHELPS": "monthly", "AWHAETP": "monthly",
    "UMCSENT": "monthly", "INDPRO": "monthly",
    "TOTALSL": "monthly", "M2SL": "monthly",
    "TCU": "monthly", "PERMIT": "monthly", "HOUST": "monthly",
    "RSAFS": "monthly", "PI": "monthly", "PCEC96": "monthly",
    "DGORDER": "monthly", "CFNAI": "monthly",
    "GACDFSA066MSFRBPHI": "monthly",
    "JTSJOL": "monthly", "JTSQUR": "monthly",
    "TOTBKCR": "monthly",

    # Quarterly (90-day lag)
    "GDPNOW": "quarterly",
}


async def ensure_historical_data(
    db: DatabasePool,
    fred: FredClient,
    polygon: PolygonClient,
    fred_years: int = 10,
    polygon_years: int = 5,
) -> dict:
    """
    Ensure we have sufficient historical data for backtesting.

    Fetches extended history if not already in the database.
    Returns status dict with counts.
    """
    status = {"fred": {}, "polygon": {}}

    fred_start = date.today() - timedelta(days=fred_years * 365)
    polygon_start = date.today() - timedelta(days=polygon_years * 365)

    # --- FRED: Extend to 10 years ---
    for series_id in MACRO_SERIES:
        # Check what we already have
        earliest = await db.fetchval(
            "SELECT MIN(obs_date) FROM economic_indicators WHERE series_id = $1",
            series_id
        )

        if earliest and earliest <= fred_start:
            # Already have enough history
            count = await db.fetchval(
                "SELECT COUNT(*) FROM economic_indicators WHERE series_id = $1",
                series_id
            )
            status["fred"][series_id] = {"ok": True, "count": count, "action": "cached"}
            continue

        # Need to fetch more history
        try:
            logger.info(f"Fetching extended history for {series_id} from {fred_start}")
            observations = await fred.get_series(
                series_id, observation_start=fred_start
            )
            if observations:
                async with db.acquire() as conn:
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
                            MACRO_SERIES.get(series_id, series_id),
                        )
                status["fred"][series_id] = {"ok": True, "count": len(observations), "action": "fetched"}
            else:
                status["fred"][series_id] = {"ok": False, "count": 0, "action": "empty"}
        except Exception as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            status["fred"][series_id] = {"ok": False, "error": str(e)}

        # Rate limit
        import asyncio
        await asyncio.sleep(0.5)

    # --- Polygon: Extend to 5 years ---
    from finias.agents.macro_strategist.data.ingestion import REQUIRED_SYMBOLS

    # Polygon symbols needed by the macro strategist
    # Filter to just the ones we use in computations (not QQQ, DIA, SHY, LQD, SLV, USO)
    BACKTEST_SYMBOLS = [
        "SPY", "IWM", "TLT", "GLD", "HYG", "CPER", "EEM", "RSP",
        "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    ]

    for symbol in BACKTEST_SYMBOLS:
        # Check what we already have
        earliest = await db.fetchval(
            "SELECT MIN(trade_date) FROM market_data_daily WHERE symbol = $1",
            symbol
        )

        if earliest and earliest <= polygon_start:
            count = await db.fetchval(
                "SELECT COUNT(*) FROM market_data_daily WHERE symbol = $1",
                symbol
            )
            status["polygon"][symbol] = {"ok": True, "count": count, "action": "cached"}
            continue

        # Fetch extended history
        try:
            logger.info(f"Fetching 5yr history for {symbol}")
            bars = await polygon.get_daily_bars(symbol, polygon_start, date.today())
            if bars:
                async with db.acquire() as conn:
                    for bar in bars:
                        t_date = bar.get("t") or bar.get("date")
                        if isinstance(t_date, str):
                            t_date = date.fromisoformat(t_date[:10])
                        elif isinstance(t_date, (int, float)):
                            from datetime import datetime
                            t_date = datetime.fromtimestamp(t_date / 1000).date()

                        await conn.execute(
                            """
                            INSERT INTO market_data_daily
                                (symbol, trade_date, open, high, low, close, volume)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (symbol, trade_date) DO UPDATE SET
                                open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                close = EXCLUDED.close,
                                volume = EXCLUDED.volume
                            """,
                            symbol, t_date,
                            float(bar.get("o", 0)), float(bar.get("h", 0)),
                            float(bar.get("l", 0)), float(bar.get("c", 0)),
                            int(bar.get("v", 0)),
                        )
                status["polygon"][symbol] = {"ok": True, "count": len(bars), "action": "fetched"}
            else:
                status["polygon"][symbol] = {"ok": False, "count": 0, "action": "empty"}
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            status["polygon"][symbol] = {"ok": False, "error": str(e)}

        import asyncio
        await asyncio.sleep(1.0)  # More conservative for Polygon

    return status


def filter_series_as_of(
    series: list[dict],
    sim_date: date,
    lag_category: str,
) -> list[dict]:
    """
    Filter a series to only include data available as of sim_date.

    Applies publication lag: on sim_date T, data with obs_date D
    is only available if D <= T - publication_lag_days.
    """
    lag_days = PUBLICATION_LAG.get(lag_category, 45)  # Default to monthly lag
    cutoff = sim_date - timedelta(days=lag_days)

    return [
        obs for obs in series
        if date.fromisoformat(obs["date"]) <= cutoff
    ]


def filter_bars_as_of(
    bars: list[dict],
    sim_date: date,
) -> list[dict]:
    """Filter Polygon bars to only include data available as of sim_date."""
    cutoff = sim_date - timedelta(days=PUBLICATION_LAG["daily_market"])

    return [
        bar for bar in bars
        if date.fromisoformat(bar["date"]) <= cutoff
    ]
