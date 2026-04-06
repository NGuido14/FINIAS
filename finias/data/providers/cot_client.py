"""
CFTC Commitment of Traders (COT) Data Client — Shared Infrastructure.

Downloads weekly COT positioning data from the official CFTC archives
via the cftc-cot library. Stores raw data in PostgreSQL for any agent
to query. The macro strategist computes percentiles and crowding signals;
future agents (Trade Decision, Risk Officer) may query raw data directly.

Data source: CFTC publishes every Friday at 3:30pm ET for the prior
Tuesday's positions. Data has an inherent 3-day lag.

Usage:
    from finias.data.providers.cot_client import fetch_and_store_cot_data, get_cot_history

    # During morning refresh:
    result = await fetch_and_store_cot_data(db)

    # For computation:
    history = await get_cot_history(db, "sp500", lookback_weeks=156)
"""

import asyncio
import logging
from datetime import date, datetime, timezone
from typing import Optional

logger = logging.getLogger("finias.data.cot")


# ============================================================================
# Contract Mapping — Verified against live CFTC data on 2026-04-05
# ============================================================================

COT_CONTRACTS = {
    "sp500": "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE",
    "treasury_10y": "UST 10Y NOTE - CHICAGO BOARD OF TRADE",
    "wti_crude": "CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE",
    "gold": "GOLD - COMMODITY EXCHANGE INC.",
    "dollar_index": "USD INDEX - ICE FUTURES U.S.",
}

# Years to fetch on first run (need 156+ weeks for 3-year percentile)
COT_START_YEAR = 2023
COT_REPORT_TYPE = "legacy_fut"


# ============================================================================
# Data Fetching
# ============================================================================

def _fetch_cot_sync(start_year: int, end_year: int) -> list[dict]:
    """
    Synchronous fetch of COT data from CFTC archives.

    Downloads official CSV ZIP files from cftc.gov, filters to our
    5 target contracts, and returns a list of row dicts.

    Runs in a thread executor because the download is blocking I/O.
    """
    try:
        import pandas as pd
        from cftc_cot import cot_download_year_range
    except ImportError:
        logger.error("cftc-cot library not installed. Run: pip install cftc-cot")
        return []

    try:
        df = cot_download_year_range(
            start_year=start_year,
            end_year=end_year,
            cot_report_type=COT_REPORT_TYPE,
            store_zip=False,
        )
    except Exception as e:
        logger.error(f"CFTC download failed: {e}")
        return []

    if df is None or len(df) == 0:
        logger.warning("CFTC returned empty DataFrame")
        return []

    # Parse date column
    df["parsed_date"] = pd.to_datetime(
        df["As of Date in Form YYYY-MM-DD"], errors="coerce"
    )

    rows = []
    for contract_key, contract_name in COT_CONTRACTS.items():
        contract_df = df[df["Market and Exchange Names"] == contract_name].copy()
        if len(contract_df) == 0:
            logger.warning(f"COT contract not found: {contract_name}")
            continue

        contract_df = contract_df.sort_values("parsed_date")

        for _, row in contract_df.iterrows():
            report_date = row["parsed_date"]
            if pd.isna(report_date):
                continue

            noncomm_long = int(row.get("Noncommercial Positions-Long (All)", 0) or 0)
            noncomm_short = int(row.get("Noncommercial Positions-Short (All)", 0) or 0)

            rows.append({
                "contract_key": contract_key,
                "contract_name": contract_name,
                "report_date": report_date.date(),
                "open_interest": int(row.get("Open Interest (All)", 0) or 0),
                "noncomm_long": noncomm_long,
                "noncomm_short": noncomm_short,
                "net_spec": noncomm_long - noncomm_short,
            })

    logger.info(f"COT: parsed {len(rows)} rows across {len(COT_CONTRACTS)} contracts "
                f"({start_year}-{end_year})")
    return rows


async def fetch_and_store_cot_data(db) -> dict:
    """
    Fetch COT data from CFTC and store new records in PostgreSQL.

    Checks what's already stored to determine fetch range:
    - If no data exists: fetches COT_START_YEAR to current year (full history)
    - If data exists: fetches current year only (incremental)

    Upserts rows (ON CONFLICT DO NOTHING) so re-fetches are idempotent.

    Args:
        db: DatabasePool instance

    Returns:
        dict with keys: new_data (bool), new_records (int), latest_date (str),
                        total_records (int), error (str or None)
    """
    result = {
        "new_data": False,
        "new_records": 0,
        "latest_date": None,
        "total_records": 0,
        "error": None,
    }

    try:
        # Check what we already have
        latest_row = await db.fetchrow(
            "SELECT MAX(report_date) as max_date, COUNT(*) as total "
            "FROM cot_positioning"
        )

        current_year = date.today().year

        if latest_row and latest_row["total"] > 0:
            result["total_records"] = latest_row["total"]
            result["latest_date"] = str(latest_row["max_date"])
            # Incremental: fetch current year only
            start_year = current_year
        else:
            # First run: fetch full history
            start_year = COT_START_YEAR

        # Fetch from CFTC (runs in thread executor)
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(
            None, _fetch_cot_sync, start_year, current_year
        )

        if not rows:
            if result["total_records"] == 0:
                result["error"] = "No COT data fetched and no cached data available"
            return result

        # Upsert into PostgreSQL
        inserted = 0
        for row in rows:
            try:
                res = await db.execute(
                    """
                    INSERT INTO cot_positioning
                        (contract_key, contract_name, report_date,
                         open_interest, noncomm_long, noncomm_short, net_spec)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (contract_key, report_date) DO NOTHING
                    """,
                    row["contract_key"],
                    row["contract_name"],
                    row["report_date"],
                    row["open_interest"],
                    row["noncomm_long"],
                    row["noncomm_short"],
                    row["net_spec"],
                )
                # asyncpg returns "INSERT 0 1" for inserted, "INSERT 0 0" for conflict
                if res and res.endswith("1"):
                    inserted += 1
            except Exception as e:
                logger.warning(f"COT insert failed for {row['contract_key']} "
                              f"{row['report_date']}: {e}")

        result["new_records"] = inserted
        result["new_data"] = inserted > 0

        # Update counts
        count_row = await db.fetchrow(
            "SELECT MAX(report_date) as max_date, COUNT(*) as total "
            "FROM cot_positioning"
        )
        if count_row:
            result["total_records"] = count_row["total"]
            result["latest_date"] = str(count_row["max_date"])

        logger.info(f"COT: {inserted} new records stored "
                    f"(total: {result['total_records']}, latest: {result['latest_date']})")

    except Exception as e:
        logger.error(f"COT fetch/store failed: {e}")
        result["error"] = str(e)

    return result


# ============================================================================
# Data Queries — Used by computation modules and future agents
# ============================================================================

async def get_cot_history(
    db, contract_key: str, lookback_weeks: int = 156
) -> list[dict]:
    """
    Get historical COT data for a single contract.

    Returns rows sorted by report_date ascending (oldest first),
    limited to the most recent `lookback_weeks` entries.

    Args:
        db: DatabasePool instance
        contract_key: One of: sp500, treasury_10y, wti_crude, gold, dollar_index
        lookback_weeks: Number of weeks of history (default 156 = 3 years)

    Returns:
        List of dicts with: report_date, open_interest, noncomm_long,
        noncomm_short, net_spec
    """
    rows = await db.fetch(
        """
        SELECT report_date, open_interest, noncomm_long, noncomm_short, net_spec
        FROM cot_positioning
        WHERE contract_key = $1
        ORDER BY report_date DESC
        LIMIT $2
        """,
        contract_key,
        lookback_weeks,
    )

    # Reverse to chronological order (oldest first)
    return [dict(row) for row in reversed(rows)]


async def get_latest_cot(db) -> dict:
    """
    Get the latest COT data for all 5 contracts.

    Returns a dict keyed by contract_key, each containing the most
    recent row. Used by diagnostics and quick status checks.

    Returns:
        {contract_key: {report_date, net_spec, open_interest, ...}}
    """
    result = {}
    for contract_key in COT_CONTRACTS:
        row = await db.fetchrow(
            """
            SELECT report_date, open_interest, noncomm_long, noncomm_short, net_spec
            FROM cot_positioning
            WHERE contract_key = $1
            ORDER BY report_date DESC
            LIMIT 1
            """,
            contract_key,
        )
        if row:
            result[contract_key] = dict(row)
    return result


async def get_cot_staleness_days(db) -> int:
    """
    Get the number of days since the latest COT report date.

    Used for staleness warnings in data notes. Normal staleness is
    3-10 days (CFTC publishes Friday for prior Tuesday). >14 days
    indicates a data gap (e.g., CFTC publication lapse).

    Returns:
        Number of days since latest report, or 999 if no data
    """
    row = await db.fetchrow(
        "SELECT MAX(report_date) as max_date FROM cot_positioning"
    )
    if row and row["max_date"]:
        return (date.today() - row["max_date"]).days
    return 999
