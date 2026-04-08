"""
FINIAS Universe Data Seeder

Fetches S&P 500 constituents from Wikipedia, populates the symbol_universe
table, and fetches historical price data from Polygon.

Usage:
    python -m finias.scripts.seed_universe              # Full seed (fetch list + prices)
    python -m finias.scripts.seed_universe --universe    # Universe table only (no Polygon)
    python -m finias.scripts.seed_universe --prices      # Prices only (assumes universe populated)
    python -m finias.scripts.seed_universe --check       # Check status only

Idempotent — safe to re-run. Uses ON CONFLICT for upserts and only
fetches missing date ranges from Polygon.

The S&P 500 list comes from Wikipedia's "List of S&P 500 companies" page,
which includes tickers, company names, GICS sectors, and sub-industries.
"""

from __future__ import annotations
import asyncio
import logging
import sys
from datetime import date, timedelta, datetime, timezone

from finias.core.config.settings import get_settings
from finias.core.database.connection import DatabasePool
from finias.core.database.migrations import run_migrations
from finias.data.providers.polygon_client import PolygonClient
from finias.data.universe import (
    MACRO_ETFS,
    fetch_sp500_from_wikipedia,
    populate_macro_etfs,
    populate_sp500_from_list,
    get_active_symbols,
    get_universe_summary,
)

logger = logging.getLogger("finias.scripts.seed_universe")

HISTORY_YEARS = 5


async def seed_universe_table(db: DatabasePool) -> dict:
    """Fetch S&P 500 from Wikipedia and populate symbol_universe."""
    print("  → Fetching S&P 500 constituents from Wikipedia...")
    try:
        constituents = fetch_sp500_from_wikipedia()
    except Exception as e:
        print(f"    ✗ Failed to fetch from Wikipedia: {e}")
        print("    Check your internet connection and that pandas/lxml are installed.")
        return {"error": str(e)}

    print(f"    Fetched {len(constituents)} constituents")

    # Show sector breakdown
    sectors = {}
    for c in constituents:
        s = c.get("sector", "Unknown")
        sectors[s] = sectors.get(s, 0) + 1
    for sector, count in sorted(sectors.items()):
        print(f"      {sector}: {count}")

    print("  → Populating macro ETFs...")
    macro_count = await populate_macro_etfs(db)
    print(f"    {macro_count} macro ETFs")

    print("  → Populating S&P 500 constituents...")
    result = await populate_sp500_from_list(db, constituents)
    print(f"    {result['sp500_count']} S&P 500 constituents")

    return result


async def seed_price_data(
    db: DatabasePool,
    polygon: PolygonClient,
    symbols: list[str] = None,
    force: bool = False,
) -> dict:
    """
    Fetch historical price data from Polygon for all active symbols.
    Only fetches missing data. Safe to re-run.
    """
    if symbols is None:
        symbols = await get_active_symbols(db)

    if not symbols:
        print("  ⚠ No active symbols in universe table. Run with --universe first.")
        return {"total": 0, "seeded": 0, "skipped": 0, "failed": 0}

    from_date = date.today() - timedelta(days=365 * HISTORY_YEARS)
    to_date = date.today()

    total = len(symbols)
    seeded = 0
    skipped = 0
    failed = 0
    failed_symbols = []

    print(f"  → Seeding price data for {total} symbols ({HISTORY_YEARS}yr history)...")
    print(f"    Polygon rate limit: {polygon.max_calls_per_minute} calls/min")
    print(f"    Date range: {from_date} → {to_date}")
    print()

    for i, symbol in enumerate(symbols, 1):
        try:
            existing = await db.fetchrow(
                """
                SELECT MIN(trade_date) as earliest, MAX(trade_date) as latest, COUNT(*) as cnt
                FROM market_data_daily WHERE symbol = $1
                """,
                symbol,
            )

            if existing and existing["cnt"] and existing["cnt"] > 0 and not force:
                earliest = existing["earliest"]
                latest = existing["latest"]
                days_stale = (date.today() - latest).days

                if earliest <= from_date + timedelta(days=30) and days_stale <= 5:
                    skipped += 1
                    if i % 50 == 0 or i == total:
                        print(f"    [{i}/{total}] {seeded} seeded, {skipped} skipped, {failed} failed")
                    continue

                fetch_from = latest + timedelta(days=1) if latest > from_date else from_date
            else:
                fetch_from = from_date

            bars = await polygon.get_daily_bars(symbol, fetch_from, to_date)

            if bars:
                async with db.acquire() as conn:
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
                            bar.get("v"), bar.get("vw"), bar.get("n"),
                        )
                seeded += 1
            else:
                logger.warning(f"No bars returned for {symbol}")
                failed += 1
                failed_symbols.append(symbol)

        except Exception as e:
            logger.error(f"Failed to seed {symbol}: {e}")
            failed += 1
            failed_symbols.append(symbol)

        if i % 25 == 0 or i == total:
            print(f"    [{i}/{total}] {seeded} seeded, {skipped} skipped, {failed} failed")

    return {
        "total": total,
        "seeded": seeded,
        "skipped": skipped,
        "failed": failed,
        "failed_symbols": failed_symbols,
    }


async def check_status(db: DatabasePool):
    """Report current universe and data status."""
    print("\n  Universe Status:")

    summary = await get_universe_summary(db)

    if summary["tiers"]:
        for tier, info in summary["tiers"].items():
            print(f"    {tier}: {info['active']}/{info['total']} active")
    else:
        print("    symbol_universe table is empty")

    if summary["sectors"]:
        print("\n  S&P 500 by GICS Sector:")
        for sector, count in sorted(summary["sectors"].items()):
            print(f"    {sector}: {count}")

    print("\n  Price Data Status:")
    row = await db.fetchrow(
        """
        SELECT COUNT(DISTINCT symbol) as symbol_count, COUNT(*) as total_bars,
               MIN(trade_date) as earliest, MAX(trade_date) as latest
        FROM market_data_daily
        """
    )
    if row:
        print(f"    Symbols with data: {row['symbol_count']}")
        print(f"    Total bars: {row['total_bars']:,}")
        if row["earliest"]:
            print(f"    Date range: {row['earliest']} → {row['latest']}")

    stale_rows = await db.fetch(
        """
        WITH latest AS (
            SELECT symbol, MAX(trade_date) as latest_date
            FROM market_data_daily GROUP BY symbol
        )
        SELECT
            CASE
                WHEN (CURRENT_DATE - latest_date) <= 2 THEN 'fresh'
                WHEN (CURRENT_DATE - latest_date) <= 5 THEN 'slight_stale'
                WHEN (CURRENT_DATE - latest_date) <= 10 THEN 'stale'
                ELSE 'very_stale'
            END as freshness,
            COUNT(*) as cnt
        FROM latest GROUP BY freshness ORDER BY freshness
        """
    )
    if stale_rows:
        print("\n  Freshness Breakdown:")
        for row in stale_rows:
            label = {
                "fresh": "✓ Fresh (≤2d)",
                "slight_stale": "~ Slightly stale (3-5d)",
                "stale": "⚠ Stale (6-10d)",
                "very_stale": "✗ Very stale (>10d)",
            }.get(row["freshness"], row["freshness"])
            print(f"    {label}: {row['cnt']} symbols")


async def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    args = set(sys.argv[1:])
    do_check = "--check" in args
    do_universe = "--universe" in args or (not args - {"--check"})
    do_prices = "--prices" in args or (not args - {"--check"})

    print("╔══════════════════════════════════════╗")
    print("║   FINIAS Universe Data Seeder        ║")
    print("╚══════════════════════════════════════╝\n")

    db = DatabasePool()
    await db.initialize()
    await run_migrations(db)

    if do_check and len(args) == 1:
        await check_status(db)
        await db.close()
        return

    polygon = PolygonClient()

    try:
        if do_universe:
            result = await seed_universe_table(db)
            if "error" in result:
                print("\n  Universe seeding failed. Fix the error above and retry.")
                return

        if do_prices:
            result = await seed_price_data(db, polygon)
            print(f"\n  Summary: {result['seeded']} seeded, {result['skipped']} cached, "
                  f"{result['failed']} failed out of {result['total']} symbols")
            if result["failed_symbols"]:
                print(f"  Failed: {', '.join(result['failed_symbols'][:20])}")
                if len(result["failed_symbols"]) > 20:
                    print(f"  ... and {len(result['failed_symbols']) - 20} more")

        await check_status(db)

    finally:
        await polygon.close()
        await db.close()

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
