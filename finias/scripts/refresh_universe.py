"""
FINIAS Universe Data Refresher

Daily script that updates price data for all active symbols.
Designed to run via cron before the morning macro refresh.

Usage:
    python -m finias.scripts.refresh_universe

Cron example (run at 6:00 AM ET before macro refresh at 6:30):
    0 6 * * 1-5 cd /path/to/finias && python -m finias.scripts.refresh_universe
"""

from __future__ import annotations
import asyncio
import logging
from datetime import date, timedelta, datetime, timezone

from finias.core.database.connection import DatabasePool
from finias.core.database.migrations import run_migrations
from finias.data.providers.polygon_client import PolygonClient
from finias.data.universe import get_active_symbols

logger = logging.getLogger("finias.scripts.refresh_universe")

STALENESS_THRESHOLD_DAYS = 2


async def refresh_prices(db: DatabasePool, polygon: PolygonClient) -> dict:
    """Refresh stale price data for all active symbols."""
    symbols = await get_active_symbols(db)
    if not symbols:
        print("  ⚠ No active symbols in universe table. Run seed_universe first.")
        return {"total": 0, "refreshed": 0, "fresh": 0, "failed": 0}

    print(f"  → Checking {len(symbols)} active symbols for staleness...")

    rows = await db.fetch(
        """
        SELECT symbol, MAX(trade_date) as latest
        FROM market_data_daily WHERE symbol = ANY($1)
        GROUP BY symbol
        """,
        symbols,
    )
    latest_by_symbol = {r["symbol"]: r["latest"] for r in rows}

    to_date = date.today()
    stale = []
    fresh = 0
    no_data = []

    for sym in symbols:
        latest = latest_by_symbol.get(sym)
        if latest is None:
            no_data.append(sym)
            stale.append((sym, date.today() - timedelta(days=365 * 5)))
        elif (to_date - latest).days > STALENESS_THRESHOLD_DAYS:
            stale.append((sym, latest + timedelta(days=1)))
        else:
            fresh += 1

    if no_data:
        print(f"  ⚠ {len(no_data)} symbols have NO data — will fetch full history")

    print(f"  → {fresh} fresh, {len(stale)} need refresh")

    if not stale:
        return {"total": len(symbols), "refreshed": 0, "fresh": fresh, "failed": 0}

    refreshed = 0
    failed = 0
    failed_symbols = []

    for i, (symbol, fetch_from) in enumerate(stale, 1):
        try:
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
            refreshed += 1
        except Exception as e:
            logger.error(f"Failed to refresh {symbol}: {e}")
            failed += 1
            failed_symbols.append(symbol)

        if i % 25 == 0 or i == len(stale):
            print(f"    [{i}/{len(stale)}] {refreshed} refreshed, {failed} failed")

    return {
        "total": len(symbols),
        "refreshed": refreshed,
        "fresh": fresh,
        "failed": failed,
        "failed_symbols": failed_symbols,
    }


async def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    print("╔══════════════════════════════════════╗")
    print("║   FINIAS Universe Data Refresher     ║")
    print("╚══════════════════════════════════════╝\n")

    db = DatabasePool()
    await db.initialize()
    await run_migrations(db)

    polygon = PolygonClient()

    try:
        result = await refresh_prices(db, polygon)
        print(f"\n  Summary: {result['refreshed']} refreshed, {result['fresh']} already fresh, "
              f"{result['failed']} failed out of {result['total']} symbols")
        if result["failed_symbols"]:
            print(f"  Failed: {', '.join(result['failed_symbols'][:20])}")
    finally:
        await polygon.close()
        await db.close()

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
