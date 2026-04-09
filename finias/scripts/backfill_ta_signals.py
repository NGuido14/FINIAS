"""
FINIAS TA Signal Backfill

Walks through 2+ years of historical dates at weekly intervals, computes
TA signals for all symbols AS OF each date (no look-ahead), and stores
in technical_signals. Then computes forward returns from actual future prices.

Usage:
    python -m finias.scripts.backfill_ta_signals                    # Full backfill
    python -m finias.scripts.backfill_ta_signals --weeks 52         # Last year only
    python -m finias.scripts.backfill_ta_signals --symbols SPY AAPL # Specific symbols
    python -m finias.scripts.backfill_ta_signals --report           # Accuracy report only

Performance: ~15-30 minutes for 120 weeks × 500 symbols.
Cost: $0.00 (pure Python, no API calls).
"""

from __future__ import annotations
import asyncio
import logging
import sys
import json
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from finias.core.database.connection import DatabasePool
from finias.core.database.migrations import run_migrations

logger = logging.getLogger("finias.scripts.backfill_ta")

# Computation imports
from finias.agents.technical_analyst.computations.trend import analyze_trend
from finias.agents.technical_analyst.computations.momentum import analyze_momentum
from finias.agents.technical_analyst.computations.levels import analyze_levels

DEFAULT_WEEKS = 120  # ~2.3 years
STEP_DAYS = 7  # Weekly intervals
MIN_BARS = 200  # Need 200 bars for 200-day MA
LOOKBACK_BARS = 504  # ~2 years of trading days to load per computation


async def backfill_signals(
    db: DatabasePool,
    weeks: int = DEFAULT_WEEKS,
    symbols: list[str] = None,
    skip_existing: bool = True,
) -> dict:
    """
    Walk through historical dates and compute TA signals.

    Args:
        db: Database pool.
        weeks: How many weeks back to compute.
        symbols: Specific symbols (default: all active in universe).
        skip_existing: Skip dates that already have signals.

    Returns:
        Summary dict.
    """
    # Get symbol list
    if symbols is None:
        rows = await db.fetch(
            "SELECT DISTINCT symbol FROM symbol_universe WHERE is_active ORDER BY symbol"
        )
        symbols = [r["symbol"] for r in rows]
        if not symbols:
            # Fallback to whatever is in market_data_daily
            rows = await db.fetch("SELECT DISTINCT symbol FROM market_data_daily ORDER BY symbol")
            symbols = [r["symbol"] for r in rows]

    print(f"  → Backfilling {weeks} weeks for {len(symbols)} symbols...")

    # Determine date range
    end_date = date.today() - timedelta(days=1)  # Yesterday (ensure bars exist)
    start_date = end_date - timedelta(weeks=weeks)

    # Load ALL price data once (much faster than per-date queries)
    print(f"  → Loading price data...")
    data_start = start_date - timedelta(days=LOOKBACK_BARS * 2)  # Extra for MA computation
    all_bars = await _load_all_bars(db, symbols, data_start, end_date)
    print(f"    Loaded {sum(len(v) for v in all_bars.values())} bars for {len(all_bars)} symbols")

    # Get existing signal dates to skip
    existing = set()
    if skip_existing:
        rows = await db.fetch(
            "SELECT DISTINCT signal_date FROM technical_signals"
        )
        existing = {r["signal_date"] for r in rows}
        if existing:
            print(f"    Skipping {len(existing)} dates with existing signals")

    # Walk through dates
    current = start_date
    total_dates = 0
    total_signals = 0
    skipped_dates = 0

    while current <= end_date:
        if current in existing:
            skipped_dates += 1
            current += timedelta(days=STEP_DAYS)
            continue

        # Compute signals for this date
        date_signals = _compute_signals_as_of(all_bars, symbols, current)

        if date_signals:
            await _persist_batch(db, date_signals, current)
            total_signals += len(date_signals)

        total_dates += 1
        if total_dates % 10 == 0:
            print(f"    [{total_dates} dates] {total_signals} signals computed...")

        current += timedelta(days=STEP_DAYS)

    print(f"  → Backfill complete: {total_dates} dates, {total_signals} signals")
    return {
        "dates_computed": total_dates,
        "dates_skipped": skipped_dates,
        "signals_stored": total_signals,
        "symbols": len(symbols),
    }


def _compute_signals_as_of(
    all_bars: dict[str, pd.DataFrame],
    symbols: list[str],
    as_of_date: date,
) -> list[dict]:
    """
    Compute TA signals for all symbols using only data available as of as_of_date.
    No look-ahead bias — future bars are excluded.
    """
    results = []

    for symbol in symbols:
        if symbol not in all_bars:
            continue

        full_df = all_bars[symbol]

        # Filter to only bars on or before as_of_date (NO LOOK-AHEAD)
        df = full_df[full_df["trade_date"] <= as_of_date].copy()

        if len(df) < MIN_BARS:
            continue

        # Take last LOOKBACK_BARS for computation
        df = df.tail(LOOKBACK_BARS).reset_index(drop=True)

        try:
            # Same sequential pipeline as live agent
            trend = analyze_trend(df, symbol=symbol)
            momentum = analyze_momentum(df, symbol=symbol, trend_regime=trend.trend_regime)
            levels = analyze_levels(df, symbol=symbol)

            close_price = float(df["close"].iloc[-1])

            results.append({
                "symbol": symbol,
                "close_price": close_price,
                "trend": trend.to_dict(),
                "momentum": momentum.to_dict(),
                "levels": levels.to_dict(),
            })
        except Exception as e:
            logger.debug(f"Computation failed for {symbol} on {as_of_date}: {e}")

    return results


async def _persist_batch(db: DatabasePool, signals: list[dict], signal_date: date):
    """Persist a batch of signals for a single date."""
    for sig in signals:
        trend = sig["trend"]
        mom = sig["momentum"]
        levels = sig["levels"]

        full_json = json.dumps({"trend": trend, "momentum": mom, "levels": levels}, default=str)

        await db.execute(
            """
            INSERT INTO technical_signals (
                symbol, signal_date, close_price,
                trend_regime, trend_score, adx, ma_alignment, ichimoku_signal, trend_maturity,
                momentum_score, rsi_14, rsi_zone, macd_direction, macd_cross, divergence_type,
                nearest_support, nearest_resistance, risk_reward_ratio,
                full_signals_json
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
            )
            ON CONFLICT (symbol, signal_date) DO UPDATE SET
                close_price = EXCLUDED.close_price,
                trend_regime = EXCLUDED.trend_regime,
                trend_score = EXCLUDED.trend_score,
                adx = EXCLUDED.adx,
                ma_alignment = EXCLUDED.ma_alignment,
                ichimoku_signal = EXCLUDED.ichimoku_signal,
                trend_maturity = EXCLUDED.trend_maturity,
                momentum_score = EXCLUDED.momentum_score,
                rsi_14 = EXCLUDED.rsi_14,
                rsi_zone = EXCLUDED.rsi_zone,
                macd_direction = EXCLUDED.macd_direction,
                macd_cross = EXCLUDED.macd_cross,
                divergence_type = EXCLUDED.divergence_type,
                nearest_support = EXCLUDED.nearest_support,
                nearest_resistance = EXCLUDED.nearest_resistance,
                risk_reward_ratio = EXCLUDED.risk_reward_ratio,
                full_signals_json = EXCLUDED.full_signals_json
            """,
            sig["symbol"], signal_date, sig["close_price"],
            trend.get("trend_regime"),
            trend.get("trend_score"),
            trend.get("adx", {}).get("value"),
            trend.get("ma", {}).get("alignment"),
            trend.get("ichimoku", {}).get("signal"),
            trend.get("maturity", {}).get("stage"),
            mom.get("momentum_score"),
            mom.get("rsi", {}).get("value"),
            mom.get("rsi", {}).get("zone"),
            mom.get("macd", {}).get("direction"),
            mom.get("macd", {}).get("cross"),
            mom.get("divergence", {}).get("type"),
            levels.get("nearest_support"),
            levels.get("nearest_resistance"),
            min(levels.get("risk_reward_ratio") or 0, 999.99),
            full_json,
        )


async def compute_forward_returns(db: DatabasePool) -> dict:
    """
    Fill in forward returns for all historical signals using actual future prices.

    For each signal at date D with close price P:
      fwd_return_5d = close(D+5) / P - 1
      fwd_return_20d = close(D+20) / P - 1
      fwd_return_60d = close(D+60) / P - 1
      fwd_max_drawdown_20d = min(low(D+1..D+20)) / P - 1
    """
    print("  → Computing forward returns...")

    # Get all signals that need forward returns
    signals = await db.fetch(
        """
        SELECT id, symbol, signal_date, close_price
        FROM technical_signals
        WHERE close_price IS NOT NULL AND fwd_return_5d IS NULL
        ORDER BY signal_date ASC
        """
    )

    if not signals:
        print("    No signals need forward returns (all computed or no signals)")
        return {"updated": 0}

    print(f"    {len(signals)} signals need forward returns")

    updated = 0
    for i, sig in enumerate(signals):
        symbol = sig["symbol"]
        sig_date = sig["signal_date"]
        sig_price = float(sig["close_price"])

        if sig_price <= 0:
            continue

        # Get future prices for this symbol
        future_bars = await db.fetch(
            """
            SELECT trade_date, close, low
            FROM market_data_daily
            WHERE symbol = $1 AND trade_date > $2
            ORDER BY trade_date ASC
            LIMIT 65
            """,
            symbol, sig_date,
        )

        if not future_bars:
            continue

        # Compute returns
        fwd_5d = None
        fwd_20d = None
        fwd_60d = None
        max_dd_20d = None

        closes = [(r["trade_date"], float(r["close"])) for r in future_bars]
        lows = [float(r["low"]) for r in future_bars[:20]]

        if len(closes) >= 5:
            fwd_5d = closes[4][1] / sig_price - 1
        if len(closes) >= 20:
            fwd_20d = closes[19][1] / sig_price - 1
        if len(closes) >= 60:
            fwd_60d = closes[59][1] / sig_price - 1
        if lows:
            min_low = min(lows)
            max_dd_20d = min_low / sig_price - 1

        # Update the signal row
        await db.execute(
            """
            UPDATE technical_signals
            SET fwd_return_5d = $1, fwd_return_20d = $2, fwd_return_60d = $3,
                fwd_max_drawdown_20d = $4
            WHERE id = $5
            """,
            fwd_5d, fwd_20d, fwd_60d, max_dd_20d, sig["id"],
        )
        updated += 1

        if (i + 1) % 5000 == 0:
            print(f"    [{i+1}/{len(signals)}] forward returns computed...")

    print(f"  → Forward returns: {updated} signals updated")
    return {"updated": updated}


async def generate_accuracy_report(db: DatabasePool):
    """
    Print a signal accuracy report showing forward returns by signal type.

    This is the key output — it tells us whether our TA signals predict returns.
    """
    def _fmt(v):
        """Format a numeric value or None."""
        return f"{v:>7.2f}%" if v is not None else "    N/A"

    def _fmt1(v):
        """Format a numeric value with 1 decimal or None."""
        return f"{v:>8.1f}%" if v is not None else "     N/A"

    print("\n" + "=" * 70)
    print("  FINIAS TA SIGNAL ACCURACY REPORT")
    print("=" * 70)

    # Count total signals
    total = await db.fetchval("SELECT COUNT(*) FROM technical_signals")
    with_returns = await db.fetchval(
        "SELECT COUNT(*) FROM technical_signals WHERE fwd_return_20d IS NOT NULL"
    )
    print(f"\n  Total signals: {total:,}")
    print(f"  With forward returns: {with_returns:,}")

    if with_returns == 0:
        print("  No forward returns computed yet. Run backfill first.")
        return

    # === TREND REGIME ACCURACY ===
    print(f"\n{'─' * 70}")
    print("  FORWARD RETURNS BY TREND REGIME")
    print(f"  (Does our trend classification predict direction?)")
    print(f"{'─' * 70}")

    rows = await db.fetch(
        """
        SELECT trend_regime,
               COUNT(*) as n,
               ROUND((AVG(fwd_return_5d) * 100)::numeric, 2) as avg_5d,
               ROUND((AVG(fwd_return_20d) * 100)::numeric, 2) as avg_20d,
               ROUND((AVG(fwd_return_60d) * 100)::numeric, 2) as avg_60d,
               ROUND((AVG(fwd_max_drawdown_20d) * 100)::numeric, 2) as avg_dd,
               ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fwd_return_20d) * 100)::numeric, 2) as median_20d
        FROM technical_signals
        WHERE fwd_return_20d IS NOT NULL AND trend_regime IS NOT NULL
        GROUP BY trend_regime
        ORDER BY avg_20d DESC
        """
    )

    print(f"\n  {'Regime':<22} {'N':>7} {'Avg 5d':>8} {'Avg 20d':>8} {'Med 20d':>8} {'Avg 60d':>8} {'Avg DD':>8}")
    print(f"  {'─' * 22} {'─' * 7} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
    for r in rows:
        print(f"  {r['trend_regime'] or 'unknown':<22} {r['n']:>7,} {_fmt(r['avg_5d'])} {_fmt(r['avg_20d'])} {_fmt(r['median_20d'])} {_fmt(r['avg_60d'])} {_fmt(r['avg_dd'])}")

    # === MOMENTUM DIVERGENCE ACCURACY ===
    print(f"\n{'─' * 70}")
    print("  FORWARD RETURNS BY DIVERGENCE TYPE")
    print(f"  (Do our divergences predict reversals?)")
    print(f"{'─' * 70}")

    rows = await db.fetch(
        """
        SELECT divergence_type,
               COUNT(*) as n,
               ROUND((AVG(fwd_return_5d) * 100)::numeric, 2) as avg_5d,
               ROUND((AVG(fwd_return_20d) * 100)::numeric, 2) as avg_20d,
               ROUND((AVG(fwd_return_60d) * 100)::numeric, 2) as avg_60d
        FROM technical_signals
        WHERE fwd_return_20d IS NOT NULL AND divergence_type IS NOT NULL
        GROUP BY divergence_type
        ORDER BY n DESC
        """
    )

    print(f"\n  {'Divergence':<22} {'N':>7} {'Avg 5d':>8} {'Avg 20d':>8} {'Avg 60d':>8}")
    print(f"  {'─' * 22} {'─' * 7} {'─' * 8} {'─' * 8} {'─' * 8}")
    for r in rows:
        print(f"  {r['divergence_type'] or 'none':<22} {r['n']:>7,} {_fmt(r['avg_5d'])} {_fmt(r['avg_20d'])} {_fmt(r['avg_60d'])}")

    # === TREND SCORE QUINTILE ACCURACY ===
    print(f"\n{'─' * 70}")
    print("  FORWARD RETURNS BY TREND SCORE QUINTILE")
    print(f"  (Does higher trend score → higher returns?)")
    print(f"{'─' * 70}")

    rows = await db.fetch(
        """
        SELECT
            CASE
                WHEN trend_score >= 0.4 THEN 'Q5 (strong bull)'
                WHEN trend_score >= 0.1 THEN 'Q4 (bull)'
                WHEN trend_score >= -0.1 THEN 'Q3 (neutral)'
                WHEN trend_score >= -0.4 THEN 'Q2 (bear)'
                ELSE 'Q1 (strong bear)'
            END as quintile,
            COUNT(*) as n,
            ROUND((AVG(fwd_return_5d) * 100)::numeric, 2) as avg_5d,
            ROUND((AVG(fwd_return_20d) * 100)::numeric, 2) as avg_20d,
            ROUND((AVG(fwd_return_60d) * 100)::numeric, 2) as avg_60d
        FROM technical_signals
        WHERE fwd_return_20d IS NOT NULL AND trend_score IS NOT NULL
        GROUP BY quintile
        ORDER BY quintile DESC
        """
    )

    print(f"\n  {'Quintile':<22} {'N':>7} {'Avg 5d':>8} {'Avg 20d':>8} {'Avg 60d':>8}")
    print(f"  {'─' * 22} {'─' * 7} {'─' * 8} {'─' * 8} {'─' * 8}")
    for r in rows:
        print(f"  {r['quintile']:<22} {r['n']:>7,} {_fmt(r['avg_5d'])} {_fmt(r['avg_20d'])} {_fmt(r['avg_60d'])}")

    # === MOMENTUM SCORE QUINTILE ACCURACY ===
    print(f"\n{'─' * 70}")
    print("  FORWARD RETURNS BY MOMENTUM SCORE QUINTILE")
    print(f"  (Does higher momentum → higher returns?)")
    print(f"{'─' * 70}")

    rows = await db.fetch(
        """
        SELECT
            CASE
                WHEN momentum_score >= 0.4 THEN 'Q5 (strong bull)'
                WHEN momentum_score >= 0.1 THEN 'Q4 (bull)'
                WHEN momentum_score >= -0.1 THEN 'Q3 (neutral)'
                WHEN momentum_score >= -0.4 THEN 'Q2 (bear)'
                ELSE 'Q1 (strong bear)'
            END as quintile,
            COUNT(*) as n,
            ROUND((AVG(fwd_return_5d) * 100)::numeric, 2) as avg_5d,
            ROUND((AVG(fwd_return_20d) * 100)::numeric, 2) as avg_20d,
            ROUND((AVG(fwd_return_60d) * 100)::numeric, 2) as avg_60d
        FROM technical_signals
        WHERE fwd_return_20d IS NOT NULL AND momentum_score IS NOT NULL
        GROUP BY quintile
        ORDER BY quintile DESC
        """
    )

    print(f"\n  {'Quintile':<22} {'N':>7} {'Avg 5d':>8} {'Avg 20d':>8} {'Avg 60d':>8}")
    print(f"  {'─' * 22} {'─' * 7} {'─' * 8} {'─' * 8} {'─' * 8}")
    for r in rows:
        print(f"  {r['quintile']:<22} {r['n']:>7,} {_fmt(r['avg_5d'])} {_fmt(r['avg_20d'])} {_fmt(r['avg_60d'])}")

    # === HIT RATE ===
    print(f"\n{'─' * 70}")
    print("  HIT RATE (% of signals where 20d return matched predicted direction)")
    print(f"{'─' * 70}")

    rows = await db.fetch(
        """
        SELECT trend_regime,
               COUNT(*) as n,
               SUM(CASE
                   WHEN trend_regime IN ('strong_uptrend', 'uptrend') AND fwd_return_20d > 0 THEN 1
                   WHEN trend_regime IN ('strong_downtrend', 'downtrend') AND fwd_return_20d < 0 THEN 1
                   WHEN trend_regime = 'consolidation' AND ABS(fwd_return_20d) < 0.03 THEN 1
                   ELSE 0
               END) as hits,
               ROUND(
                   (SUM(CASE
                       WHEN trend_regime IN ('strong_uptrend', 'uptrend') AND fwd_return_20d > 0 THEN 1
                       WHEN trend_regime IN ('strong_downtrend', 'downtrend') AND fwd_return_20d < 0 THEN 1
                       WHEN trend_regime = 'consolidation' AND ABS(fwd_return_20d) < 0.03 THEN 1
                       ELSE 0
                   END)::numeric / NULLIF(COUNT(*), 0) * 100)::numeric, 1
               ) as hit_rate
        FROM technical_signals
        WHERE fwd_return_20d IS NOT NULL AND trend_regime IS NOT NULL
        GROUP BY trend_regime
        ORDER BY hit_rate DESC
        """
    )

    print(f"\n  {'Regime':<22} {'N':>7} {'Hits':>7} {'Hit Rate':>9}")
    print(f"  {'─' * 22} {'─' * 7} {'─' * 7} {'─' * 9}")
    for r in rows:
        print(f"  {r['trend_regime'] or 'unknown':<22} {r['n']:>7,} {r['hits']:>7,} {_fmt1(r['hit_rate'])}")

    print(f"\n{'=' * 70}")


async def _load_all_bars(
    db: DatabasePool,
    symbols: list[str],
    from_date: date,
    to_date: date,
) -> dict[str, pd.DataFrame]:
    """Load all price data as DataFrames, partitioned by symbol."""
    rows = await db.fetch(
        """
        SELECT symbol, trade_date, open, high, low, close, volume
        FROM market_data_daily
        WHERE symbol = ANY($1) AND trade_date BETWEEN $2 AND $3
        ORDER BY symbol, trade_date ASC
        """,
        symbols, from_date, to_date,
    )

    result = {}
    current_symbol = None
    current_rows = []

    for row in rows:
        sym = row["symbol"]
        if sym != current_symbol:
            if current_symbol and current_rows:
                df = pd.DataFrame(current_rows)
                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
                result[current_symbol] = df
            current_symbol = sym
            current_rows = []
        current_rows.append({
            "trade_date": row["trade_date"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        })

    # Last symbol
    if current_symbol and current_rows:
        df = pd.DataFrame(current_rows)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        result[current_symbol] = df

    return result


async def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    args = set(sys.argv[1:])
    report_only = "--report" in args

    # Parse --weeks N
    weeks = DEFAULT_WEEKS
    if "--weeks" in sys.argv:
        idx = sys.argv.index("--weeks")
        if idx + 1 < len(sys.argv):
            weeks = int(sys.argv[idx + 1])

    # Parse --symbols SYM1 SYM2
    symbols = None
    if "--symbols" in sys.argv:
        idx = sys.argv.index("--symbols")
        symbols = [s.upper() for s in sys.argv[idx + 1:] if not s.startswith("--")]

    print("╔══════════════════════════════════════╗")
    print("║   FINIAS TA Signal Backfill          ║")
    print("╚══════════════════════════════════════╝\n")

    db = DatabasePool()
    await db.initialize()
    await run_migrations(db)

    try:
        if report_only:
            await generate_accuracy_report(db)
        else:
            # Step 1: Backfill signals
            result = await backfill_signals(db, weeks=weeks, symbols=symbols)
            print(f"\n  Backfill: {result['signals_stored']:,} signals for "
                  f"{result['symbols']} symbols over {result['dates_computed']} dates")

            # Step 2: Compute forward returns
            fwd = await compute_forward_returns(db)
            print(f"  Forward returns: {fwd['updated']:,} signals updated")

            # Step 3: Generate accuracy report
            await generate_accuracy_report(db)

    finally:
        await db.close()

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
