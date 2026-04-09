"""
FINIAS TA Signal Validation — Proper Statistical Tests

Tests:
1. Out-of-sample split (train 2022-2024, test 2025)
2. Non-overlapping returns (monthly signals only)
3. Crisis episode count (how many independent events?)
4. Win rate and distribution (not just averages)
"""

import asyncio
from datetime import date
from finias.core.database.connection import DatabasePool


async def main():
    db = DatabasePool()
    await db.initialize()

    # =========================================================================
    # 1. OUT-OF-SAMPLE SPLIT
    # =========================================================================
    print("=" * 70)
    print("  TEST 1: OUT-OF-SAMPLE SPLIT")
    print("  Train: 2022-2024 | Test: 2025+")
    print("  If pattern holds in test period, it's likely real.")
    print("=" * 70)

    for period, start, end in [
        ("TRAIN 2022-2024", date(2022, 1, 1), date(2024, 12, 31)),
        ("TEST 2025+", date(2025, 1, 1), date(2026, 12, 31)),
    ]:
        rows = await db.fetch(
            "WITH spy AS ("
            "  SELECT signal_date, fwd_return_20d as spy_20d "
            "  FROM technical_signals WHERE symbol = $1"
            ") "
            "SELECT t.macro_regime, t.trend_regime, "
            "  COUNT(*) as n, "
            "  ROUND((AVG(t.fwd_return_20d - s.spy_20d) * 100)::numeric, 2) as excess "
            "FROM technical_signals t "
            "JOIN spy s ON t.signal_date = s.signal_date "
            "WHERE t.fwd_return_20d IS NOT NULL "
            "  AND t.symbol != $1 "
            "  AND t.signal_date BETWEEN $2::date AND $3::date "
            "  AND t.macro_regime IS NOT NULL "
            "  AND t.trend_regime IN ($4, $5) "
            "GROUP BY t.macro_regime, t.trend_regime "
            "ORDER BY t.macro_regime, excess DESC",
            "SPY", start, end, "strong_downtrend", "strong_uptrend",
        )

        print(f"\n  {period}:")
        print(f"  {'Macro':<18} {'Trend':<22} {'N':>6} {'Excess 20d':>10}")
        print(f"  {'-'*18} {'-'*22} {'-'*6} {'-'*10}")
        for r in rows:
            print(f"  {r['macro_regime']:<18} {r['trend_regime']:<22} {r['n']:>6,} {r['excess']:>9.2f}%")

    # =========================================================================
    # 2. NON-OVERLAPPING RETURNS (one signal per symbol per month)
    # =========================================================================
    print()
    print("=" * 70)
    print("  TEST 2: NON-OVERLAPPING RETURNS")
    print("  One signal per symbol per month — eliminates autocorrelation.")
    print("=" * 70)

    rows = await db.fetch(
        "WITH monthly AS ("
        "  SELECT DISTINCT ON (symbol, date_trunc('month', signal_date)) "
        "    symbol, signal_date, trend_regime, divergence_type, "
        "    macro_regime, fwd_return_20d "
        "  FROM technical_signals "
        "  WHERE fwd_return_20d IS NOT NULL "
        "  ORDER BY symbol, date_trunc('month', signal_date), signal_date DESC"
        "), spy AS ("
        "  SELECT signal_date, fwd_return_20d as spy_20d "
        "  FROM monthly WHERE symbol = $1"
        ") "
        "SELECT m.macro_regime, m.trend_regime, "
        "  COUNT(*) as n, "
        "  ROUND((AVG(m.fwd_return_20d - s.spy_20d) * 100)::numeric, 2) as excess "
        "FROM monthly m "
        "JOIN spy s ON m.signal_date = s.signal_date "
        "WHERE m.symbol != $1 "
        "  AND m.macro_regime IS NOT NULL "
        "  AND m.trend_regime IN ($2, $3) "
        "GROUP BY m.macro_regime, m.trend_regime "
        "ORDER BY m.macro_regime, excess DESC",
        "SPY", "strong_downtrend", "strong_uptrend",
    )

    print(f"\n  {'Macro':<18} {'Trend':<22} {'N':>6} {'Excess 20d':>10}")
    print(f"  {'-'*18} {'-'*22} {'-'*6} {'-'*10}")
    for r in rows:
        print(f"  {r['macro_regime']:<18} {r['trend_regime']:<22} {r['n']:>6,} {r['excess']:>9.2f}%")

    # =========================================================================
    # 3. CRISIS EPISODE COUNT
    # =========================================================================
    print()
    print("=" * 70)
    print("  TEST 3: CRISIS EPISODES")
    print("  How many independent crisis periods drive the result?")
    print("=" * 70)

    rows = await db.fetch(
        "SELECT date_trunc('month', signal_date)::date as month, COUNT(*) as n "
        "FROM technical_signals "
        "WHERE macro_regime = $1 "
        "GROUP BY month ORDER BY month",
        "crisis",
    )
    print(f"\n  Crisis months: {len(rows)}")
    for r in rows:
        print(f"    {r['month']}: {r['n']:,} signals")

    # =========================================================================
    # 4. WIN RATE AND DISTRIBUTION (not just averages)
    # =========================================================================
    print()
    print("=" * 70)
    print("  TEST 4: WIN RATE & RETURN DISTRIBUTION")
    print("  Averages can be skewed by outliers. Check win rates and percentiles.")
    print("=" * 70)

    for regime, trend in [
        ("crisis", "strong_downtrend"),
        ("moderate_bull", "strong_downtrend"),
        ("risk_on", "strong_uptrend"),
    ]:
        row = await db.fetchrow(
            "WITH spy AS ("
            "  SELECT signal_date, fwd_return_20d as spy_20d "
            "  FROM technical_signals WHERE symbol = $1"
            ") "
            "SELECT "
            "  COUNT(*) as n, "
            "  ROUND((AVG(t.fwd_return_20d - s.spy_20d) * 100)::numeric, 2) as avg_excess, "
            "  ROUND((PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY t.fwd_return_20d - s.spy_20d) * 100)::numeric, 2) as p25, "
            "  ROUND((PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY t.fwd_return_20d - s.spy_20d) * 100)::numeric, 2) as p50, "
            "  ROUND((PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY t.fwd_return_20d - s.spy_20d) * 100)::numeric, 2) as p75, "
            "  ROUND((SUM(CASE WHEN t.fwd_return_20d - s.spy_20d > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100)::numeric, 1) as win_rate, "
            "  ROUND((MIN(t.fwd_return_20d - s.spy_20d) * 100)::numeric, 2) as worst, "
            "  ROUND((MAX(t.fwd_return_20d - s.spy_20d) * 100)::numeric, 2) as best "
            "FROM technical_signals t "
            "JOIN spy s ON t.signal_date = s.signal_date "
            "WHERE t.fwd_return_20d IS NOT NULL "
            "  AND t.symbol != $1 "
            "  AND t.macro_regime = $2 "
            "  AND t.trend_regime = $3",
            "SPY", regime, trend,
        )

        if row and row["n"] > 0:
            print(f"\n  {regime} + {trend} (N={row['n']:,}):")
            print(f"    Avg excess:  {row['avg_excess']:>7.2f}%")
            print(f"    Win rate:    {row['win_rate']:>7.1f}%")
            print(f"    P25 / P50 / P75: {row['p25']:>6.2f}% / {row['p50']:>6.2f}% / {row['p75']:>6.2f}%")
            print(f"    Worst / Best:    {row['worst']:>6.2f}% / {row['best']:>6.2f}%")

    # =========================================================================
    # 5. DIVERGENCE EDGE — TRAIN vs TEST
    # =========================================================================
    print()
    print("=" * 70)
    print("  TEST 5: DIVERGENCE EDGE — TRAIN vs TEST")
    print("  Does bullish divergence add alpha out-of-sample?")
    print("=" * 70)

    for period, start, end in [
        ("TRAIN 2022-2024", "2022-01-01", "2024-12-31"),
        ("TEST 2025+", "2025-01-01", "2026-12-31"),
    ]:
        rows = await db.fetch(
            "WITH spy AS ("
            "  SELECT signal_date, fwd_return_20d as spy_20d "
            "  FROM technical_signals WHERE symbol = $1"
            ") "
            "SELECT t.divergence_type, "
            "  COUNT(*) as n, "
            "  ROUND((AVG(t.fwd_return_20d - s.spy_20d) * 100)::numeric, 2) as excess "
            "FROM technical_signals t "
            "JOIN spy s ON t.signal_date = s.signal_date "
            "WHERE t.fwd_return_20d IS NOT NULL "
            "  AND t.symbol != $1 "
            "  AND t.signal_date BETWEEN $2::date AND $3::date "
            "  AND t.trend_regime = $4 "
            "GROUP BY t.divergence_type "
            "ORDER BY excess DESC",
            "SPY", start, end, "strong_downtrend",
        )

        print(f"\n  {period} — strong_downtrend by divergence:")
        print(f"  {'Divergence':<22} {'N':>7} {'Excess':>8}")
        print(f"  {'-'*22} {'-'*7} {'-'*8}")
        for r in rows:
            print(f"  {r['divergence_type'] or 'none':<22} {r['n']:>7,} {r['excess']:>7.2f}%")

    await db.close()
    print()
    print("=" * 70)
    print("  VALIDATION COMPLETE")
    print("=" * 70)


asyncio.run(main())