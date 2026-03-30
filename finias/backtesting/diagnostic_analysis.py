"""
FINIAS Backtest Diagnostic Analysis

Analyzes the 196 scored observations from the walk-forward backtest
to determine whether momentum/rate-of-change signals predict returns
BEFORE committing to building a momentum layer.

Tests:
1. Does the 4-week CHANGE in composite score predict forward returns?
2. Do binding constraint TRANSITIONS predict returns?
3. Does composite IMPROVEMENT (regardless of level) predict returns?
4. Which individual category's rate-of-change is most predictive?
5. Does a simple momentum-enhanced signal beat the raw composite?

Run: python finias/backtesting/diagnostic_analysis.py
"""

import asyncio
import numpy as np
from collections import defaultdict
from finias.core.database.connection import DatabasePool
from finias.core.database.migrations import run_migrations


async def main():
    db = DatabasePool()
    await db.initialize()
    await run_migrations(db)

    # Load all scored (non-warmup) results ordered by date
    rows = await db.fetch("""
        SELECT sim_date, composite_score, growth_score, monetary_score,
               inflation_score, market_score, primary_regime, stress_index,
               binding_constraint, confidence,
               spx_fwd_5d, spx_fwd_20d, spx_fwd_60d, spx_max_dd_20d
        FROM backtest_results
        WHERE warmup = FALSE AND composite_score IS NOT NULL
        ORDER BY sim_date ASC
    """)

    if not rows:
        print("No scored results found.")
        await db.close()
        return

    # Convert to arrays
    dates = [r["sim_date"] for r in rows]
    composite = np.array([float(r["composite_score"]) for r in rows])
    growth = np.array([float(r["growth_score"]) for r in rows])
    monetary = np.array([float(r["monetary_score"]) for r in rows])
    inflation = np.array([float(r["inflation_score"]) for r in rows])
    market = np.array([float(r["market_score"]) for r in rows])
    stress = np.array([float(r["stress_index"]) for r in rows])
    regimes = [r["primary_regime"] for r in rows]
    binding = [r["binding_constraint"] for r in rows]

    fwd_5d = np.array([float(r["spx_fwd_5d"]) if r["spx_fwd_5d"] is not None else np.nan for r in rows])
    fwd_20d = np.array([float(r["spx_fwd_20d"]) if r["spx_fwd_20d"] is not None else np.nan for r in rows])
    fwd_60d = np.array([float(r["spx_fwd_60d"]) if r["spx_fwd_60d"] is not None else np.nan for r in rows])
    max_dd = np.array([float(r["spx_max_dd_20d"]) if r["spx_max_dd_20d"] is not None else np.nan for r in rows])

    n = len(rows)

    def safe_corr(a, b):
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 10:
            return None
        c = np.corrcoef(a[mask], b[mask])[0, 1]
        return float(c) if not np.isnan(c) else None

    print("\n" + "=" * 70)
    print("  FINIAS BACKTEST DIAGNOSTIC ANALYSIS")
    print("  Testing Momentum Hypotheses on Existing Data")
    print("=" * 70)
    print(f"  Observations: {n}")
    print(f"  Period: {dates[0]} to {dates[-1]}")

    # ================================================================
    # TEST 1: Does the 4-week CHANGE in composite predict returns?
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  TEST 1: COMPOSITE RATE-OF-CHANGE (4-week change)")
    print(f"{'=' * 70}")

    for lookback_name, lookback in [("2-week", 2), ("4-week", 4), ("8-week", 8), ("13-week", 13)]:
        composite_roc = np.full(n, np.nan)
        for i in range(lookback, n):
            composite_roc[i] = composite[i] - composite[i - lookback]

        for horizon_name, fwd in [("5d", fwd_5d), ("20d", fwd_20d), ("60d", fwd_60d)]:
            corr = safe_corr(composite_roc, fwd)
            corr_str = f"{corr:+.3f}" if corr is not None else "N/A"
            print(f"  {lookback_name:10s} change vs {horizon_name:3s} fwd return: {corr_str}")
        print()

    # ================================================================
    # TEST 2: Does composite IMPROVEMENT predict positive returns?
    # ================================================================
    print(f"{'=' * 70}")
    print("  TEST 2: COMPOSITE IMPROVING vs DETERIORATING (4-week)")
    print(f"{'=' * 70}")

    composite_4w_change = np.full(n, np.nan)
    for i in range(4, n):
        composite_4w_change[i] = composite[i] - composite[i - 4]

    for horizon_name, fwd in [("20d", fwd_20d), ("60d", fwd_60d)]:
        improving_mask = composite_4w_change > 0.02
        deteriorating_mask = composite_4w_change < -0.02
        flat_mask = (~improving_mask) & (~deteriorating_mask) & (~np.isnan(composite_4w_change))

        imp_mask = improving_mask & ~np.isnan(fwd)
        det_mask = deteriorating_mask & ~np.isnan(fwd)
        flt_mask = flat_mask & ~np.isnan(fwd)

        print(f"\n  {horizon_name} forward returns:")
        if imp_mask.sum() > 0:
            print(f"    Improving  (Δ > +0.02): {imp_mask.sum():3d} obs, "
                  f"avg return {np.mean(fwd[imp_mask]):+.2f}%, "
                  f"hit rate {np.mean(fwd[imp_mask] > 0):.0%}")
        if det_mask.sum() > 0:
            print(f"    Deteriorating (Δ < -0.02): {det_mask.sum():3d} obs, "
                  f"avg return {np.mean(fwd[det_mask]):+.2f}%, "
                  f"hit rate {np.mean(fwd[det_mask] > 0):.0%}")
        if flt_mask.sum() > 0:
            print(f"    Flat       (|Δ| ≤ 0.02): {flt_mask.sum():3d} obs, "
                  f"avg return {np.mean(fwd[flt_mask]):+.2f}%, "
                  f"hit rate {np.mean(fwd[flt_mask] > 0):.0%}")

    # ================================================================
    # TEST 3: Do BINDING CONSTRAINT TRANSITIONS predict returns?
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  TEST 3: BINDING CONSTRAINT TRANSITIONS")
    print(f"{'=' * 70}")

    # Track transitions
    transition_returns_20d = defaultdict(list)
    transition_returns_60d = defaultdict(list)
    no_transition_returns_20d = []
    no_transition_returns_60d = []

    for i in range(1, n):
        if binding[i] != binding[i-1] and binding[i] is not None and binding[i-1] is not None:
            key = f"{binding[i-1]} → {binding[i]}"
            if not np.isnan(fwd_20d[i]):
                transition_returns_20d[key].append(fwd_20d[i])
            if not np.isnan(fwd_60d[i]):
                transition_returns_60d[key].append(fwd_60d[i])
        else:
            if not np.isnan(fwd_20d[i]):
                no_transition_returns_20d.append(fwd_20d[i])
            if not np.isnan(fwd_60d[i]):
                no_transition_returns_60d.append(fwd_60d[i])

    print(f"\n  20-day forward returns after binding constraint shifts:")
    for key, rets in sorted(transition_returns_20d.items(), key=lambda x: -len(x[1])):
        avg = np.mean(rets)
        print(f"    {key:45s}: {len(rets):3d} obs, avg {avg:+.2f}%")

    if no_transition_returns_20d:
        print(f"    {'No transition':45s}: {len(no_transition_returns_20d):3d} obs, "
              f"avg {np.mean(no_transition_returns_20d):+.2f}%")

    print(f"\n  60-day forward returns after binding constraint shifts:")
    for key, rets in sorted(transition_returns_60d.items(), key=lambda x: -len(x[1])):
        avg = np.mean(rets)
        print(f"    {key:45s}: {len(rets):3d} obs, avg {avg:+.2f}%")

    if no_transition_returns_60d:
        print(f"    {'No transition':45s}: {len(no_transition_returns_60d):3d} obs, "
              f"avg {np.mean(no_transition_returns_60d):+.2f}%")

    # Aggregate: ANY transition vs no transition
    all_trans_20d = [r for rets in transition_returns_20d.values() for r in rets]
    all_trans_60d = [r for rets in transition_returns_60d.values() for r in rets]
    if all_trans_20d and no_transition_returns_20d:
        print(f"\n  AGGREGATE — Any transition vs no transition:")
        print(f"    Transition week (20d):    {len(all_trans_20d):3d} obs, avg {np.mean(all_trans_20d):+.2f}%")
        print(f"    No transition (20d):      {len(no_transition_returns_20d):3d} obs, avg {np.mean(no_transition_returns_20d):+.2f}%")
        if all_trans_60d and no_transition_returns_60d:
            print(f"    Transition week (60d):    {len(all_trans_60d):3d} obs, avg {np.mean(all_trans_60d):+.2f}%")
            print(f"    No transition (60d):      {len(no_transition_returns_60d):3d} obs, avg {np.mean(no_transition_returns_60d):+.2f}%")

    # ================================================================
    # TEST 4: Individual CATEGORY rate-of-change — which is most predictive?
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  TEST 4: CATEGORY RATE-OF-CHANGE CORRELATIONS (4-week change)")
    print(f"{'=' * 70}")

    for cat_name, cat_data in [("Growth", growth), ("Monetary", monetary),
                                ("Inflation", inflation), ("Market", market),
                                ("Stress", stress)]:
        cat_roc = np.full(n, np.nan)
        for i in range(4, n):
            cat_roc[i] = cat_data[i] - cat_data[i - 4]

        print(f"\n  {cat_name} 4-week change:")
        for horizon_name, fwd in [("5d", fwd_5d), ("20d", fwd_20d), ("60d", fwd_60d)]:
            corr = safe_corr(cat_roc, fwd)
            corr_str = f"{corr:+.3f}" if corr is not None else "N/A"
            print(f"    vs {horizon_name:3s} fwd return: {corr_str}")

    # ================================================================
    # TEST 5: COMPOSITE LEVEL vs RATE-OF-CHANGE combined signal
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  TEST 5: COMBINED SIGNAL (level + momentum)")
    print(f"{'=' * 70}")

    # Create a combined signal: 50% level + 50% 4-week momentum (normalized)
    composite_roc_4w = np.full(n, np.nan)
    for i in range(4, n):
        composite_roc_4w[i] = composite[i] - composite[i - 4]

    # Normalize both to similar scales
    valid = ~np.isnan(composite_roc_4w)
    if valid.sum() > 20:
        level_norm = (composite - np.mean(composite[valid])) / (np.std(composite[valid]) + 1e-8)
        roc_norm = np.full(n, np.nan)
        roc_mean = np.nanmean(composite_roc_4w)
        roc_std = np.nanstd(composite_roc_4w)
        roc_norm[valid] = (composite_roc_4w[valid] - roc_mean) / (roc_std + 1e-8)

        for level_weight in [1.0, 0.75, 0.50, 0.25, 0.0]:
            roc_weight = 1.0 - level_weight
            combined = np.full(n, np.nan)
            combined[valid] = level_weight * level_norm[valid] + roc_weight * roc_norm[valid]

            label = f"Level {level_weight:.0%} + Momentum {roc_weight:.0%}"
            print(f"\n  {label}:")
            for horizon_name, fwd in [("5d", fwd_5d), ("20d", fwd_20d), ("60d", fwd_60d)]:
                corr = safe_corr(combined, fwd)
                corr_str = f"{corr:+.3f}" if corr is not None else "N/A"
                print(f"    vs {horizon_name:3s} fwd return: {corr_str}")

    # ================================================================
    # TEST 6: INFLATION SURPRISE — actual vs expected
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  TEST 6: INFLATION SCORE TRAJECTORY (not level)")
    print(f"{'=' * 70}")

    infl_roc_4w = np.full(n, np.nan)
    for i in range(4, n):
        infl_roc_4w[i] = inflation[i] - inflation[i - 4]

    # When inflation pressure is EASING (score becoming less negative / more positive)
    easing = infl_roc_4w > 0.02
    tightening = infl_roc_4w < -0.02
    infl_flat = (~easing) & (~tightening) & (~np.isnan(infl_roc_4w))

    for horizon_name, fwd in [("20d", fwd_20d), ("60d", fwd_60d)]:
        e_mask = easing & ~np.isnan(fwd)
        t_mask = tightening & ~np.isnan(fwd)
        f_mask = infl_flat & ~np.isnan(fwd)

        print(f"\n  {horizon_name} forward returns:")
        if e_mask.sum() > 0:
            print(f"    Inflation EASING  (Δ > +0.02): {e_mask.sum():3d} obs, "
                  f"avg {np.mean(fwd[e_mask]):+.2f}%, "
                  f"hit rate {np.mean(fwd[e_mask] > 0):.0%}")
        if t_mask.sum() > 0:
            print(f"    Inflation TIGHTENING (Δ < -0.02): {t_mask.sum():3d} obs, "
                  f"avg {np.mean(fwd[t_mask]):+.2f}%, "
                  f"hit rate {np.mean(fwd[t_mask] > 0):.0%}")
        if f_mask.sum() > 0:
            print(f"    Inflation STABLE  (|Δ| ≤ 0.02): {f_mask.sum():3d} obs, "
                  f"avg {np.mean(fwd[f_mask]):+.2f}%, "
                  f"hit rate {np.mean(fwd[f_mask] > 0):.0%}")

    # ================================================================
    # TEST 7: Raw composite distribution — understand the bias
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  TEST 7: COMPOSITE SCORE DISTRIBUTION")
    print(f"{'=' * 70}")

    percentiles = [0, 10, 25, 50, 75, 90, 100]
    pct_values = np.percentile(composite, percentiles)
    print(f"\n  Composite score percentiles:")
    for p, v in zip(percentiles, pct_values):
        print(f"    {p:3d}th percentile: {v:+.3f}")

    print(f"\n  Composite score by year:")
    year_groups = defaultdict(list)
    for i in range(n):
        year_groups[dates[i].year].append(composite[i])

    for year in sorted(year_groups.keys()):
        vals = year_groups[year]
        print(f"    {year}: avg {np.mean(vals):+.3f}, min {np.min(vals):+.3f}, "
              f"max {np.max(vals):+.3f}, count {len(vals)}")

    # ================================================================
    # TEST 8: REGIME TRANSITIONS with direction
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  TEST 8: REGIME TRANSITION DIRECTION")
    print(f"{'=' * 70}")

    for i in range(1, n):
        if regimes[i] != regimes[i-1]:
            ret_20 = f"{fwd_20d[i]:+.2f}%" if not np.isnan(fwd_20d[i]) else "N/A"
            print(f"  {dates[i]}: {regimes[i-1]:12s} → {regimes[i]:12s}  "
                  f"20d return: {ret_20}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  SUMMARY — KEY FINDINGS")
    print(f"{'=' * 70}")

    # Best momentum correlation
    best_roc_corr = None
    best_roc_label = ""
    for lookback in [2, 4, 8, 13]:
        roc = np.full(n, np.nan)
        for i in range(lookback, n):
            roc[i] = composite[i] - composite[i - lookback]
        c = safe_corr(roc, fwd_20d)
        if c is not None and (best_roc_corr is None or abs(c) > abs(best_roc_corr)):
            best_roc_corr = c
            best_roc_label = f"{lookback}-week"

    level_corr = safe_corr(composite, fwd_20d)

    print(f"\n  Composite LEVEL vs 20d returns:        {level_corr:+.3f}" if level_corr else "  Composite LEVEL: N/A")
    print(f"  Best composite ROC vs 20d returns:     {best_roc_corr:+.3f} ({best_roc_label})" if best_roc_corr else "  Best ROC: N/A")

    if level_corr and best_roc_corr:
        if best_roc_corr > level_corr:
            improvement = best_roc_corr - level_corr
            print(f"\n  → Momentum IMPROVES prediction by {improvement:+.3f}")
            print(f"  → RECOMMENDATION: BUILD the momentum layer")
        elif best_roc_corr > 0:
            print(f"\n  → Momentum has POSITIVE correlation (level has negative)")
            print(f"  → RECOMMENDATION: BUILD the momentum layer")
        else:
            print(f"\n  → Momentum does NOT improve prediction")
            print(f"  → RECOMMENDATION: Investigate other approaches")

    # Category ROC comparison
    print(f"\n  Category 4-week ROC correlations with 20d returns:")
    for cat_name, cat_data in [("Growth", growth), ("Monetary", monetary),
                                ("Inflation", inflation), ("Market", market)]:
        roc = np.full(n, np.nan)
        for i in range(4, n):
            roc[i] = cat_data[i] - cat_data[i - 4]
        c = safe_corr(roc, fwd_20d)
        marker = " ← STRONGEST" if c and abs(c) > 0.1 else ""
        print(f"    {cat_name:12s}: {c:+.3f}{marker}" if c else f"    {cat_name:12s}: N/A")

    print(f"\n{'=' * 70}")
    print("  ANALYSIS COMPLETE")
    print(f"{'=' * 70}\n")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())