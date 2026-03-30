"""
Backtest validation report generator.

Reads results from the database and prints a comprehensive
statistical validation report.
"""

from __future__ import annotations
import logging
from finias.core.database.connection import DatabasePool
from finias.backtesting.metrics import correlation, directional_hit_rate, strategy_performance

logger = logging.getLogger("finias.backtesting.report")


async def generate_report(db: DatabasePool, run_id: str) -> None:
    """Generate and print a full validation report for a backtest run."""

    # Load scored (non-warmup) results
    rows = await db.fetch(
        """
        SELECT * FROM backtest_results
        WHERE backtest_run_id = $1 AND warmup = FALSE
        ORDER BY sim_date ASC
        """,
        run_id
    )

    if not rows:
        print("No scored results found for this backtest run.")
        return

    # Extract arrays
    dates = [r["sim_date"] for r in rows]
    composite = [float(r["composite_score"]) if r["composite_score"] is not None else float('nan') for r in rows]
    growth = [float(r["growth_score"]) if r["growth_score"] is not None else float('nan') for r in rows]
    monetary = [float(r["monetary_score"]) if r["monetary_score"] is not None else float('nan') for r in rows]
    inflation = [float(r["inflation_score"]) if r["inflation_score"] is not None else float('nan') for r in rows]
    market = [float(r["market_score"]) if r["market_score"] is not None else float('nan') for r in rows]
    stress = [float(r["stress_index"]) if r["stress_index"] is not None else float('nan') for r in rows]
    regimes = [r["primary_regime"] for r in rows]
    binding = [r["binding_constraint"] for r in rows]

    fwd_5d = [float(r["spx_fwd_5d"]) if r["spx_fwd_5d"] is not None else float('nan') for r in rows]
    fwd_20d = [float(r["spx_fwd_20d"]) if r["spx_fwd_20d"] is not None else float('nan') for r in rows]
    fwd_60d = [float(r["spx_fwd_60d"]) if r["spx_fwd_60d"] is not None else float('nan') for r in rows]
    max_dd = [float(r["spx_max_dd_20d"]) if r["spx_max_dd_20d"] is not None else float('nan') for r in rows]

    print("\n" + "=" * 70)
    print("  FINIAS MACRO STRATEGIST — BACKTEST VALIDATION REPORT")
    print("=" * 70)
    print(f"  Run ID: {run_id[:8]}...")
    print(f"  Period: {dates[0]} to {dates[-1]}")
    print(f"  Scored observations: {len(rows)}")
    print()

    # --- 1. SIGNAL QUALITY: Correlation ---
    print("-" * 70)
    print("  1. SIGNAL QUALITY — Correlation with Forward SPX Returns")
    print("-" * 70)

    for horizon_name, fwd_data in [("5-day", fwd_5d), ("20-day", fwd_20d), ("60-day", fwd_60d)]:
        print(f"\n  {horizon_name} forward returns:")
        for score_name, score_data in [
            ("Composite", composite), ("Growth", growth),
            ("Monetary", monetary), ("Inflation", inflation), ("Market", market)
        ]:
            corr = correlation(score_data, fwd_data)
            corr_str = f"{corr:+.3f}" if corr is not None else "N/A"
            print(f"    {score_name:12s} correlation: {corr_str}")

    # --- 2. DIRECTIONAL HIT RATE ---
    print(f"\n{'-' * 70}")
    print("  2. DIRECTIONAL HIT RATE — Composite Score vs Returns")
    print("-" * 70)

    for horizon_name, fwd_data in [("20-day", fwd_20d), ("60-day", fwd_60d)]:
        print(f"\n  {horizon_name} forward returns (threshold = ±0.15):")
        hits = directional_hit_rate(composite, fwd_data, signal_threshold=0.15)

        if hits.get("bullish_count", 0) > 0:
            print(f"    Bullish (composite > 0.15): {hits['bullish_count']} obs, "
                  f"avg return {hits['bullish_avg_return']:+.2f}%, "
                  f"hit rate {hits['bullish_hit_rate']:.0%}")
        else:
            print(f"    Bullish: No observations")

        if hits.get("bearish_count", 0) > 0:
            print(f"    Bearish (composite < -0.15): {hits['bearish_count']} obs, "
                  f"avg return {hits['bearish_avg_return']:+.2f}%, "
                  f"hit rate {hits['bearish_hit_rate']:.0%}")
        else:
            print(f"    Bearish: No observations")

        if hits.get("neutral_count", 0) > 0:
            print(f"    Neutral: {hits['neutral_count']} obs, "
                  f"avg return {hits['neutral_avg_return']:+.2f}%")

    # --- 3. REGIME PERFORMANCE ---
    print(f"\n{'-' * 70}")
    print("  3. REGIME-SPECIFIC PERFORMANCE — 20-day Forward Returns by Regime")
    print("-" * 70)

    import numpy as np
    regime_set = sorted(set(r for r in regimes if r is not None))
    for regime in regime_set:
        regime_returns = [fwd_20d[i] for i in range(len(regimes))
                         if regimes[i] == regime and not np.isnan(fwd_20d[i])]
        if regime_returns:
            avg = np.mean(regime_returns)
            std = np.std(regime_returns)
            count = len(regime_returns)
            print(f"    {regime:15s}: avg {avg:+.2f}%, std {std:.2f}%, "
                  f"count {count}, hit rate {np.mean(np.array(regime_returns) > 0):.0%}")

    # --- 4. STRESS INDEX VALIDATION ---
    print(f"\n{'-' * 70}")
    print("  4. STRESS INDEX VALIDATION — Does High Stress Predict Drawdowns?")
    print("-" * 70)

    stress_arr = np.array(stress)
    dd_arr = np.array(max_dd)
    mask = ~(np.isnan(stress_arr) | np.isnan(dd_arr))

    if mask.sum() > 10:
        high_stress = stress_arr[mask] > 0.4
        low_stress = stress_arr[mask] <= 0.4

        if high_stress.sum() > 0:
            print(f"    High stress (>0.4): avg max DD {np.mean(dd_arr[mask][high_stress]):.2f}%, "
                  f"count {high_stress.sum()}")
        if low_stress.sum() > 0:
            print(f"    Low stress (<=0.4): avg max DD {np.mean(dd_arr[mask][low_stress]):.2f}%, "
                  f"count {low_stress.sum()}")

        corr_stress = correlation(list(stress_arr[mask]), list(dd_arr[mask]))
        if corr_stress is not None:
            print(f"    Stress-Drawdown correlation: {corr_stress:+.3f}")

    # --- 5. BINDING CONSTRAINT ACCURACY ---
    print(f"\n{'-' * 70}")
    print("  5. BINDING CONSTRAINT FREQUENCY")
    print("-" * 70)

    from collections import Counter
    bc_counts = Counter(b for b in binding if b is not None)
    for bc, count in bc_counts.most_common():
        pct = count / len(binding) * 100
        bc_returns = [fwd_20d[i] for i in range(len(binding))
                      if binding[i] == bc and not np.isnan(fwd_20d[i])]
        avg_ret = np.mean(bc_returns) if bc_returns else float('nan')
        print(f"    {bc:25s}: {count:3d} obs ({pct:.0f}%), avg 20d return: {avg_ret:+.2f}%")

    # --- 6. STRATEGY PERFORMANCE ---
    print(f"\n{'-' * 70}")
    print("  6. SIMPLE STRATEGY — Long/Cash Based on Composite Score")
    print("-" * 70)

    # Use weekly returns (5-day forward)
    perf = strategy_performance(composite, fwd_5d)

    if "error" not in perf:
        print(f"\n    {'':30s} {'Strategy':>12s} {'Buy & Hold':>12s}")
        print(f"    {'Total Return':30s} {perf['strategy_total_return']:>11.1%} {perf['buyhold_total_return']:>11.1%}")
        print(f"    {'Annual Return':30s} {perf['strategy_annual_return']:>11.1%} {perf['buyhold_annual_return']:>11.1%}")
        print(f"    {'Sharpe Ratio':30s} {perf['strategy_sharpe']:>11.2f} {perf['buyhold_sharpe']:>11.2f}")
        print(f"    {'Max Drawdown':30s} {perf['strategy_max_drawdown']:>11.1%} {perf['buyhold_max_drawdown']:>11.1%}")
        print(f"    {'Volatility':30s} {perf['strategy_volatility']:>11.1%} {perf['buyhold_volatility']:>11.1%}")
        print(f"    {'Total Position Changes':30s} {perf['total_trades']:>11d}")
        print(f"    {'Time Fully Invested':30s} {perf['pct_time_fully_invested']:>11.0%}")
        print(f"    {'Time in Cash':30s} {perf['pct_time_cash']:>11.0%}")
    else:
        print(f"    {perf['error']}")

    # --- 7. REGIME TRANSITIONS ---
    print(f"\n{'-' * 70}")
    print("  7. REGIME TRANSITIONS")
    print("-" * 70)

    transitions = 0
    transition_returns = []
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i-1] and regimes[i] is not None and regimes[i-1] is not None:
            transitions += 1
            if not np.isnan(fwd_20d[i]):
                transition_returns.append(fwd_20d[i])

    print(f"    Total regime transitions: {transitions}")
    if transition_returns:
        print(f"    Avg 20d return after transition: {np.mean(transition_returns):+.2f}%")
        print(f"    Std: {np.std(transition_returns):.2f}%")

    print(f"\n{'=' * 70}")
    print("  REPORT COMPLETE")
    print(f"{'=' * 70}\n")
