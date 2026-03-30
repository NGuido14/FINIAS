"""
FINIAS Sector-Macro Sensitivity Analysis

Using the 196 backtest observations + 11 sector ETF returns,
compute which sectors benefit from which macro conditions.

This answers: "Given the current macro environment, which sectors
should outperform and which should underperform?"

Run: python finias/backtesting/sector_sensitivity.py
"""

import asyncio
import numpy as np
from datetime import date, timedelta
from collections import defaultdict
from finias.core.database.connection import DatabasePool
from finias.core.database.migrations import run_migrations

SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
SECTOR_NAMES = {
    "XLB": "Materials", "XLC": "Comm Svcs", "XLE": "Energy",
    "XLF": "Financials", "XLI": "Industrials", "XLK": "Technology",
    "XLP": "Staples", "XLRE": "Real Estate", "XLU": "Utilities",
    "XLV": "Healthcare", "XLY": "Cons Disc",
}


async def main():
    db = DatabasePool()
    await db.initialize()
    await run_migrations(db)

    # Load backtest results
    bt_rows = await db.fetch("""
        SELECT sim_date, composite_score, growth_score, monetary_score,
               inflation_score, market_score, stress_index, binding_constraint
        FROM backtest_results
        WHERE warmup = FALSE AND composite_score IS NOT NULL
        ORDER BY sim_date ASC
    """)

    # Load sector ETF prices
    sector_data = {}
    for symbol in SECTOR_ETFS:
        rows = await db.fetch("""
            SELECT trade_date, close FROM market_data_daily
            WHERE symbol = $1 ORDER BY trade_date ASC
        """, symbol)
        sector_data[symbol] = {str(r["trade_date"]): float(r["close"]) for r in rows}

    # Load SPY for relative returns
    spy_rows = await db.fetch("""
        SELECT trade_date, close FROM market_data_daily
        WHERE symbol = 'SPY' ORDER BY trade_date ASC
    """)
    spy_data = {str(r["trade_date"]): float(r["close"]) for r in spy_rows}

    # For each backtest date, compute 20-day forward RELATIVE returns for each sector
    # (sector return - SPY return = relative outperformance)
    dates = [r["sim_date"] for r in bt_rows]
    composite = [float(r["composite_score"]) for r in bt_rows]
    growth = [float(r["growth_score"]) for r in bt_rows]
    monetary = [float(r["monetary_score"]) for r in bt_rows]
    inflation = [float(r["inflation_score"]) for r in bt_rows]
    market = [float(r["market_score"]) for r in bt_rows]
    stress = [float(r["stress_index"]) for r in bt_rows]
    binding = [r["binding_constraint"] for r in bt_rows]

    def get_forward_return(price_dict, sim_date, days=20):
        """Get forward return from sim_date."""
        sim_str = str(sim_date)
        # Find closest date on or after sim_date
        sorted_dates = sorted(price_dict.keys())
        start_idx = None
        for i, d in enumerate(sorted_dates):
            if d >= sim_str:
                start_idx = i
                break
        if start_idx is None or start_idx + days >= len(sorted_dates):
            return None
        start_price = price_dict[sorted_dates[start_idx]]
        end_price = price_dict[sorted_dates[start_idx + days]]
        return (end_price / start_price - 1) * 100

    # Build sector relative return matrix
    n = len(dates)
    sector_rel_returns = {sym: [] for sym in SECTOR_ETFS}

    for i in range(n):
        spy_ret = get_forward_return(spy_data, dates[i], 20)
        for sym in SECTOR_ETFS:
            sec_ret = get_forward_return(sector_data[sym], dates[i], 20)
            if spy_ret is not None and sec_ret is not None:
                sector_rel_returns[sym].append(sec_ret - spy_ret)
            else:
                sector_rel_returns[sym].append(np.nan)

    # Convert to numpy
    for sym in SECTOR_ETFS:
        sector_rel_returns[sym] = np.array(sector_rel_returns[sym])

    def safe_corr(a, b):
        a, b = np.array(a), np.array(b)
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 15:
            return None
        c = np.corrcoef(a[mask], b[mask])[0, 1]
        return float(c) if not np.isnan(c) else None

    print("\n" + "=" * 80)
    print("  FINIAS SECTOR-MACRO SENSITIVITY ANALYSIS")
    print("  Which sectors benefit from which macro conditions?")
    print("=" * 80)
    print(f"  Observations: {n} weeks")
    print(f"  Period: {dates[0]} to {dates[-1]}")
    print(f"  Returns: 20-day forward RELATIVE to SPY (sector minus SPY)")

    # ================================================================
    # 1. SECTOR CORRELATION WITH EACH MACRO CATEGORY SCORE
    # ================================================================
    print(f"\n{'=' * 80}")
    print("  1. SECTOR SENSITIVITY TO MACRO CATEGORY SCORES (level)")
    print(f"     Correlation of category LEVEL with 20d forward sector relative return")
    print(f"{'=' * 80}")

    categories = {
        "Growth": growth, "Monetary": monetary,
        "Inflation": inflation, "Market": market,
        "Stress": stress, "Composite": composite
    }

    # Print header
    print(f"\n  {'Sector':12s}", end="")
    for cat_name in categories:
        print(f"  {cat_name:>10s}", end="")
    print()
    print("  " + "-" * 78)

    for sym in SECTOR_ETFS:
        name = SECTOR_NAMES[sym]
        print(f"  {name:12s}", end="")
        for cat_name, cat_data in categories.items():
            corr = safe_corr(cat_data, sector_rel_returns[sym])
            if corr is not None:
                # Highlight strong correlations
                marker = "*" if abs(corr) > 0.15 else " "
                print(f"  {corr:+.3f}{marker}   ", end="")
            else:
                print(f"  {'N/A':>10s}", end="")
        print()

    print(f"\n  * = |correlation| > 0.15")

    # ================================================================
    # 2. SECTOR SENSITIVITY TO MACRO CATEGORY CHANGES (momentum)
    # ================================================================
    print(f"\n{'=' * 80}")
    print("  2. SECTOR SENSITIVITY TO MACRO CATEGORY CHANGES (4-week)")
    print(f"     Correlation of category CHANGE with 20d forward sector relative return")
    print(f"{'=' * 80}")

    cat_changes = {}
    for cat_name, cat_data in categories.items():
        roc = [np.nan] * 4 + [cat_data[i] - cat_data[i - 4] for i in range(4, n)]
        cat_changes[cat_name] = roc

    print(f"\n  {'Sector':12s}", end="")
    for cat_name in cat_changes:
        print(f"  {cat_name:>10s}", end="")
    print()
    print("  " + "-" * 78)

    for sym in SECTOR_ETFS:
        name = SECTOR_NAMES[sym]
        print(f"  {name:12s}", end="")
        for cat_name, cat_roc in cat_changes.items():
            corr = safe_corr(cat_roc, sector_rel_returns[sym])
            if corr is not None:
                marker = "*" if abs(corr) > 0.15 else " "
                print(f"  {corr:+.3f}{marker}   ", end="")
            else:
                print(f"  {'N/A':>10s}", end="")
        print()

    print(f"\n  * = |correlation| > 0.15")

    # ================================================================
    # 3. SECTOR PERFORMANCE BY BINDING CONSTRAINT
    # ================================================================
    print(f"\n{'=' * 80}")
    print("  3. SECTOR RELATIVE PERFORMANCE BY BINDING CONSTRAINT")
    print(f"     Avg 20d relative return (vs SPY) when each factor is binding")
    print(f"{'=' * 80}")

    binding_set = sorted(set(b for b in binding if b is not None))

    print(f"\n  {'Sector':12s}", end="")
    for bc in binding_set:
        short_name = bc.replace("_", " ")[:15]
        print(f"  {short_name:>15s}", end="")
    print()
    print("  " + "-" * (14 + 17 * len(binding_set)))

    for sym in SECTOR_ETFS:
        name = SECTOR_NAMES[sym]
        print(f"  {name:12s}", end="")
        for bc in binding_set:
            bc_returns = [sector_rel_returns[sym][i] for i in range(n)
                          if binding[i] == bc and not np.isnan(sector_rel_returns[sym][i])]
            if bc_returns:
                avg = np.mean(bc_returns)
                marker = "+" if avg > 0.3 else ("-" if avg < -0.3 else " ")
                print(f"  {avg:+.2f}%{marker}        ", end="")
            else:
                print(f"  {'N/A':>15s}", end="")
        print()

    print(f"\n  + = outperforms by >0.3%, - = underperforms by >0.3%")

    # ================================================================
    # 4. SECTOR PERFORMANCE BY INFLATION TRAJECTORY
    # ================================================================
    print(f"\n{'=' * 80}")
    print("  4. SECTOR RELATIVE PERFORMANCE BY INFLATION TRAJECTORY")
    print(f"     Avg 20d relative return when inflation is easing vs tightening")
    print(f"{'=' * 80}")

    infl_4w_change = [np.nan] * 4 + [inflation[i] - inflation[i - 4] for i in range(4, n)]

    print(f"\n  {'Sector':12s}  {'Easing':>12s}  {'Stable':>12s}  {'Tightening':>12s}  {'Spread':>12s}")
    print("  " + "-" * 64)

    spreads = {}
    for sym in SECTOR_ETFS:
        name = SECTOR_NAMES[sym]
        easing_rets = [sector_rel_returns[sym][i] for i in range(n)
                       if not np.isnan(infl_4w_change[i]) and infl_4w_change[i] > 0.02
                       and not np.isnan(sector_rel_returns[sym][i])]
        tight_rets = [sector_rel_returns[sym][i] for i in range(n)
                      if not np.isnan(infl_4w_change[i]) and infl_4w_change[i] < -0.02
                      and not np.isnan(sector_rel_returns[sym][i])]
        stable_rets = [sector_rel_returns[sym][i] for i in range(n)
                       if not np.isnan(infl_4w_change[i]) and abs(infl_4w_change[i]) <= 0.02
                       and not np.isnan(sector_rel_returns[sym][i])]

        e_avg = np.mean(easing_rets) if easing_rets else np.nan
        t_avg = np.mean(tight_rets) if tight_rets else np.nan
        s_avg = np.mean(stable_rets) if stable_rets else np.nan
        spread = e_avg - t_avg if not np.isnan(e_avg) and not np.isnan(t_avg) else np.nan
        spreads[sym] = spread

        e_str = f"{e_avg:+.2f}%" if not np.isnan(e_avg) else "N/A"
        t_str = f"{t_avg:+.2f}%" if not np.isnan(t_avg) else "N/A"
        s_str = f"{s_avg:+.2f}%" if not np.isnan(s_avg) else "N/A"
        sp_str = f"{spread:+.2f}pp" if not np.isnan(spread) else "N/A"
        print(f"  {name:12s}  {e_str:>12s}  {s_str:>12s}  {t_str:>12s}  {sp_str:>12s}")

    # Sort by spread
    print(f"\n  Sectors most helped by inflation easing (largest spread):")
    sorted_spreads = sorted(spreads.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -99, reverse=True)
    for i, (sym, sp) in enumerate(sorted_spreads[:3]):
        if not np.isnan(sp):
            print(f"    {i + 1}. {SECTOR_NAMES[sym]:12s}: {sp:+.2f}pp spread")

    print(f"\n  Sectors most hurt by inflation easing (smallest/negative spread):")
    for i, (sym, sp) in enumerate(reversed(sorted_spreads[-3:])):
        if not np.isnan(sp):
            print(f"    {i + 1}. {SECTOR_NAMES[sym]:12s}: {sp:+.2f}pp spread")

    # ================================================================
    # 5. SECTOR PERFORMANCE BY STRESS LEVEL
    # ================================================================
    print(f"\n{'=' * 80}")
    print("  5. SECTOR RELATIVE PERFORMANCE BY STRESS LEVEL")
    print(f"     Which sectors outperform when stress rises?")
    print(f"{'=' * 80}")

    stress_arr = np.array(stress)
    high_stress = stress_arr > np.percentile(stress_arr, 75)
    low_stress = stress_arr < np.percentile(stress_arr, 25)

    print(f"\n  {'Sector':12s}  {'High Stress':>12s}  {'Low Stress':>12s}  {'Spread':>12s}")
    print("  " + "-" * 52)

    for sym in SECTOR_ETFS:
        name = SECTOR_NAMES[sym]
        hi_rets = [sector_rel_returns[sym][i] for i in range(n)
                   if high_stress[i] and not np.isnan(sector_rel_returns[sym][i])]
        lo_rets = [sector_rel_returns[sym][i] for i in range(n)
                   if low_stress[i] and not np.isnan(sector_rel_returns[sym][i])]

        hi_avg = np.mean(hi_rets) if hi_rets else np.nan
        lo_avg = np.mean(lo_rets) if lo_rets else np.nan
        spread = hi_avg - lo_avg if not np.isnan(hi_avg) and not np.isnan(lo_avg) else np.nan

        hi_str = f"{hi_avg:+.2f}%" if not np.isnan(hi_avg) else "N/A"
        lo_str = f"{lo_avg:+.2f}%" if not np.isnan(lo_avg) else "N/A"
        sp_str = f"{spread:+.2f}pp" if not np.isnan(spread) else "N/A"
        print(f"  {name:12s}  {hi_str:>12s}  {lo_str:>12s}  {sp_str:>12s}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 80}")
    print("  SUMMARY — MACRO-SECTOR SENSITIVITY MAP")
    print(f"{'=' * 80}")

    print("""
  This data should be encoded into the Macro Strategist so it can tell
  downstream agents: "Given current macro conditions (inflation binding,
  stress elevated, growth decelerating), the following sectors should
  be over/underweighted..."

  The relationships above are EMPIRICAL — computed from actual sector
  returns during actual macro regimes, not textbook assumptions.
    """)

    print(f"{'=' * 80}")
    print("  ANALYSIS COMPLETE")
    print(f"{'=' * 80}\n")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())