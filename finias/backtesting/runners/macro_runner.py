"""
Macro Strategist Backtest Runner

Entry point for validating the Macro Strategist's predictive power.

Usage:
    python -m finias.backtesting.runners.macro_runner

This runner:
1. Ensures historical data is available (5yr Polygon, 10yr FRED)
2. Steps through each week from start to end date
3. Filters data with publication lag to prevent look-ahead bias
4. Calls the EXACT same computation modules as the live agent
5. Records regime scores and forward SPX returns
6. Generates a validation report
"""

from __future__ import annotations
from datetime import date, timedelta
import asyncio
import logging
import sys

from finias.core.database.connection import DatabasePool
from finias.core.database.migrations import run_migrations
from finias.core.state.redis_state import RedisState
from finias.data.providers.fred_client import FredClient, MACRO_SERIES
from finias.data.providers.polygon_client import PolygonClient

from finias.backtesting.data_loader import (
    ensure_historical_data,
    filter_series_as_of, filter_bars_as_of,
    SERIES_LAG_CATEGORY,
)
from finias.backtesting.engine import run_walk_forward
from finias.backtesting.report import generate_report

# Import the EXACT same computation modules as the live agent
from finias.agents.macro_strategist.computations.yield_curve import analyze_yield_curve
from finias.agents.macro_strategist.computations.volatility import (
    analyze_volatility, compute_sector_correlation, classify_correlation_regime
)
from finias.agents.macro_strategist.computations.monetary_policy import analyze_monetary_policy
from finias.agents.macro_strategist.computations.business_cycle import analyze_business_cycle
from finias.agents.macro_strategist.computations.inflation import analyze_inflation
from finias.agents.macro_strategist.computations.breadth import analyze_breadth
from finias.agents.macro_strategist.computations.cross_asset import analyze_cross_assets
from finias.agents.macro_strategist.computations.regime import detect_regime

logger = logging.getLogger("finias.backtesting.macro_runner")

# Sector ETF symbols for breadth analysis
SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]

# All Polygon symbols needed
ALL_POLYGON_SYMBOLS = ["SPY", "RSP", "IWM", "TLT", "GLD", "HYG", "CPER", "EEM"] + SECTOR_ETFS


async def main():
    """Run the full Macro Strategist backtest."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("\n" + "=" * 70)
    print("  FINIAS MACRO STRATEGIST BACKTEST")
    print("  Walk-Forward Validation")
    print("=" * 70 + "\n")

    # Initialize infrastructure
    db = DatabasePool()
    await db.initialize()
    await run_migrations(db)

    state = RedisState()
    await state.initialize()

    fred = FredClient()
    polygon = PolygonClient()

    try:
        # Step 1: Ensure historical data
        print("Step 1: Ensuring historical data (5yr Polygon, 10yr FRED)...")
        print("  This may take several minutes on first run.\n")
        data_status = await ensure_historical_data(db, fred, polygon)

        fred_ok = sum(1 for v in data_status["fred"].values() if v.get("ok"))
        poly_ok = sum(1 for v in data_status["polygon"].values() if v.get("ok"))
        print(f"  FRED: {fred_ok}/{len(data_status['fred'])} series available")
        print(f"  Polygon: {poly_ok}/{len(data_status['polygon'])} symbols available\n")

        # Step 2: Load all historical data into memory
        print("Step 2: Loading all historical data into memory...")

        all_fred = {}
        for series_id in MACRO_SERIES:
            rows = await db.fetch(
                """
                SELECT obs_date, value FROM economic_indicators
                WHERE series_id = $1 ORDER BY obs_date ASC
                """,
                series_id
            )
            all_fred[series_id] = [
                {"date": str(r["obs_date"]), "value": float(r["value"])}
                for r in rows
            ]

        all_polygon = {}
        for symbol in ALL_POLYGON_SYMBOLS:
            rows = await db.fetch(
                """
                SELECT trade_date, open, high, low, close, volume
                FROM market_data_daily
                WHERE symbol = $1 ORDER BY trade_date ASC
                """,
                symbol
            )
            all_polygon[symbol] = [
                {"date": str(r["trade_date"]), "close": float(r["close"])}
                for r in rows
            ]

        print(f"  Loaded {len(all_fred)} FRED series, {len(all_polygon)} Polygon symbols\n")

        # Step 3: Define the runner function
        async def compute_regime_for_date(sim_date: date) -> dict:
            """Run the full macro computation pipeline for a given date."""

            # Filter FRED data with publication lag
            fred_as_of = {}
            for sid, series in all_fred.items():
                lag_cat = SERIES_LAG_CATEGORY.get(sid, "monthly")
                fred_as_of[sid] = filter_series_as_of(series, sim_date, lag_cat)

            # Filter Polygon data (1-day lag)
            polygon_as_of = {}
            for sym, bars in all_polygon.items():
                polygon_as_of[sym] = filter_bars_as_of(bars, sim_date)

            spx_prices = polygon_as_of.get("SPY", [])
            if len(spx_prices) < 201:
                return None  # Not enough history for MAs

            # Build sector prices dict
            sector_prices = {}
            for etf in SECTOR_ETFS:
                if etf in polygon_as_of and len(polygon_as_of[etf]) >= 201:
                    sector_prices[etf] = polygon_as_of[etf]

            try:
                # 1. Yield Curve
                yc = analyze_yield_curve(
                    yields_2y=fred_as_of.get("DGS2", []),
                    yields_5y=fred_as_of.get("DGS5", []),
                    yields_10y=fred_as_of.get("DGS10", []),
                    yields_30y=fred_as_of.get("DGS30", []),
                    yields_3m=fred_as_of.get("DTB3", []),
                    fed_funds=fred_as_of.get("FEDFUNDS", []),
                    real_yields_5y=fred_as_of.get("DFII5", []),
                    real_yields_10y=fred_as_of.get("DFII10", []),
                    term_premium_10y=fred_as_of.get("THREEFYTP10", []),
                )

                # 2. Volatility
                vol = analyze_volatility(
                    vix_series=fred_as_of.get("VIXCLS", []),
                    spx_prices=spx_prices,
                    vix3m_series=fred_as_of.get("VXVCLS", []),
                )
                if len(sector_prices) >= 5:
                    avg_corr = compute_sector_correlation(sector_prices)
                    vol.sector_correlation = avg_corr
                    vol.correlation_regime = classify_correlation_regime(avg_corr)

                # 3. Monetary Policy
                mp = analyze_monetary_policy(
                    fed_funds=fred_as_of.get("FEDFUNDS", []),
                    fed_target_upper=fred_as_of.get("DFEDTARU", []),
                    fed_target_lower=fred_as_of.get("DFEDTARL", []),
                    fed_total_assets=fred_as_of.get("WALCL", []),
                    fed_treasuries=fred_as_of.get("TREAST", []),
                    fed_mbs=fred_as_of.get("WSHOMCB", []),
                    tga=fred_as_of.get("WTREGEN", []),
                    reverse_repo=fred_as_of.get("RRPONTSYD", []),
                    bank_reserves=fred_as_of.get("WRESBAL", []),
                    nfci_series=fred_as_of.get("NFCI", []),
                    stress_series=fred_as_of.get("STLFSI4", []),
                    bank_credit=fred_as_of.get("TOTBKCR", []),
                    consumer_credit=fred_as_of.get("TOTALSL", []),
                    m2_series=fred_as_of.get("M2SL", []),
                )

                # 4. Business Cycle
                bc = analyze_business_cycle(
                    lei_series=[],
                    unemployment=fred_as_of.get("UNRATE", []),
                    initial_claims=fred_as_of.get("ICSA", []),
                    continuing_claims=fred_as_of.get("CCSA", []),
                    jolts_openings=fred_as_of.get("JTSJOL", []),
                    jolts_quits=fred_as_of.get("JTSQUR", []),
                    temp_employment=fred_as_of.get("TEMPHELPS", []),
                    avg_weekly_hours=fred_as_of.get("AWHAETP", []),
                    building_permits=fred_as_of.get("PERMIT", []),
                    housing_starts=fred_as_of.get("HOUST", []),
                    retail_sales=fred_as_of.get("RSAFS", []),
                    consumer_sentiment=fred_as_of.get("UMCSENT", []),
                    industrial_production=fred_as_of.get("INDPRO", []),
                    capacity_utilization=fred_as_of.get("TCU", []),
                    cfnai_series=fred_as_of.get("CFNAI", []),
                    personal_income=fred_as_of.get("PI", []),
                    durable_goods=fred_as_of.get("DGORDER", []),
                    nfp_series=fred_as_of.get("PAYEMS", []),
                    philly_fed=fred_as_of.get("GACDFSA066MSFRBPHI", []),
                    gdp_nowcast_series=fred_as_of.get("GDPNOW", []),
                )

                # 5. Inflation
                infl = analyze_inflation(
                    cpi_all=fred_as_of.get("CPIAUCSL", []),
                    cpi_core=fred_as_of.get("CPILFESL", []),
                    cpi_shelter=fred_as_of.get("CUSR0000SEHC", []),
                    cpi_services=fred_as_of.get("CUSR0000SAS", []),
                    pce=fred_as_of.get("PCEPI", []),
                    core_pce=fred_as_of.get("PCEPILFE", []),
                    sticky_cpi=fred_as_of.get("STICKCPIM159SFRBATL", []),
                    flexible_cpi=fred_as_of.get("FLEXCPIM159SFRBATL", []),
                    trimmed_mean=fred_as_of.get("PCETRIM12M159SFRBDAL", []),
                    breakeven_5y=fred_as_of.get("T5YIE", []),
                    breakeven_10y=fred_as_of.get("T10YIE", []),
                    forward_5y5y=fred_as_of.get("T5YIFR", []),
                    ppi=fred_as_of.get("PPIACO", []),
                    ahe=fred_as_of.get("CES0500000003", []),
                    oil=fred_as_of.get("DCOILWTICO", []),
                )

                # 6. Breadth
                breadth = analyze_breadth(
                    spx_prices=spx_prices,
                    sector_prices=sector_prices,
                    rsp_prices=polygon_as_of.get("RSP"),
                )

                # 7. Cross-Asset
                ca = analyze_cross_assets(
                    dxy_series=fred_as_of.get("DTWEXBGS", []),
                    hy_spread_series=fred_as_of.get("BAMLH0A0HYM2", []),
                    breakeven_5y=fred_as_of.get("T5YIE", []),
                    breakeven_10y=fred_as_of.get("T10YIE", []),
                    copper_prices=polygon_as_of.get("CPER"),
                    gold_prices=polygon_as_of.get("GLD"),
                    oil_series=fred_as_of.get("DCOILWTICO", []),
                    spy_prices=spx_prices,
                    tlt_prices=polygon_as_of.get("TLT"),
                    iwm_prices=polygon_as_of.get("IWM"),
                    hyg_prices=polygon_as_of.get("HYG"),
                    eem_prices=polygon_as_of.get("EEM"),
                )

                # 8. Regime Detection
                regime = detect_regime(
                    yield_curve=yc,
                    volatility=vol,
                    breadth=breadth,
                    cross_asset=ca,
                    monetary_policy=mp,
                    business_cycle=bc,
                    inflation_analysis=infl,
                )

                return {
                    "composite_score": regime.composite_score,
                    "growth_score": regime.growth_cycle_score,
                    "monetary_score": regime.monetary_liquidity_score,
                    "inflation_score": regime.inflation_score,
                    "market_score": regime.market_signals_score,
                    "primary_regime": regime.primary_regime.value,
                    "cycle_phase": regime.cycle_phase,
                    "stress_index": regime.stress_index,
                    "confidence": regime.confidence,
                    "binding_constraint": regime.binding_constraint,
                    "modules_used": "full",
                }

            except Exception as e:
                logger.warning(f"Computation failed for {sim_date}: {e}")
                return None

        # Step 4: Define forward return function
        async def get_forward_returns(sim_date: date) -> dict:
            """Get actual SPX returns after sim_date."""
            spy = all_polygon.get("SPY", [])
            if not spy:
                return {}

            # Find the index of sim_date in SPY data
            spy_dates = [s["date"] for s in spy]
            spy_closes = [s["close"] for s in spy]

            # Find closest date on or after sim_date
            sim_str = sim_date.isoformat()
            start_idx = None
            for i, d in enumerate(spy_dates):
                if d >= sim_str:
                    start_idx = i
                    break

            if start_idx is None:
                return {}

            base_price = spy_closes[start_idx]
            result = {}

            # 5-day forward return
            if start_idx + 5 < len(spy_closes):
                result["5d"] = (spy_closes[start_idx + 5] / base_price - 1) * 100

            # 20-day forward return
            if start_idx + 20 < len(spy_closes):
                result["20d"] = (spy_closes[start_idx + 20] / base_price - 1) * 100

            # 60-day forward return
            if start_idx + 60 < len(spy_closes):
                result["60d"] = (spy_closes[start_idx + 60] / base_price - 1) * 100

            # Max drawdown over next 20 days
            if start_idx + 20 < len(spy_closes):
                future_20 = spy_closes[start_idx:start_idx + 21]
                peak = base_price
                max_dd = 0
                for price in future_20:
                    peak = max(peak, price)
                    dd = (price / peak - 1) * 100
                    max_dd = min(max_dd, dd)
                result["max_dd_20d"] = max_dd

            return result

        # Step 5: Determine backtest window
        # Full pipeline needs 1yr warmup + enough Polygon data
        # Use earliest available SPY data + 1yr warmup
        spy_data = all_polygon.get("SPY", [])
        if not spy_data:
            print("ERROR: No SPY data available. Cannot run backtest.")
            return

        earliest_spy = date.fromisoformat(spy_data[0]["date"])
        latest_spy = date.fromisoformat(spy_data[-1]["date"])

        # Start 1 year after earliest data (warmup period)
        # But we need ~60 trading days of future data for forward returns
        start = earliest_spy
        end = latest_spy - timedelta(days=90)  # Leave room for 60-day forward returns

        print(f"\nStep 3: Running walk-forward backtest...")
        print(f"  Data range: {earliest_spy} to {latest_spy}")
        print(f"  Backtest range: {start} to {end}")
        print(f"  Warmup: 52 weeks from {start}")
        print(f"  Step: weekly (every 7 days)")
        print()

        # Step 6: Run the backtest
        run_id = await run_walk_forward(
            db=db,
            runner_fn=compute_regime_for_date,
            start_date=start,
            end_date=end,
            step_days=7,
            warmup_weeks=52,
            forward_return_fn=get_forward_returns,
        )

        # Step 7: Generate report
        print("\nStep 4: Generating validation report...")
        await generate_report(db, run_id)

    finally:
        await fred.close()
        await polygon.close()
        await state.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
