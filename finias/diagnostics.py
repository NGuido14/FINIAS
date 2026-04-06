#!/usr/bin/env python3
"""
FINIAS Data Diagnostics Script

Tests data population, table usage, and computation pipeline WITHOUT
calling the Claude API. Zero cost to run.

Usage:
    python -m finias.diagnostics

What it checks:
    1. Database connectivity and table existence
    2. FRED data population — are all ~60 series being fetched?
    3. Polygon data population — are market symbols being fetched?
    4. Macro matrix population — is the pivot table working?
    5. Net liquidity computation — is date alignment working?
    6. All computation modules — do they produce valid output from real data?
    7. Table redundancy — which tables have data, which are empty?
"""

import asyncio
import json
import sys
import os
from datetime import date, timedelta, datetime, timezone
from typing import Any

# Ensure project is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finias.core.config.settings import get_settings
from finias.core.database.connection import DatabasePool
from finias.core.database.migrations import run_migrations
from finias.core.state.redis_state import RedisState
from finias.data.providers.polygon_client import PolygonClient
from finias.data.providers.fred_client import FredClient, MACRO_SERIES
from finias.data.cache.market_cache import MarketDataCache
from finias.data.cache.matrix_mapping import SERIES_TO_COLUMN


# ============================================================
# Formatting helpers
# ============================================================

def header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def ok(msg: str):
    print(f"  ✓ {msg}")


def warn(msg: str):
    print(f"  ⚠ {msg}")


def fail(msg: str):
    print(f"  ✗ {msg}")


def info(msg: str):
    print(f"    {msg}")


# ============================================================
# 1. Database & Table Check
# ============================================================

async def check_tables(db: DatabasePool):
    header("1. DATABASE TABLES")

    # Get all tables
    rows = await db.fetch("""
        SELECT tablename FROM pg_tables 
        WHERE schemaname = 'public' 
        ORDER BY tablename
    """)
    tables = [r["tablename"] for r in rows]

    expected = [
        "market_data_daily", "economic_indicators", "agent_opinions",
        "schema_migrations", "agent_health_log",
        "regime_assessments",
        "macro_data_matrix",
    ]

    for table in expected:
        if table in tables:
            # Get row count
            count = await db.fetchval(f"SELECT COUNT(*) FROM {table}")
            if count > 0:
                ok(f"{table}: {count:,} rows")
            else:
                warn(f"{table}: EMPTY (0 rows)")
        else:
            fail(f"{table}: TABLE MISSING")

    # Flag redundancies
    print()
    info("REDUNDANCY CHECK:")

    regime_new = await db.fetchval("SELECT COUNT(*) FROM regime_assessments") if "regime_assessments" in tables else 0

    if regime_new == 0:
        warn("regime_assessments is EMPTY — agent doesn't write to it yet")


# ============================================================
# 2. FRED Data Check
# ============================================================

async def check_fred_data(db: DatabasePool):
    header("2. FRED DATA POPULATION")

    total_series = len(MACRO_SERIES)

    rows = await db.fetch("""
        SELECT series_id, COUNT(*) as obs_count, 
               MIN(obs_date) as earliest, MAX(obs_date) as latest
        FROM economic_indicators
        GROUP BY series_id
        ORDER BY series_id
    """)

    populated = {r["series_id"]: r for r in rows}

    missing = []
    stale = []
    healthy = []
    today = date.today()

    for series_id, description in MACRO_SERIES.items():
        if series_id not in populated:
            missing.append(series_id)
        else:
            r = populated[series_id]
            days_old = (today - r["latest"]).days

            # Daily series should be <5 days old, weekly <10, monthly <35
            if days_old > 35:
                stale.append((series_id, days_old, r["obs_count"]))
            else:
                healthy.append((series_id, r["obs_count"], r["latest"]))

    ok(f"Total FRED series defined: {total_series}")
    ok(f"Series with data: {len(populated)}")

    if missing:
        fail(f"Series with NO DATA ({len(missing)}):")
        for s in missing:
            info(f"  {s}: {MACRO_SERIES[s]}")

    if stale:
        warn(f"Series with STALE data ({len(stale)}):")
        for s, days, count in stale:
            info(f"  {s}: {count} obs, latest {days} days old")

    ok(f"Series with fresh data: {len(healthy)}")

    # Check key monetary policy series specifically
    print()
    info("KEY MONETARY POLICY SERIES:")
    key_monetary = ["WALCL", "WTREGEN", "RRPONTSYD", "WRESBAL", "NFCI", "M2SL"]
    for sid in key_monetary:
        if sid in populated:
            r = populated[sid]
            ok(f"  {sid}: {r['obs_count']} obs, latest={r['latest']}")
        else:
            fail(f"  {sid}: NO DATA — this is why liquidity shows 'unavailable'")


# ============================================================
# 3. Polygon Data Check
# ============================================================

async def check_polygon_data(db: DatabasePool):
    header("3. POLYGON MARKET DATA")

    rows = await db.fetch("""
        SELECT symbol, COUNT(*) as bar_count,
               MIN(trade_date) as earliest, MAX(trade_date) as latest
        FROM market_data_daily
        GROUP BY symbol
        ORDER BY symbol
    """)

    if not rows:
        fail("No market data in database")
        return

    for r in rows:
        days_old = (date.today() - r["latest"]).days
        status = "✓" if days_old < 5 else "⚠"
        print(f"  {status} {r['symbol']}: {r['bar_count']} bars, "
              f"{r['earliest']} → {r['latest']} ({days_old}d old)")

    # Check for expected symbols
    expected = ["SPY", "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLC", "XLY", "XLRE", "XLB"]
    present = {r["symbol"] for r in rows}
    missing = [s for s in expected if s not in present]
    if missing:
        warn(f"Missing expected symbols: {', '.join(missing)}")


# ============================================================
# 4. Macro Matrix Check
# ============================================================

async def check_matrix(db: DatabasePool):
    header("4. MACRO DATA MATRIX")

    count = await db.fetchval("SELECT COUNT(*) FROM macro_data_matrix")

    if count == 0:
        fail("macro_data_matrix is EMPTY — populate_macro_matrix() hasn't run")
        return

    ok(f"Matrix has {count} date rows")

    # Check column population rates
    columns_to_check = [
        ("vix", "VIX"), ("dgs10", "10Y Treasury"), ("fedfunds", "Fed Funds"),
        ("fed_total_assets", "Fed Total Assets"), ("tga_balance", "TGA"),
        ("reverse_repo", "Reverse Repo"), ("net_liquidity", "Net Liquidity"),
        ("nfci", "NFCI"), ("unemployment", "Unemployment"),
        ("cpi_core", "Core CPI"), ("core_pce", "Core PCE"),
        ("lei", "LEI"), ("initial_claims", "Initial Claims"),
        ("consumer_sentiment", "Consumer Sentiment"),
        ("m2", "M2 Money Supply"), ("dxy", "Dollar Index"),
    ]

    print()
    info("COLUMN FILL RATES:")
    for col, label in columns_to_check:
        try:
            filled = await db.fetchval(
                f"SELECT COUNT(*) FROM macro_data_matrix WHERE {col} IS NOT NULL"
            )
            pct = (filled / count * 100) if count > 0 else 0
            status = "✓" if pct > 10 else ("⚠" if pct > 0 else "✗")
            print(f"  {status} {label} ({col}): {filled}/{count} rows ({pct:.0f}%)")
        except Exception as e:
            fail(f"  {label} ({col}): ERROR — {e}")

    # Check net liquidity specifically
    print()
    info("NET LIQUIDITY COMPUTATION:")
    nl_rows = await db.fetch("""
        SELECT obs_date, fed_total_assets, tga_balance, reverse_repo, net_liquidity
        FROM macro_data_matrix
        WHERE net_liquidity IS NOT NULL
        ORDER BY obs_date DESC
        LIMIT 5
    """)
    if nl_rows:
        ok(f"Net liquidity computed for {len(nl_rows)}+ dates")
        for r in nl_rows:
            info(f"  {r['obs_date']}: Assets={r['fed_total_assets']:.0f} "
                 f"- TGA={r['tga_balance']:.0f} - RRP={r['reverse_repo']:.0f} "
                 f"= NetLiq={r['net_liquidity']:.0f}")
    else:
        fail("Net liquidity NOT computed — check components")
        # Diagnose why
        for col in ["fed_total_assets", "tga_balance", "reverse_repo"]:
            c = await db.fetchval(
                f"SELECT COUNT(*) FROM macro_data_matrix WHERE {col} IS NOT NULL"
            )
            info(f"  {col}: {c} non-null rows")


# ============================================================
# 5. Computation Module Tests (no Claude API needed)
# ============================================================

async def check_computations(cache: MarketDataCache):
    header("5. COMPUTATION MODULES (pure Python, no Claude API)")

    # Force matrix repopulation to ensure computed values are current
    try:
        repop_count = await cache.populate_macro_matrix()
        ok(f"Matrix repopulated: {repop_count} dates refreshed")
    except Exception as e:
        warn(f"Matrix repopulation failed: {e}")

    from_date = date.today() - timedelta(days=730)
    to_date = date.today()

    # Fetch data through cache (will use DB, no API calls if cached)
    # Initialize all computation results to None so later tests
    # don't crash if an earlier module fails
    yc = None
    vol = None
    mp = None
    cycle = None
    infl = None
    spx_prices = []

    # Pre-fetch sector data for breadth and correlation tests
    sector_etfs_diag = ["XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLC", "XLY", "XLRE", "XLB"]
    sector_prices = {}
    for etf in sector_etfs_diag:
        try:
            bars = await cache.get_daily_bars(etf, from_date, to_date)
            sector_prices[etf] = [{"date": str(b["trade_date"]), "close": float(b["close"])} for b in bars]
        except:
            pass

    try:
        # Yield Curve
        info("Testing yield curve computation...")
        yc_series = {}
        for sid in ["DGS2", "DGS5", "DGS10", "DGS30", "DTB3", "FEDFUNDS", "DFII5", "DFII10", "THREEFYTP10"]:
            yc_series[sid] = await cache.get_fred_series(sid, from_date=from_date)

        from finias.agents.macro_strategist.computations.yield_curve import analyze_yield_curve
        yc = analyze_yield_curve(
            yields_2y=yc_series["DGS2"], yields_5y=yc_series["DGS5"],
            yields_10y=yc_series["DGS10"], yields_30y=yc_series["DGS30"],
            yields_3m=yc_series["DTB3"], fed_funds=yc_series["FEDFUNDS"],
            real_yields_5y=yc_series.get("DFII5", []),
            real_yields_10y=yc_series.get("DFII10", []),
            term_premium_10y=yc_series.get("THREEFYTP10", []),
        )
        ok(f"Yield Curve: 2s10s={yc.spread_2s10s}, shape={yc.curve_shape}, "
           f"recession_score={yc.recession_signal_score:.2f}")
        if yc.forward_1y1y is not None:
            ok(f"  Forward 1Y1Y={yc.forward_1y1y:.2f}%, implied policy change={yc.implied_policy_change_1y:.0f}bp")
        if yc.forward_2y3y is not None:
            ok(f"  Forward 2Y3Y={yc.forward_2y3y:.2f}%")
    except Exception as e:
        fail(f"Yield Curve: {e}")

    try:
        # Volatility
        info("Testing volatility computation...")
        vix = await cache.get_fred_series("VIXCLS", from_date=from_date)
        vix3m = await cache.get_fred_series("VXVCLS", from_date=from_date)
        spx_bars = await cache.get_daily_bars("SPY", from_date, to_date)
        spx_prices = [{"date": str(b["trade_date"]), "close": float(b["close"])} for b in spx_bars]

        from finias.agents.macro_strategist.computations.volatility import analyze_volatility
        skew = await cache.get_fred_series("SKEW", from_date=from_date)
        vol = analyze_volatility(vix_series=vix, spx_prices=spx_prices, vix3m_series=vix3m, skew_series=skew)
        ok(f"Volatility: VIX={vol.vix_current}, regime={vol.vol_regime}, "
           f"risk_score={vol.vol_risk_score:.2f}")
        if vol.vix3m_current is not None:
            ok(f"  VIX3M={vol.vix3m_current:.2f}, ratio={vol.vix_vix3m_ratio:.3f}, "
               f"term_structure={vol.term_structure_shape}")
        if vol.variance_risk_premium is not None:
            ok(f"  VRP={vol.variance_risk_premium:.2f}, vrp_regime={vol.vrp_regime}")
            skew = vol.skew_current
            skew_regime = vol.skew_regime
            if skew is not None:
                ok(f"  SKEW={skew:.0f}, skew_regime={skew_regime}")
    except Exception as e:
        fail(f"Volatility: {e}")

    try:
        # Monetary Policy
        info("Testing monetary policy computation...")
        mp_series = {}
        for sid in ["FEDFUNDS", "DFEDTARU", "DFEDTARL", "WALCL", "TREAST",
                    "WSHOMCB", "RRPONTSYD", "WTREGEN", "WRESBAL",
                    "NFCI", "STLFSI4", "TOTBKCR", "TOTALSL", "M2SL"]:
            mp_series[sid] = await cache.get_fred_series(sid, from_date=from_date)

        from finias.agents.macro_strategist.computations.monetary_policy import analyze_monetary_policy
        mp = analyze_monetary_policy(
            fed_funds=mp_series["FEDFUNDS"],
            fed_target_upper=mp_series["DFEDTARU"],
            fed_target_lower=mp_series["DFEDTARL"],
            fed_total_assets=mp_series["WALCL"],
            fed_treasuries=mp_series["TREAST"],
            fed_mbs=mp_series["WSHOMCB"],
            tga=mp_series["WTREGEN"],
            reverse_repo=mp_series["RRPONTSYD"],
            bank_reserves=mp_series["WRESBAL"],
            nfci_series=mp_series["NFCI"],
            stress_series=mp_series["STLFSI4"],
            bank_credit=mp_series["TOTBKCR"],
            consumer_credit=mp_series["TOTALSL"],
            m2_series=mp_series["M2SL"],
        )
        ok(f"Monetary Policy: fed_funds={mp.fed_funds_current}, "
           f"net_liq={mp.net_liquidity}, stance={mp.policy_stance}, "
           f"liq_regime={mp.liquidity_regime}")

        if mp.net_liquidity is None:
            fail("NET LIQUIDITY IS NONE — date alignment still broken")
            info(f"  WALCL data points: {len(mp_series['WALCL'])}")
            info(f"  WTREGEN data points: {len(mp_series['WTREGEN'])}")
            info(f"  RRPONTSYD data points: {len(mp_series['RRPONTSYD'])}")
            if mp_series['WALCL']:
                info(f"  WALCL latest date: {mp_series['WALCL'][-1]['date']}")
            if mp_series['WTREGEN']:
                info(f"  WTREGEN latest date: {mp_series['WTREGEN'][-1]['date']}")
            if mp_series['RRPONTSYD']:
                info(f"  RRPONTSYD latest date: {mp_series['RRPONTSYD'][-1]['date']}")
        else:
            ok(f"  Net Liquidity: {mp.net_liquidity:,.0f}")
            ok(f"  TGA: {mp.tga_level:,.0f}" if mp.tga_level else "  TGA: None")
            ok(f"  Reverse Repo: {mp.reverse_repo_level:,.0f}" if mp.reverse_repo_level else "  RRP: None")
            ok(f"  NFCI: {mp.nfci}" if mp.nfci else "  NFCI: None")
    except Exception as e:
        fail(f"Monetary Policy: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Business Cycle
        info("Testing business cycle computation...")
        bc_series = {}
        for sid in ["UNRATE", "ICSA", "CCSA", "JTSJOL", "JTSQUR",
                    "TEMPHELPS", "AWHAETP", "PERMIT", "HOUST", "RSAFS",
                    "UMCSENT", "INDPRO", "TCU", "CFNAI", "PI", "DGORDER",
                    "PAYEMS", "GACDFSA066MSFRBPHI", "GDPNOW"]:
            bc_series[sid] = await cache.get_fred_series(sid, from_date=from_date)

        from finias.agents.macro_strategist.computations.business_cycle import analyze_business_cycle
        cycle = analyze_business_cycle(
            lei_series=[],  # Conference Board LEI removed from FRED
            unemployment=bc_series["UNRATE"],
            initial_claims=bc_series["ICSA"],
            continuing_claims=bc_series["CCSA"],
            jolts_openings=bc_series["JTSJOL"],
            jolts_quits=bc_series["JTSQUR"],
            temp_employment=bc_series["TEMPHELPS"],
            avg_weekly_hours=bc_series["AWHAETP"],
            building_permits=bc_series["PERMIT"],
            housing_starts=bc_series["HOUST"],
            retail_sales=bc_series["RSAFS"],
            consumer_sentiment=bc_series["UMCSENT"],
            industrial_production=bc_series["INDPRO"],
            capacity_utilization=bc_series["TCU"],
            cfnai_series=bc_series["CFNAI"],
            personal_income=bc_series["PI"],
            durable_goods=bc_series["DGORDER"],
            nfp_series=bc_series["PAYEMS"],
            philly_fed=bc_series["GACDFSA066MSFRBPHI"],
            gdp_nowcast_series=bc_series.get("GDPNOW", []),
            yield_curve_slope=yc.spread_3m10y if yc else None,
        )
        ok(f"Business Cycle: phase={cycle.cycle_phase}, confidence={cycle.phase_confidence:.2f}, "
           f"sahm={cycle.sahm_value:.3f}, triggered={cycle.sahm_triggered}, "
           f"recession_prob={cycle.recession_probability:.2f}")
        if cycle.gdp_nowcast is not None:
            ok(f"  GDPNow={cycle.gdp_nowcast:.1f}% (trend={cycle.gdp_nowcast_trend})")
    except Exception as e:
        fail(f"Business Cycle: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Inflation
        info("Testing inflation computation...")
        inf_series = {}
        for sid in ["CPIAUCSL", "CPILFESL", "CUSR0000SEHC", "CUSR0000SAS",
                    "PCEPI", "PCEPILFE", "STICKCPIM159SFRBATL", "FLEXCPIM159SFRBATL",
                    "PCETRIM12M159SFRBDAL", "T5YIE", "T10YIE", "T5YIFR",
                    "PPIACO", "CES0500000003", "DCOILWTICO"]:
            inf_series[sid] = await cache.get_fred_series(sid, from_date=from_date)

        from finias.agents.macro_strategist.computations.inflation import analyze_inflation
        infl = analyze_inflation(
            cpi_all=inf_series["CPIAUCSL"],
            cpi_core=inf_series["CPILFESL"],
            cpi_shelter=inf_series["CUSR0000SEHC"],
            cpi_services=inf_series["CUSR0000SAS"],
            pce=inf_series["PCEPI"],
            core_pce=inf_series["PCEPILFE"],
            sticky_cpi=inf_series["STICKCPIM159SFRBATL"],
            flexible_cpi=inf_series["FLEXCPIM159SFRBATL"],
            trimmed_mean=inf_series["PCETRIM12M159SFRBDAL"],
            breakeven_5y=inf_series["T5YIE"],
            breakeven_10y=inf_series["T10YIE"],
            forward_5y5y=inf_series["T5YIFR"],
            ppi=inf_series["PPIACO"],
            ahe=inf_series["CES0500000003"],
            oil=inf_series["DCOILWTICO"],
        )
        ok(f"Inflation: core_pce_yoy={infl.core_pce_yoy}, core_cpi_3m={infl.core_cpi_3m_annualized}, "
           f"regime={infl.inflation_regime}, trend={infl.inflation_trend}, "
           f"spiral_risk={infl.spiral_risk:.2f}")
    except Exception as e:
        fail(f"Inflation: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Breadth
        info("Testing breadth computation...")
        rsp_bars = await cache.get_daily_bars("RSP", from_date, to_date)
        rsp_p = [{"date": str(b["trade_date"]), "close": float(b["close"])} for b in rsp_bars] if rsp_bars else None

        from finias.agents.macro_strategist.computations.breadth import analyze_breadth
        breadth_test = analyze_breadth(
            spx_prices=spx_prices,
            sector_prices=sector_prices,
            rsp_prices=rsp_p,
        )
        ok(f"Breadth: health={breadth_test.breadth_health}, score={breadth_test.breadth_score:.2f}, "
           f"sectors_above_200ma={breadth_test.sectors_above_200ma}/11, "
           f"narrow_leadership={breadth_test.narrow_leadership}, "
           f"rotation={breadth_test.rotation_signal}")
        if breadth_test.leading_sectors:
            ok(f"  Leading: {', '.join(breadth_test.leading_sectors)}")
            ok(f"  Lagging: {', '.join(breadth_test.lagging_sectors)}")
        if breadth_test.breadth_divergence:
            warn(f"  DIVERGENCE: {breadth_test.divergence_description}")
    except Exception as e:
        fail(f"Breadth: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Cross-Asset (expanded)
        info("Testing expanded cross-asset computation...")
        from finias.agents.macro_strategist.computations.cross_asset import analyze_cross_assets

        # Fetch additional symbols
        add_syms = {}
        for sym in ["GLD", "CPER", "IWM", "TLT", "HYG", "EEM"]:
            try:
                bars = await cache.get_daily_bars(sym, from_date, to_date)
                add_syms[sym] = [{"date": str(b["trade_date"]), "close": float(b["close"])} for b in bars]
            except:
                add_syms[sym] = None

        ca_test = analyze_cross_assets(
            dxy_series=await cache.get_fred_series("DTWEXBGS", from_date=from_date),
            hy_spread_series=await cache.get_fred_series("BAMLH0A0HYM2", from_date=from_date),
            breakeven_5y=await cache.get_fred_series("T5YIE", from_date=from_date),
            breakeven_10y=await cache.get_fred_series("T10YIE", from_date=from_date),
            copper_prices=add_syms.get("CPER"),
            gold_prices=add_syms.get("GLD"),
            oil_series=await cache.get_fred_series("DCOILWTICO", from_date=from_date),
            brent_series=await cache.get_fred_series("DCOILBRENTEU", from_date=from_date),
            spy_prices=spx_prices,
            tlt_prices=add_syms.get("TLT"),
            iwm_prices=add_syms.get("IWM"),
            hyg_prices=add_syms.get("HYG"),
            eem_prices=add_syms.get("EEM"),
            vix_series=await cache.get_fred_series("VIXCLS", from_date=from_date),
        )
        ok(f"Cross-Asset: score={ca_test.cross_asset_score:.3f}, "
           f"cu/au_signal={ca_test.copper_gold_signal}, "
           f"oil_signal={ca_test.oil_signal}, "
           f"risk_appetite={ca_test.risk_appetite}, "
           f"stock_bond_corr={ca_test.stock_bond_corr_20d}")
        if ca_test.risk_parity_stress:
            warn(f"  RISK PARITY STRESS: stocks and bonds falling together")
        if ca_test.credit_equity_divergence:
            warn(f"  CREDIT-EQUITY DIVERGENCE: {ca_test.divergence_type}")
        if ca_test.em_stress:
            warn(f"  EM STRESS: relative perf {ca_test.em_relative_performance_20d:.1f}%")
        # Correlation matrix diagnostics
        if ca_test.correlation_matrix:
            corr = ca_test.correlation_matrix
            agg = corr.get("aggregate", {})
            ok(f"  Correlation Matrix: diversification={agg.get('diversification_regime')}, "
               f"avg|corr|={agg.get('avg_absolute_correlation', 0):.3f}, "
               f"stress_couplings={agg.get('stress_coupling_count', 0)}, "
               f"breakdowns={agg.get('breakdown_count', 0)}")
            for pair_name, pair_data in corr.get("pairs", {}).items():
                rc = pair_data.get("rolling_correlations", {})
                ok(f"    {pair_name}: corr_60d={rc.get('corr_60d', 'N/A')}, "
                   f"regime={pair_data.get('regime_label', 'unknown')}")
        else:
            warn("  Correlation Matrix: not computed")
    except Exception as e:
        fail(f"Cross-Asset: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Full Regime Detection
        info("Testing full regime detection...")

        if any(x is None for x in [yc, vol]):
            fail("Regime Detection: SKIPPED — yield curve or volatility module failed")
        else:
            from finias.agents.macro_strategist.computations.breadth import analyze_breadth
            from finias.agents.macro_strategist.computations.cross_asset import analyze_cross_assets
            from finias.agents.macro_strategist.computations.regime import detect_regime

            # Fetch RSP for breadth
            rsp_bars = await cache.get_daily_bars("RSP", from_date, to_date)
            rsp_prices = [{"date": str(b["trade_date"]), "close": float(b["close"])} for b in rsp_bars] if rsp_bars else None

            breadth = analyze_breadth(
                spx_prices=spx_prices,
                sector_prices=sector_prices if 'sector_prices' in dir() else {},
                rsp_prices=rsp_prices,
            )

            # Fetch additional symbols for cross-asset
            diag_syms = {}
            for sym in ["GLD", "CPER", "IWM", "TLT", "HYG", "EEM"]:
                try:
                    bars = await cache.get_daily_bars(sym, from_date, to_date)
                    diag_syms[sym] = [{"date": str(b["trade_date"]), "close": float(b["close"])} for b in bars]
                except:
                    diag_syms[sym] = None

            ca = analyze_cross_assets(
                dxy_series=await cache.get_fred_series("DTWEXBGS", from_date=from_date),
                hy_spread_series=await cache.get_fred_series("BAMLH0A0HYM2", from_date=from_date),
                breakeven_5y=await cache.get_fred_series("T5YIE", from_date=from_date),
                breakeven_10y=await cache.get_fred_series("T10YIE", from_date=from_date),
                copper_prices=diag_syms.get("CPER"),
                gold_prices=diag_syms.get("GLD"),
                oil_series=await cache.get_fred_series("DCOILWTICO", from_date=from_date),
                brent_series=await cache.get_fred_series("DCOILBRENTEU", from_date=from_date),
                spy_prices=spx_prices,
                tlt_prices=diag_syms.get("TLT"),
                iwm_prices=diag_syms.get("IWM"),
                hyg_prices=diag_syms.get("HYG"),
                eem_prices=diag_syms.get("EEM"),
                vix_series=await cache.get_fred_series("VIXCLS", from_date=from_date),
            )

            regime = detect_regime(
                yield_curve=yc, volatility=vol, breadth=breadth, cross_asset=ca,
                monetary_policy=mp, business_cycle=cycle, inflation_analysis=infl,
            )

            ok(f"REGIME ASSESSMENT:")
            ok(f"  Primary: {regime.primary_regime.value} (confidence={regime.confidence:.2f})")
            ok(f"  Cycle: {regime.cycle_phase}")
            ok(f"  Liquidity: {regime.liquidity_regime}")
            ok(f"  Volatility: {regime.volatility_regime}")
            ok(f"  Inflation: {regime.inflation_regime}")
            ok(f"  Composite: {regime.composite_score:.3f}")
            ok(f"  Stress: {regime.stress_index:.3f}")
            ok(f"  Binding: {regime.binding_constraint}")
            ok(f"  Scores: growth={regime.growth_cycle_score:.3f}, "
               f"monetary={regime.monetary_liquidity_score:.3f}, "
               f"inflation={regime.inflation_score:.3f}, "
               f"market={regime.market_signals_score:.3f}")
            # Downstream context
            ctx = regime.to_downstream_context()
            ok(f"  DOWNSTREAM CONTEXT:")
            ok(f"    Sector: favor_cyclicals={ctx.favor_cyclicals}, favor_defensives={ctx.favor_defensives}, favor_duration={ctx.favor_duration}")
            ok(f"    Rates: environment={ctx.rate_environment}, implied_direction={ctx.implied_rate_direction}, change={ctx.implied_policy_change_bp:.0f}bp")
            ok(f"    Liquidity: supportive={ctx.liquidity_supportive}, net_liq={ctx.net_liquidity_trillion:.2f}T")
            ok(f"    Vol: persistent={ctx.vol_persistent}, vrp={ctx.vrp_regime}")
            ok(f"    Risk: credit_stress={ctx.credit_stress}, rp_stress={ctx.risk_parity_stress}, recession_prob={ctx.recession_probability:.0%}")
            if ctx.gdp_nowcast is not None:
                ok(f"    GDPNow: {ctx.gdp_nowcast:.1f}%")
            if ctx.consistency_warnings:
                for w in ctx.consistency_warnings:
                    warn(f"    CONSISTENCY: {w}")
            else:
                ok(f"    Consistency: ALL CHECKS PASSED")

            # Trajectory assessment
            from finias.agents.macro_strategist.computations.trajectory import compute_trajectory

            # Get DFEDTARU for rate history
            dfedtaru = await cache.get_fred_series("DFEDTARU", from_date=from_date)

            # Fetch most recent stored regime for trajectory comparison
            prior_regime = None
            try:
                prior_row = await cache.db.fetchrow(
                    """
                    SELECT inflation_score, stress_index, binding_constraint
                    FROM regime_assessments
                    ORDER BY id DESC LIMIT 1
                    """
                )
                if prior_row:
                    class _PriorRegime:
                        pass
                    prior_regime = _PriorRegime()
                    prior_regime.inflation_score = float(prior_row["inflation_score"]) if prior_row["inflation_score"] else 0.0
                    prior_regime.stress_index = float(prior_row["stress_index"]) if prior_row["stress_index"] else 0.0
                    prior_regime.binding_constraint = prior_row["binding_constraint"] or "none"
                    ok(f"  (Using prior regime from regime_assessments for trajectory comparison)")
                else:
                    ok(f"  (No prior regime found — trajectory signals will show defaults)")
            except Exception as e:
                logger.warning(f"Could not fetch prior regime: {e}")

            traj = compute_trajectory(
                regime_assessment=regime,
                fed_target_upper=dfedtaru,
                prior_regime_assessment=prior_regime,
            )
            ok(f"  TRAJECTORY ASSESSMENT:")
            ok(f"    Rate history: {traj.policy_trajectory} ({traj.cumulative_rate_change_12m_bp:+.0f}bp in 12m, "
               f"{len(traj.rate_decisions_12m)} decisions)")
            ok(f"    Inflation surprise: {traj.inflation_surprise_direction} ({traj.inflation_surprise_pp:+.2f}pp)")
            ok(f"    Inflation trajectory: {traj.inflation_trajectory}")
            ok(f"    Stress contrarian: {traj.stress_contrarian_signal}")
            ok(f"    Binding shift: {traj.binding_shift_direction}")
            ok(f"    Forward bias: {traj.forward_bias} (score={traj.forward_bias_score:+.3f}, "
               f"confidence={traj.forward_bias_confidence})")
            if traj.sector_overweights:
                ok(f"    Sector OW: {', '.join(traj.sector_overweights)}")
                ok(f"    Sector UW: {', '.join(traj.sector_underweights)}")
            # Position sizing and velocity
            if hasattr(traj, 'max_single_position_pct'):
                ok(f"    Position: max {traj.max_single_position_pct}% per stock, "
                   f"beta target {traj.portfolio_beta_target}, cash {traj.cash_target_pct}%"
                   f"{' [REDUCE EXPOSURE]' if traj.reduce_overall_exposure else ''}")
            if hasattr(traj, 'vix_velocity'):
                ok(f"    Velocity: VIX={traj.vix_velocity}, spreads={traj.spread_velocity}, "
                   f"breadth={traj.breadth_velocity}, urgency={traj.urgency}")
            if hasattr(traj, 'upcoming_events') and traj.upcoming_events:
                next_evt = traj.upcoming_events[0]
                ok(f"    Next event: {next_evt['event']} in {next_evt['days_away']}d "
                   f"(sizing mult: {traj.pre_event_sizing_multiplier}x)")
            if hasattr(traj, 'scenario_triggers'):
                critical = [t for t in traj.scenario_triggers if t.get("severity") == "critical"]
                for t in critical:
                    warn(f"    TRIGGER: {t['metric']} at {t['current']}, "
                         f"threshold {t['threshold']} (distance: {t['distance']})")
    except Exception as e:
        fail(f"Regime Detection: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# 6. Table Usage Summary
# ============================================================

async def check_ground_truth(db: DatabasePool, fred: FredClient):
    header("7. GROUND-TRUTH VALIDATION")
    try:
        from finias.validation.ground_truth import run_all_validations
        await run_all_validations(db, fred)
    except Exception as e:
        print(f"  ⚠ Ground-truth validation failed: {e}")


async def check_interpretation_validation(db: DatabasePool):
    # === Section 8: Interpretation Validation History ===
    print("\n" + "=" * 60)
    print("  8. INTERPRETATION VALIDATION HISTORY (last 5)")
    print("=" * 60)
    try:
        val_rows = await db.fetch(
            """
            SELECT id, assessed_at, interpretation_json
            FROM regime_assessments
            WHERE interpretation_json IS NOT NULL
            ORDER BY id DESC LIMIT 5
            """
        )

        total_corrections = 0
        total_warnings = 0
        total_passed = 0

        for row in val_rows:
            try:
                interp = json.loads(row["interpretation_json"]) if isinstance(row["interpretation_json"], str) else row["interpretation_json"]
                val = interp.get("_validation", {})
                if val:
                    corr = val.get("corrected", 0)
                    warn = val.get("warned", 0)
                    passed = val.get("passed", 0)
                    total = val.get("total_checks", 0)
                    total_corrections += corr
                    total_warnings += warn
                    total_passed += passed

                    status = "✓" if corr == 0 and warn == 0 else "⚠" if corr == 0 else "✗"
                    print(f"  {status} Row {row['id']} ({row['assessed_at']}): "
                          f"{passed} passed, {corr} corrected, {warn} warned / {total} checks")

                    for c in val.get("corrections", []):
                        print(f"      CORRECTED: {c['field']}: '{c.get('claude_value', '')}' → '{c.get('computed_value', '')}'")
                    for w in val.get("warnings", []):
                        print(f"      WARNING: {w['field']}: {w.get('note', '')}")
                else:
                    print(f"  - Row {row['id']}: No validation data (pre-validation assessment)")
            except (json.JSONDecodeError, TypeError, KeyError):
                print(f"  - Row {row['id']}: Could not parse interpretation")

        if total_corrections + total_warnings + total_passed > 0:
            total_all = total_corrections + total_warnings + total_passed
            print(f"\n  Summary: {total_passed}/{total_all} passed, "
                  f"{total_corrections} corrections, {total_warnings} warnings")
            if total_corrections > 0:
                print(f"  ⚠ Claude required {total_corrections} corrections across {len(val_rows)} assessments")
    except Exception as e:
        print(f"  ⚠ Could not fetch validation history: {e}")


async def table_usage_summary(db: DatabasePool):
    header("6. TABLE USAGE SUMMARY & RECOMMENDATIONS")

    tables = {
        "market_data_daily": "KEEP — Polygon price data, actively used",
        "economic_indicators": "KEEP — Raw FRED landing zone, feeds matrix",
        "macro_data_matrix": "KEEP — Pivoted view for efficient queries",
        "agent_opinions": "KEEP — Audit trail for all agent outputs",
        "schema_migrations": "KEEP — Migration tracking",
        "agent_health_log": "KEEP — Health monitoring",
        "regime_assessments": "KEEP — Historical regime classifications for trajectory comparison",
    }

    for table, recommendation in tables.items():
        count = await db.fetchval(f"SELECT COUNT(*) FROM {table}")
        status = "ACTIVE" if count > 0 else "EMPTY"
        print(f"  [{status:6s}] {table}: {recommendation}")


async def check_positioning(db: DatabasePool):
    header("9. CFTC Positioning")
    try:
        from finias.data.providers.cot_client import get_latest_cot, get_cot_staleness_days, COT_CONTRACTS
        from finias.agents.macro_strategist.computations.positioning import compute_contract_positioning
        from finias.data.providers.cot_client import get_cot_history

        staleness = await get_cot_staleness_days(db)
        latest = await get_latest_cot(db)

        if not latest:
            warn("No COT positioning data in database. Run morning refresh to fetch.")
            return

        total_row = await db.fetchrow("SELECT COUNT(*) as cnt FROM cot_positioning")
        total = total_row["cnt"] if total_row else 0
        ok(f"COT data: {total} records, {staleness} days since latest report")

        for contract_key in COT_CONTRACTS:
            if contract_key in latest:
                row = latest[contract_key]
                history = await get_cot_history(db, contract_key, lookback_weeks=156)
                cp = compute_contract_positioning(contract_key, history)
                crowding_label = f" ★{cp.crowding.upper()}" if cp.crowding != "neutral" else ""
                ok(f"  {contract_key}: net={cp.net_spec:+,}, "
                   f"percentile={cp.net_spec_percentile:.1f}, "
                   f"crowding={cp.crowding}{crowding_label}, "
                   f"roc_4w={cp.rate_of_change_4w:+,}, "
                   f"weeks={cp.lookback_weeks}, confidence={cp.confidence}")
            else:
                warn(f"  {contract_key}: NO DATA")

        if staleness > 14:
            warn(f"  COT data is {staleness} days stale — CFTC may have a publication gap")
        elif staleness > 10:
            warn(f"  COT data is {staleness} days stale — slightly older than normal")
        else:
            ok(f"  Staleness: {staleness} days (normal)")

    except Exception as e:
        fail(f"Positioning check failed: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# Main
# ============================================================

async def main():
    print("\n" + "=" * 60)
    print("  FINIAS DATA DIAGNOSTICS")
    print("  Zero-cost verification (no Claude API calls)")
    print("=" * 60)

    try:
        # Initialize
        db = DatabasePool()
        await db.initialize()
        await run_migrations(db)

        state = RedisState()
        await state.initialize()

        polygon = PolygonClient()
        fred = FredClient()
        cache = MarketDataCache(db=db, state=state, polygon=polygon, fred=fred)

        # Run all checks
        await check_tables(db)
        await check_fred_data(db)
        await check_polygon_data(db)
        await check_matrix(db)
        await check_computations(cache)
        await check_ground_truth(db, fred)
        await check_positioning(db)
        await check_interpretation_validation(db)
        await table_usage_summary(db)

        # Cleanup
        await polygon.close()
        await fred.close()
        await state.close()
        await db.close()

        print(f"\n{'=' * 60}")
        print("  DIAGNOSTICS COMPLETE")
        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"\n  FATAL: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())