"""
Macro Strategist Agent

Layer 1 Domain Expert: Macro-economic analysis and regime assessment.

This agent:
  1. Fetches macro data from FRED and Polygon (via cache)
  2. Runs pure Python computations (yield curve, vol, breadth, cross-asset)
  3. Synthesizes into a regime assessment
  4. Asks Claude to interpret the composite picture in context
  5. Returns a structured AgentOpinion

The Python does the math. Claude explains what the math means.
"""

from __future__ import annotations
from typing import Any, Optional
from datetime import date, datetime, timedelta, timezone
import json
import logging
import re

import numpy as np
import anthropic

from finias.core.agents.base import BaseAgent
from finias.core.agents.models import (
    AgentOpinion, AgentQuery, AgentLayer,
    ConfidenceLevel, SignalDirection, MarketRegime, HealthStatus
)
from finias.core.config.settings import get_settings
from finias.core.state.redis_state import RedisState
from finias.data.cache.market_cache import MarketDataCache

# Computations
from finias.agents.macro_strategist.computations.yield_curve import (
    analyze_yield_curve, YieldCurveAnalysis
)
from finias.agents.macro_strategist.computations.volatility import (
    analyze_volatility, VolatilityAnalysis
)
from finias.agents.macro_strategist.computations.breadth import (
    analyze_breadth, BreadthAnalysis
)
from finias.agents.macro_strategist.computations.cross_asset import (
    analyze_cross_assets, CrossAssetAnalysis
)
from finias.agents.macro_strategist.computations.regime import detect_regime
from finias.agents.macro_strategist.computations.monetary_policy import (
    analyze_monetary_policy, MonetaryPolicyAnalysis
)
from finias.agents.macro_strategist.computations.business_cycle import (
    analyze_business_cycle, BusinessCycleAnalysis
)
from finias.agents.macro_strategist.computations.inflation import (
    analyze_inflation, InflationAnalysis
)
from finias.agents.macro_strategist.computations.trajectory import compute_trajectory
from finias.agents.macro_strategist.prompts.interpretation import MACRO_ANALYSIS_PROMPT

logger = logging.getLogger("finias.agent.macro_strategist")


def _generate_correlation_notes_from_dict(corr_dict: dict) -> list[str]:
    """
    Generate plain-English correlation data notes from the stored dict.
    This avoids reconstructing full dataclass objects — works directly
    from the to_dict() output that's already in the regime JSON.
    """
    notes = []
    pairs = corr_dict.get("pairs", {})
    if not pairs:
        return notes

    notes.append(
        "- CROSS-ASSET CORRELATIONS (COMPUTED — use these exact numbers, "
        "do NOT invent correlation values or thresholds):"
    )

    for pair_name, pair_data in pairs.items():
        rolling = pair_data.get("rolling_correlations", {})
        beta = pair_data.get("beta", {})
        vol_cond = pair_data.get("vol_regime_conditional", {})
        convex = pair_data.get("convexity", {})
        assets = pair_data.get("assets", {})
        regime = pair_data.get("regime_label", "normal")

        parts = [f"    {assets.get('a', '?')} vs {assets.get('b', '?')}:"]

        c60 = rolling.get("corr_60d")
        if c60 is not None:
            parts.append(f"60d corr = {c60:.3f}")

        pctl = rolling.get("percentile_vs_1y")
        if pctl is not None:
            parts.append(f"({pctl:.0f}th pctl vs 1Y)")

        b60 = beta.get("beta_60d")
        if b60 is not None:
            parts.append(f"beta = {b60:.3f}")

        spread = vol_cond.get("spread")
        if spread is not None:
            if spread > 0.15:
                parts.append("STRENGTHENS in stress")
            elif spread < -0.15:
                parts.append("WEAKENS in stress")

        cscore = convex.get("score")
        if cscore is not None:
            if cscore > 0.05:
                parts.append("convex (amplifies at extremes)")
            elif cscore < -0.05:
                parts.append("concave (dampens at extremes)")

        if regime and regime != "normal":
            parts.append(f"REGIME: {regime}")

        notes.append(", ".join(parts))

    # Aggregate
    agg = corr_dict.get("aggregate", {})
    div_regime = agg.get("diversification_regime")
    avg_corr = agg.get("avg_absolute_correlation")
    stress_count = agg.get("stress_coupling_count", 0)
    breakdown_count = agg.get("breakdown_count", 0)
    if div_regime:
        notes.append(
            f"    Diversification: {div_regime} (avg |corr| = {avg_corr:.2f}). "
            f"Stress couplings: {stress_count}. Breakdowns: {breakdown_count}."
        )

    return notes


def _extract_interpretation_json(text: str) -> dict:
    """
    Extract the interpretation JSON from Claude's response text.

    Claude's response may contain web search content with embedded JSON
    (JSON-LD, API responses, stray { } in HTML/JS). The balanced-brace
    scanner breaks when web-search text has unmatched braces.

    This version uses KEY-TARGETED EXTRACTION: search for a key that only
    appears in our interpretation schema (e.g. "macro_regime"), walk
    backwards to the nearest '{', then try json.loads on progressively
    larger substrings ending at later '}' characters.
    """
    # Markers unique to the interpretation JSON (not in typical web content)
    markers = ['"macro_regime"', '"key_findings"', '"summary"']

    for marker in markers:
        idx = text.find(marker)
        if idx == -1:
            continue

        # Walk backwards from the marker to find the opening brace
        open_brace = text.rfind("{", 0, idx)
        if open_brace == -1:
            continue

        # Try closing braces from the END of text backwards
        # (the interpretation JSON is usually the last/largest block)
        search_from = idx
        while True:
            close_brace = text.rfind("}", search_from)
            if close_brace == -1 or close_brace <= open_brace:
                break
            candidate = text[open_brace:close_brace + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and {"summary", "key_findings"}.intersection(parsed.keys()):
                    return parsed
            except json.JSONDecodeError:
                # Try a smaller substring (earlier closing brace)
                search_from = close_brace - 1
                if search_from <= open_brace:
                    break
                continue

    # Fallback: try the raw text as-is (no web search prefix)
    text_stripped = text.strip()
    if text_stripped.startswith("{"):
        try:
            parsed = json.loads(text_stripped)
            if isinstance(parsed, dict) and "summary" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    return {}


def _strip_cite_tags(text: str) -> str:
    """Remove <cite index="...">...</cite> tags, keeping the inner text."""
    if not isinstance(text, str):
        return text
    return re.sub(r'</?cite[^>]*>', '', text)


class MacroStrategist(BaseAgent):
    """
    The Macro Strategist — understands the big picture.

    Produces a regime assessment (risk-on, risk-off, transition, crisis)
    backed by yield curve analysis, volatility analysis, breadth analysis,
    and cross-asset signals. Claude interprets the composite picture.
    """

    def __init__(self, cache: MarketDataCache, state: RedisState):
        super().__init__()
        self.cache = cache
        self.state = state
        self._client = anthropic.AsyncAnthropic(api_key=get_settings().anthropic_api_key)

    @property
    def name(self) -> str:
        return "macro_strategist"

    @property
    def layer(self) -> AgentLayer:
        return AgentLayer.DOMAIN_EXPERT

    @property
    def description(self) -> str:
        return (
            "Macro Strategist: Analyzes the macro-economic environment including "
            "yield curve dynamics, volatility regimes, market breadth, and cross-asset "
            "signals. Produces a market regime assessment (risk-on, risk-off, transition, "
            "crisis) with confidence levels and supporting data."
        )

    @property
    def capabilities(self) -> list[str]:
        return [
            "Assess current market regime (risk-on, risk-off, transition, crisis)",
            "Analyze yield curve shape and recession probability signals",
            "Evaluate volatility environment (VIX levels, term structure, realized vs implied)",
            "Assess market breadth and internal health",
            "Monitor cross-asset signals (dollar, credit spreads, inflation expectations)",
            "Identify macro risks and watch items",
            "Provide context for how macro environment affects specific sectors or trades",
        ]

    async def query(self, query: AgentQuery) -> AgentOpinion:
        """
        Process a macro query with the full Phase 1 computation pipeline.
        """
        logger.info(f"Processing query: {query.question[:100]}")

        lookback = 730  # 2 years for better cycle analysis
        from_date = date.today() - timedelta(days=lookback)
        to_date = date.today()

        # Fetch ALL FRED series
        fred_series_needed = [
            # Yields & Curve
            "DGS2", "DGS5", "DGS10", "DGS30", "DTB3", "T10Y2Y", "T10Y3M",
            "DFII5", "DFII10", "THREEFYTP10",
            # Monetary Policy
            "FEDFUNDS", "DFEDTARU", "DFEDTARL",
            "WALCL", "TREAST", "WSHOMCB", "RRPONTSYD", "WTREGEN", "WRESBAL",
            "NFCI", "ANFCI", "STLFSI4",
            "TOTBKCR", "TOTALSL", "M2SL",
            # Volatility
            "VIXCLS", "VXVCLS",
            # Cross-Asset
            "BAMLH0A0HYM2", "DTWEXBGS",
            # Inflation
            "CPIAUCSL", "CPILFESL", "CUSR0000SEHC", "CUSR0000SAS",
            "PCEPI", "PCEPILFE",
            "STICKCPIM159SFRBATL", "FLEXCPIM159SFRBATL",
            "PCETRIM12M159SFRBDAL",
            "T5YIE", "T10YIE", "T5YIFR",
            "PPIACO", "CES0500000003", "DCOILWTICO", "DCOILBRENTEU",
            # Business Cycle
            "UNRATE", "U6RATE", "ICSA", "CCSA",
            "JTSJOL", "JTSQUR",
            "TEMPHELPS", "AWHAETP",
            "PERMIT", "HOUST", "RSAFS",
            "UMCSENT", "INDPRO", "TCU", "CFNAI",
            "PI", "DGORDER", "PAYEMS",
            "GACDFSA066MSFRBPHI", "CIVPART", "LNS11300060",
            "GDPNOW",
        ]

        fred_data = {}
        for series_id in fred_series_needed:
            try:
                fred_data[series_id] = await self.cache.get_fred_series(
                    series_id, from_date=from_date,
                    force_refresh=query.require_fresh_data
                )
            except Exception as e:
                logger.warning(f"Failed to fetch FRED series {series_id}: {e}")
                fred_data[series_id] = []

        # Track missing FRED series for data quality reporting
        missing_fred = [sid for sid, data in fred_data.items() if not data]
        if missing_fred:
            logger.warning(f"Missing FRED series ({len(missing_fred)}): {', '.join(missing_fred[:10])}")

        # === Pre-computation data quality validation ===
        _quality_report = None
        _quality_warnings = []
        try:
            from finias.data.validation.quality import (
                check_series_gaps, validate_series, DataQualityReport, CONSECUTIVE_CRITICAL,
            )
            from finias.data.validation.fred_quality import FRED_FREQUENCY

            _quality_report = DataQualityReport()

            for series_id, series_data in fred_data.items():
                if series_id in CONSECUTIVE_CRITICAL and series_data:
                    frequency = FRED_FREQUENCY.get(series_id, "monthly")
                    report = validate_series(
                        observations=series_data,
                        series_id=series_id,
                        expected_frequency=frequency,
                        consecutive_required=True,
                    )
                    _quality_report.series_reports[series_id] = report

                    if report.status == "critical":
                        affected = CONSECUTIVE_CRITICAL[series_id]
                        issue = f"{series_id}: gap detected — {affected} may be inaccurate"
                        _quality_report.critical_issues.append(issue)
                        _quality_warnings.append(issue)
                        logger.warning(f"DATA QUALITY CRITICAL: {issue}")

            # Determine overall status
            if _quality_report.critical_issues:
                _quality_report.overall_status = "critical"
            elif missing_fred:
                _quality_report.overall_status = "degraded"
                _quality_report.warnings.append(
                    f"{len(missing_fred)} FRED series returned no data"
                )
            else:
                _quality_report.overall_status = "healthy"

        except Exception as e:
            logger.warning(f"Pre-computation quality check failed (non-blocking): {e}")

        # Populate the macro matrix from raw economic_indicators
        try:
            matrix_count = await self.cache.populate_macro_matrix()
            logger.info(f"Macro matrix populated: {matrix_count} dates")
        except Exception as e:
            logger.warning(f"Failed to populate macro matrix: {e}")

        # Fetch market data
        spx_bars = await self.cache.get_daily_bars(
            "SPY", from_date, to_date, force_refresh=query.require_fresh_data
        )
        spx_prices = [{"date": str(b["trade_date"]), "close": float(b["close"])} for b in spx_bars]

        # Sector ETFs for correlation analysis
        sector_etfs = ["XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLC", "XLY", "XLRE", "XLB"]
        sector_prices = {}
        for etf in sector_etfs:
            try:
                bars = await self.cache.get_daily_bars(etf, from_date, to_date)
                sector_prices[etf] = [{"date": str(b["trade_date"]), "close": float(b["close"])} for b in bars]
            except Exception as e:
                logger.warning(f"Failed to fetch {etf}: {e}")

        # Additional symbols for cross-asset and breadth analysis
        additional_symbols = {
            "RSP": None, "IWM": None, "TLT": None, "GLD": None,
            "HYG": None, "EEM": None, "CPER": None,
        }
        for symbol in additional_symbols:
            try:
                bars = await self.cache.get_daily_bars(symbol, from_date, to_date)
                additional_symbols[symbol] = [
                    {"date": str(b["trade_date"]), "close": float(b["close"])} for b in bars
                ]
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                additional_symbols[symbol] = None

        # Track missing Polygon symbols for data quality reporting
        missing_polygon = []
        if not spx_bars:
            missing_polygon.append("SPY")
        for symbol, data in additional_symbols.items():
            if not data:
                missing_polygon.append(symbol)
        for etf in sector_etfs:
            if etf not in sector_prices or not sector_prices[etf]:
                missing_polygon.append(etf)

        if missing_polygon:
            logger.warning(f"Missing Polygon symbols ({len(missing_polygon)}): {', '.join(missing_polygon)}")

        # === Read live prices from shared Redis cache ===
        live_prices = {}
        try:
            from finias.data.providers.price_feed import get_live_prices
            live_prices = await get_live_prices(self.state) or {}
            if live_prices and not live_prices.get("error"):
                fetched = {k: v for k, v in live_prices.items()
                           if k not in ("fetched_at", "source", "error") and v is not None}
                logger.info(f"Live prices from Redis: {len(fetched)} instruments")
            else:
                logger.info("No live prices available in Redis")
        except Exception as e:
            logger.warning(f"Could not read live prices: {e}")

        # Store for use in data notes during interpretation
        self._live_prices = live_prices

        # === Run ALL computation modules ===
        _computation_failures = []

        # 1. Yield Curve (enhanced)
        try:
            yc_analysis = analyze_yield_curve(
                yields_2y=fred_data.get("DGS2", []),
                yields_5y=fred_data.get("DGS5", []),
                yields_10y=fred_data.get("DGS10", []),
                yields_30y=fred_data.get("DGS30", []),
                yields_3m=fred_data.get("DTB3", []),
                fed_funds=fred_data.get("FEDFUNDS", []),
                real_yields_5y=fred_data.get("DFII5", []),
                real_yields_10y=fred_data.get("DFII10", []),
                term_premium_10y=fred_data.get("THREEFYTP10", []),
            )
        except Exception as e:
            logger.error(f"Yield curve computation failed: {e}")
            yc_analysis = YieldCurveAnalysis(
                t3m=None, t2y=None, t5y=None, t10y=None, t30y=None,
                spread_2s10s=None, spread_3m10y=None, spread_2s30s=None,
                spread_2s10s_change_30d=None, spread_2s10s_change_90d=None,
                is_2s10s_inverted=False, is_3m10y_inverted=False,
                inversion_depth_2s10s=0.0, days_inverted_2s10s=0,
                curve_shape="unknown", recession_signal_score=0.0
            )
            _computation_failures.append("yield_curve")

        # 2. Volatility (enhanced with term structure)
        try:
            # SKEW comes from live prices (yfinance) — not available on FRED
            skew_series = []
            skew_val = live_prices.get("skew") if live_prices else None
            if skew_val is not None:
                skew_series = [{"date": date.today().isoformat(), "value": skew_val}]

            vol_analysis = analyze_volatility(
                vix_series=fred_data.get("VIXCLS", []),
                spx_prices=spx_prices,
                vix3m_series=fred_data.get("VXVCLS", []),
                skew_series=skew_series,
            )
            # Add correlation if we have sector data
            if len(sector_prices) >= 5:
                from finias.agents.macro_strategist.computations.volatility import (
                    compute_sector_correlation, classify_correlation_regime
                )
                avg_corr = compute_sector_correlation(sector_prices)
                vol_analysis.sector_correlation = avg_corr
                vol_analysis.correlation_regime = classify_correlation_regime(avg_corr)
        except Exception as e:
            logger.error(f"Volatility computation failed: {e}")
            vol_analysis = VolatilityAnalysis(
                vix_current=None, vix_percentile_1y=None,
                vix_change_1d=None, vix_change_5d=None, vix_change_20d=None,
                vix_sma_20=None, vix_is_elevated=False, vix_is_spike=False,
                realized_vol_20d=None, realized_vol_60d=None, iv_rv_spread=None,
                vol_regime="unknown", vol_risk_score=0.0
            )
            _computation_failures.append("volatility")

        # 3. Breadth (now uses real ETF data)
        try:
            breadth_analysis = analyze_breadth(
                spx_prices=spx_prices,
                sector_prices=sector_prices,
                rsp_prices=additional_symbols.get("RSP"),
            )
        except Exception as e:
            logger.error(f"Breadth computation failed: {e}")
            breadth_analysis = BreadthAnalysis()
            _computation_failures.append("breadth")

        # 4. Cross-Asset (expanded with intermarket signals)
        try:
            ca_analysis = analyze_cross_assets(
                dxy_series=fred_data.get("DTWEXBGS", []),
                hy_spread_series=fred_data.get("BAMLH0A0HYM2", []),
                breakeven_5y=fred_data.get("T5YIE", []),
                breakeven_10y=fred_data.get("T10YIE", []),
                copper_prices=additional_symbols.get("CPER"),
                gold_prices=additional_symbols.get("GLD"),
                oil_series=fred_data.get("DCOILWTICO", []),
                brent_series=fred_data.get("DCOILBRENTEU", []),
                spy_prices=spx_prices,
                tlt_prices=additional_symbols.get("TLT"),
                iwm_prices=additional_symbols.get("IWM"),
                hyg_prices=additional_symbols.get("HYG"),
                eem_prices=additional_symbols.get("EEM"),
                vix_series=fred_data.get("VIXCLS", []),
            )
        except Exception as e:
            logger.error(f"Cross-asset computation failed: {e}")
            ca_analysis = CrossAssetAnalysis()
            _computation_failures.append("cross_asset")

        # 5. Monetary Policy (NEW)
        try:
            mp_analysis = analyze_monetary_policy(
                fed_funds=fred_data.get("FEDFUNDS", []),
                fed_target_upper=fred_data.get("DFEDTARU", []),
                fed_target_lower=fred_data.get("DFEDTARL", []),
                fed_total_assets=fred_data.get("WALCL", []),
                fed_treasuries=fred_data.get("TREAST", []),
                fed_mbs=fred_data.get("WSHOMCB", []),
                tga=fred_data.get("WTREGEN", []),
                reverse_repo=fred_data.get("RRPONTSYD", []),
                bank_reserves=fred_data.get("WRESBAL", []),
                nfci_series=fred_data.get("NFCI", []),
                stress_series=fred_data.get("STLFSI4", []),
                bank_credit=fred_data.get("TOTBKCR", []),
                consumer_credit=fred_data.get("TOTALSL", []),
                m2_series=fred_data.get("M2SL", []),
            )
        except Exception as e:
            logger.error(f"Monetary policy computation failed: {e}")
            mp_analysis = MonetaryPolicyAnalysis()
            _computation_failures.append("monetary_policy")

        # 6. Business Cycle (NEW)
        try:
            cycle_analysis = analyze_business_cycle(
                lei_series=[],  # Conference Board LEI removed from FRED; module handles gracefully
                unemployment=fred_data.get("UNRATE", []),
                initial_claims=fred_data.get("ICSA", []),
                continuing_claims=fred_data.get("CCSA", []),
                jolts_openings=fred_data.get("JTSJOL", []),
                jolts_quits=fred_data.get("JTSQUR", []),
                temp_employment=fred_data.get("TEMPHELPS", []),
                avg_weekly_hours=fred_data.get("AWHAETP", []),
                building_permits=fred_data.get("PERMIT", []),
                housing_starts=fred_data.get("HOUST", []),
                retail_sales=fred_data.get("RSAFS", []),
                consumer_sentiment=fred_data.get("UMCSENT", []),
                industrial_production=fred_data.get("INDPRO", []),
                capacity_utilization=fred_data.get("TCU", []),
                cfnai_series=fred_data.get("CFNAI", []),
                personal_income=fred_data.get("PI", []),
                durable_goods=fred_data.get("DGORDER", []),
                nfp_series=fred_data.get("PAYEMS", []),
                philly_fed=fred_data.get("GACDFSA066MSFRBPHI", []),
                gdp_nowcast_series=fred_data.get("GDPNOW", []),
                yield_curve_slope=yc_analysis.spread_3m10y,
            )
        except Exception as e:
            logger.error(f"Business cycle computation failed: {e}")
            cycle_analysis = BusinessCycleAnalysis()
            _computation_failures.append("business_cycle")

        # 7. Inflation (NEW)
        try:
            infl_analysis = analyze_inflation(
                cpi_all=fred_data.get("CPIAUCSL", []),
                cpi_core=fred_data.get("CPILFESL", []),
                cpi_shelter=fred_data.get("CUSR0000SEHC", []),
                cpi_services=fred_data.get("CUSR0000SAS", []),
                pce=fred_data.get("PCEPI", []),
                core_pce=fred_data.get("PCEPILFE", []),
                sticky_cpi=fred_data.get("STICKCPIM159SFRBATL", []),
                flexible_cpi=fred_data.get("FLEXCPIM159SFRBATL", []),
                trimmed_mean=fred_data.get("PCETRIM12M159SFRBDAL", []),
                breakeven_5y=fred_data.get("T5YIE", []),
                breakeven_10y=fred_data.get("T10YIE", []),
                forward_5y5y=fred_data.get("T5YIFR", []),
                ppi=fred_data.get("PPIACO", []),
                ahe=fred_data.get("CES0500000003", []),
                oil=fred_data.get("DCOILWTICO", []),
            )
        except Exception as e:
            logger.error(f"Inflation computation failed: {e}")
            infl_analysis = InflationAnalysis()
            _computation_failures.append("inflation")

        # === Detect regime with full hierarchy ===
        # Load historical data for dynamic weighting
        spx_returns_np = None
        historical_scores = None
        try:
            # Get historical regime assessments for dynamic weighting
            history_rows = await self.cache.db.fetch(
                """
                SELECT growth_cycle_score, monetary_liquidity_score,
                       inflation_score, market_signals_score, assessed_at
                FROM regime_assessments
                ORDER BY assessed_at DESC
                LIMIT 60
                """
            )

            if len(history_rows) >= 10:
                # Reverse to chronological order
                history_rows = list(reversed(history_rows))
                historical_scores = {
                    "growth": [float(r["growth_cycle_score"]) for r in history_rows],
                    "monetary": [float(r["monetary_liquidity_score"]) for r in history_rows],
                    "inflation": [float(r["inflation_score"]) for r in history_rows],
                    "market": [float(r["market_signals_score"]) for r in history_rows],
                }

                # Compute SPX daily returns for the same period
                if spx_prices and len(spx_prices) >= len(history_rows):
                    closes = np.array([p["close"] for p in spx_prices[-(len(history_rows) + 1):]])
                    spx_returns_np = np.diff(np.log(closes))
                    # Align lengths
                    min_len = min(len(spx_returns_np), len(history_rows))
                    spx_returns_np = spx_returns_np[-min_len:]
                    for key in historical_scores:
                        historical_scores[key] = historical_scores[key][-min_len:]
        except Exception as e:
            logger.debug(f"Could not load historical scores for dynamic weighting: {e}")

        regime_assessment = detect_regime(
            yield_curve=yc_analysis,
            volatility=vol_analysis,
            breadth=breadth_analysis,
            cross_asset=ca_analysis,
            monetary_policy=mp_analysis,
            business_cycle=cycle_analysis,
            inflation_analysis=infl_analysis,
            spx_returns=spx_returns_np,
            historical_category_scores=historical_scores,
        )

        # Attach data gaps to regime assessment for data notes
        regime_assessment._data_gaps = {
            "fred_missing": missing_fred,
            "polygon_missing": missing_polygon,
            "fred_available": len(fred_series_needed) - len(missing_fred),
            "polygon_available": len(sector_etfs) + len(additional_symbols) + 1 - len(missing_polygon),
        }

        # Attach computation failures if any
        if _computation_failures:
            regime_assessment._data_gaps["computation_failures"] = _computation_failures

        # === Post-computation bounds check ===
        try:
            from finias.data.validation.bounds import check_computation_bounds
            bounds_violations = check_computation_bounds(
                regime_assessment.key_levels,
                additional_values={
                    "recession_prob": regime_assessment.key_levels.get("recession_prob"),
                    "composite_score": regime_assessment.composite_score,
                    "stress_index": regime_assessment.stress_index,
                    "confidence": regime_assessment.confidence,
                },
            )
            if bounds_violations:
                for v in bounds_violations:
                    logger.error(f"BOUNDS VIOLATION: {v}")
                    _quality_warnings.append(v)
                if _quality_report:
                    _quality_report.critical_issues.extend(bounds_violations)
                    _quality_report.overall_status = "critical"
        except Exception as e:
            logger.warning(f"Bounds check failed (non-blocking): {e}")

        # Attach quality report and warnings to regime assessment for downstream use
        regime_assessment._quality_report = _quality_report
        regime_assessment._quality_warnings = _quality_warnings

        # === Compute Trajectory Layer ===
        # Get prior assessment for trajectory change signals
        prior_regime = None
        try:
            prior_row = await self.cache.db.fetchrow(
                """
                SELECT inflation_score, stress_index, binding_constraint
                FROM regime_assessments
                ORDER BY id DESC LIMIT 1
                """
            )
            if prior_row:
                # Create a minimal prior object for comparison
                class _PriorRegime:
                    pass
                prior_regime = _PriorRegime()
                prior_regime.inflation_score = float(prior_row["inflation_score"]) if prior_row["inflation_score"] else 0.0
                prior_regime.stress_index = float(prior_row["stress_index"]) if prior_row["stress_index"] else 0.0
                prior_regime.binding_constraint = prior_row["binding_constraint"] or "none"
        except Exception as e:
            logger.warning(f"Could not fetch prior regime for trajectory: {e}")

        # Fetch prior trigger values for momentum computation
        prior_trigger_values = {}
        try:
            prior_regime_row = await self.cache.db.fetchrow(
                "SELECT full_regime_json FROM regime_assessments ORDER BY id DESC LIMIT 1"
            )
            if prior_regime_row and prior_regime_row["full_regime_json"]:
                prior_full = json.loads(prior_regime_row["full_regime_json"])
                prior_kl = prior_full.get("key_levels", {})
                prior_trigger_values = {
                    "sahm_value": prior_kl.get("sahm_value"),
                    "vix": prior_kl.get("vix"),
                    "hy_spread": prior_kl.get("hy_spread"),
                    "core_pce_3m_annualized": prior_kl.get("core_pce_3m_ann"),
                    "core_pce_yoy": prior_kl.get("core_pce_yoy"),
                    "net_liquidity_trillion": (prior_kl.get("net_liquidity", 0) or 0) / 1_000_000 if (prior_kl.get("net_liquidity") or 0) > 1000 else prior_kl.get("net_liquidity"),
                    "inflation_surprise_pp": None,  # Computed dynamically, not stored in key_levels
                }
        except Exception as e:
            logger.warning(f"Could not fetch prior trigger values: {e}")

        trajectory = compute_trajectory(
            regime_assessment=regime_assessment,
            fed_target_upper=fred_data.get("DFEDTARU", []),
            prior_regime_assessment=prior_regime,
            prior_trigger_values=prior_trigger_values,
        )

        # === CFTC Positioning Integration ===
        # Fetch COT data from PostgreSQL and compute positioning signals.
        # The S&P 500 positioning signal replaces stress_contrarian in forward_bias.
        positioning_analysis = None
        try:
            from finias.data.providers.cot_client import get_cot_history, get_cot_staleness_days, COT_CONTRACTS
            from finias.agents.macro_strategist.computations.positioning import (
                compute_positioning_analysis, generate_positioning_data_notes
            )

            contract_data = {}
            for contract_key in COT_CONTRACTS:
                history = await get_cot_history(self.cache.db, contract_key, lookback_weeks=156)
                if history:
                    contract_data[contract_key] = history

            if contract_data:
                staleness = await get_cot_staleness_days(self.cache.db)
                positioning_analysis = compute_positioning_analysis(contract_data, staleness_days=staleness)

                # Update trajectory with positioning signal
                trajectory.positioning_signal = positioning_analysis.sp500_positioning_signal
                sp500_cp = positioning_analysis.contracts.get("sp500")
                if sp500_cp:
                    trajectory.sp500_net_spec_percentile = sp500_cp.net_spec_percentile

                # Recompute forward_bias with positioning signal instead of stress_contrarian
                from finias.agents.macro_strategist.computations.trajectory import compute_forward_bias
                bias_info = compute_forward_bias(
                    trajectory.inflation_trajectory,
                    trajectory.positioning_signal,
                    trajectory.binding_shift_direction,
                )
                trajectory.forward_bias = bias_info["bias"]
                trajectory.forward_bias_score = bias_info["score"]
                trajectory.forward_bias_confidence = bias_info["confidence"]

                # Store positioning in regime assessment components
                regime_assessment.positioning = positioning_analysis.to_dict()
                self._positioning_analysis = positioning_analysis
            else:
                logger.info("No COT positioning data available — using stress_contrarian fallback")
        except Exception as e:
            logger.warning(f"Positioning computation failed (using stress_contrarian fallback): {e}")

        # Inject quality warnings into trajectory's data_freshness_warnings
        # These flow through to MacroContext.data_freshness_warnings for downstream agents
        if _quality_warnings:
            trajectory.data_freshness_warnings.extend(_quality_warnings)

        regime_assessment.trajectory = trajectory.to_dict()
        self._last_trajectory = trajectory

        # === Claude interpretation ===
        interpretation = await self._interpret(regime_assessment, query.question)

        # === Validate interpretation against computed data ===
        interpretation = self._validate_interpretation(
            interpretation,
            regime_assessment,
            live_prices=getattr(self, '_live_prices', None),
        )

        # Prepend binding constraint to summary (after validation may have corrected it)
        if interpretation.get("binding_constraint") and interpretation["binding_constraint"] not in interpretation.get("summary", ""):
            interpretation["summary"] = (
                f"Binding constraint: {interpretation['binding_constraint']}. "
                + interpretation.get("summary", "")
            )

        # === Publish to shared state (Redis) ===
        regime_dict = regime_assessment.to_dict()
        # Strip citation tags before publishing
        clean_interp = {}
        for key, value in interpretation.items():
            if isinstance(value, str):
                clean_interp[key] = _strip_cite_tags(value)
            elif isinstance(value, list):
                clean_interp[key] = [_strip_cite_tags(item) if isinstance(item, str) else item for item in value]
            else:
                clean_interp[key] = value
        regime_dict["interpretation"] = clean_interp
        await self.state.set_regime(regime_dict)
        await self.state.publish_opinion(self.name, regime_dict)

        # === Persist to database (PostgreSQL) ===
        try:
            await self._persist_opinion(regime_assessment, query, interpretation)
        except Exception as e:
            logger.warning(f"Failed to persist opinion to database: {e}")

        # === Build opinion ===
        direction = self._regime_to_direction(
            regime_assessment.primary_regime, regime_assessment.composite_score
        )
        confidence = self._score_to_confidence(regime_assessment.confidence)

        return AgentOpinion(
            agent_name=self.name,
            agent_layer=self.layer,
            direction=direction,
            confidence=confidence,
            regime=regime_assessment.primary_regime,
            summary=interpretation["summary"],
            key_findings=interpretation["key_findings"],
            data_points=regime_assessment.to_director_summary(),
            methodology=(
                "Hierarchical regime model with 4 category scores: "
                f"Growth/Cycle ({regime_assessment.weight_growth:.0%}), "
                f"Monetary/Liquidity ({regime_assessment.weight_monetary:.0%}), "
                f"Inflation ({regime_assessment.weight_inflation:.0%}), "
                f"Market Signals ({regime_assessment.weight_market:.0%}). "
                f"Binding constraint: {regime_assessment.binding_constraint}. "
                "Each category synthesizes multiple domain analyses. "
                "Python computes all indicators. Claude interprets the composite."
            ),
            risks_to_view=interpretation["risks"],
            watch_items=interpretation["watch_items"],
            data_freshness=datetime.now(timezone.utc),
        )

    async def health_check(self) -> HealthStatus:
        """Check if macro data is fresh and computations work."""
        try:
            fred_fresh = await self.state.get_data_freshness("fred:VIXCLS")
            polygon_fresh = await self.state.get_data_freshness("polygon:SPY")

            is_healthy = True
            details = {}

            if fred_fresh:
                details["fred_last_refresh"] = fred_fresh.isoformat()
            else:
                details["fred_status"] = "never refreshed"

            if polygon_fresh:
                details["polygon_last_refresh"] = polygon_fresh.isoformat()
            else:
                details["polygon_status"] = "never refreshed"

            return HealthStatus(
                agent_name=self.name,
                is_healthy=is_healthy,
                data_freshness=fred_fresh or polygon_fresh,
                details=details,
            )
        except Exception as e:
            return HealthStatus(
                agent_name=self.name,
                is_healthy=False,
                error_message=str(e),
            )

    async def _interpret(self, regime, question: str) -> dict:
        """
        Two-step interpretation: analysis then structuring.

        Step 1: Claude Opus + web search produces free-text analysis.
                No JSON requirement — Claude thinks freely, searches naturally.
        Step 2: Claude Sonnet (no web search) structures the free-text
                into the required JSON format. Clean input, clean output.

        This eliminates JSON parsing failures caused by web search content
        (JSON-LD, stray braces, JS snippets) contaminating the JSON output.
        """
        from finias.core.config.settings import get_settings
        from finias.agents.macro_strategist.prompts.interpretation import (
            MACRO_ANALYSIS_PROMPT,
            MACRO_STRUCTURING_PROMPT,
        )
        settings = get_settings()

        # Build regime data and data notes (same as before)
        regime_data = json.dumps(regime.to_dict(), indent=2, default=str)
        data_notes = self._build_data_notes(regime, live_prices=getattr(self, '_live_prices', None))

        date_context = (
            f"TODAY'S DATE: {date.today().isoformat()}. "
            f"All analysis and forward-looking statements should reference "
            f"dates relative to today.\n\n"
        )

        # Build temporal context from recent regime history
        historical_context = await self._build_historical_context()

        # Build continuity context from prior interpretation
        prior_assessment_context = await self._build_prior_assessment_context()

        # === STEP 1: Analysis (Opus + web search) ===
        # Claude produces free-text analysis — no JSON requirement
        analysis_prompt = (
            date_context
            + data_notes
            + historical_context
            + prior_assessment_context
            + MACRO_ANALYSIS_PROMPT.format(
                regime_data=regime_data,
                question=question,
            )
        )

        from finias.core.utils.retry import retry_claude_call

        logger.info("Step 1: Opus analysis with web search...")
        analysis_response = await retry_claude_call(
            lambda: self._client.messages.create(
                model=settings.claude_model,
                max_tokens=4000,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                }],
                messages=[{"role": "user", "content": analysis_prompt}],
            )
        )

        # Extract all text blocks from the analysis response
        analysis_parts = []
        for block in analysis_response.content:
            if hasattr(block, "text"):
                analysis_parts.append(block.text)
        analysis_text = "\n".join(analysis_parts)

        logger.info(f"Step 1 complete: {len(analysis_text)} chars of analysis")

        # === STEP 2: Structuring (Sonnet, NO web search) ===
        # Clean text in, clean JSON out. No web search contamination possible.
        structuring_prompt = MACRO_STRUCTURING_PROMPT.format(
            analysis_text=analysis_text,
        )

        logger.info("Step 2: Sonnet JSON structuring...")
        structure_response = await retry_claude_call(
            lambda: self._client.messages.create(
                model=settings.claude_model_fast,
                max_tokens=4000,
                messages=[{"role": "user", "content": structuring_prompt}],
            )
        )

        # Extract text from structuring response (should be clean JSON)
        structure_text = ""
        for block in structure_response.content:
            if hasattr(block, "text"):
                structure_text += block.text

        logger.info(f"Step 2 complete: {len(structure_text)} chars")

        # === Parse the structured JSON ===
        # Step 2 output should be clean JSON, but we keep fallbacks for safety
        try:
            # Primary: direct JSON parse (should work ~99% of the time with Step 2)
            result = json.loads(structure_text.strip())
            if not isinstance(result, dict) or "summary" not in result:
                raise json.JSONDecodeError("Missing expected keys", structure_text, 0)

        except json.JSONDecodeError:
            # Secondary: try the key-targeted parser on Step 2 output
            logger.warning("Step 2 direct parse failed, trying key-targeted extraction...")
            result = _extract_interpretation_json(structure_text)

            if not result:
                # Tertiary: try key-targeted parser on Step 1 output (the analysis text)
                logger.warning("Step 2 extraction failed, trying Step 1 output...")
                result = _extract_interpretation_json(analysis_text)

            if not result:
                # Final fallback: store analysis as summary
                logger.error("All JSON extraction failed. Storing raw analysis as summary.")
                result = {
                    "summary": analysis_text[:2000],  # Truncate if very long
                    "key_findings": [],
                    "risks": [],
                    "watch_items": [],
                    "macro_regime": "",
                    "binding_constraint": "",
                    "key_metrics": {},
                }

        # Ensure all expected keys exist
        result.setdefault("summary", "")
        result.setdefault("key_findings", [])
        result.setdefault("risks", [])
        result.setdefault("watch_items", [])
        result.setdefault("macro_regime", "")
        result.setdefault("binding_constraint", "")
        result.setdefault("key_metrics", {})
        result.setdefault("scenarios", [])
        result.setdefault("catalysts", [])
        result.setdefault("opportunities", [])
        result.setdefault("regime_change_conditions", {})

        # NOTE: binding_constraint prepend moved to query() — runs AFTER validation

        return result

    def _build_data_notes(self, regime, live_prices: dict = None) -> str:
        """
        Build plain-English notes for fields that Claude tends to misinterpret.

        These notes are prepended to the interpretation prompt so Claude reads
        unambiguous descriptions BEFORE encountering the raw JSON numbers.
        This is more reliable than metadata annotations in to_dict() because
        Claude processes the notes as natural language context, not as data labels.
        """
        notes = []
        regime_dict = regime.to_dict()

        # --- Cross-Asset: IWM/SPY relative return ---
        ca = regime_dict.get("components", {}).get("cross_asset", {})
        ra = ca.get("risk_appetite", {})
        iwm_val = ra.get("iwm_vs_spy_relative_return_20d_percentage_points")
        if iwm_val is not None:
            direction = "outperformed" if iwm_val > 0 else "underperformed"
            notes.append(
                f"- IWM vs SPY: IWM {direction} SPY by {abs(iwm_val):.2f} percentage points "
                f"over the last 20 trading days. The raw value is {iwm_val:.2f}pp. "
                f"This is NOT {abs(iwm_val * 10):.1f}% — it is {abs(iwm_val):.2f} percentage points."
            )

        # --- Live Price Divergences (yfinance vs FRED) ---
        lp = live_prices or {}
        if lp and not lp.get("error"):
            divergence_lines = []

            # VIX
            live_vix = lp.get("vix")
            fred_vix = regime_dict.get("components", {}).get("volatility", {}).get("vix", {}).get("current")
            if live_vix is not None and fred_vix is not None and abs(live_vix - fred_vix) > 1.0:
                divergence_lines.append(
                    f"VIX: FRED={fred_vix:.2f}, Live={live_vix:.2f}. USE LIVE VALUE."
                )

            # WTI
            live_wti = lp.get("wti")
            fred_wti = regime_dict.get("components", {}).get("cross_asset", {}).get("oil", {}).get("wti_price")
            if live_wti is not None and fred_wti is not None and abs(live_wti - fred_wti) > 2.0:
                divergence_lines.append(
                    f"WTI Oil: FRED=${fred_wti:.2f}, Live=${live_wti:.2f}. USE LIVE VALUE."
                )

            # Brent
            live_brent = lp.get("brent")
            fred_brent = regime_dict.get("components", {}).get("cross_asset", {}).get("oil", {}).get("brent_price")
            if live_brent is not None and fred_brent is not None and abs(live_brent - fred_brent) > 2.0:
                divergence_lines.append(
                    f"Brent Oil: FRED=${fred_brent:.2f}, Live=${live_brent:.2f}. USE LIVE VALUE."
                )

            # Live WTI-Brent spread (more current than FRED-based spread)
            if live_wti is not None and live_brent is not None:
                live_spread = live_wti - live_brent
                spread_note = f"LIVE WTI-Brent Spread: ${live_spread:.2f}"
                if live_spread > 0:
                    spread_note += " (WTI premium — unusual, suggests domestic supply tightness)"
                else:
                    spread_note += " (Brent premium — standard during global supply disruptions)"
                if abs(live_spread) > 5:
                    spread_note += f". |Spread| > $5 = geopolitical supply disruption signal."
                divergence_lines.append(spread_note)

            # Dollar
            live_dxy = lp.get("dxy")
            fred_dxy = regime_dict.get("components", {}).get("cross_asset", {}).get("dollar", {}).get("dxy")
            if live_dxy is not None and fred_dxy is not None and abs(live_dxy - fred_dxy) > 1.0:
                divergence_lines.append(
                    f"Dollar/DXY: FRED={fred_dxy:.2f}, Live={live_dxy:.2f}. USE LIVE VALUE."
                )

            # Gold and SPX (informational, always show if available)
            live_gold = lp.get("gold")
            if live_gold is not None:
                divergence_lines.append(f"Gold (live): ${live_gold:.2f}")

            live_spx = lp.get("spx")
            if live_spx is not None:
                divergence_lines.append(f"S&P 500 (live): {live_spx:.2f}")

            # SKEW (always show — not available from FRED)
            live_skew = lp.get("skew")
            if live_skew is not None:
                skew_label = ("complacent" if live_skew < 120 else "normal" if live_skew < 135
                             else "elevated" if live_skew < 150 else "extreme")
                divergence_lines.append(
                    f"CBOE SKEW (live): {live_skew:.0f} ({skew_label} tail risk hedging)"
                )

            if divergence_lines:
                notes.append(
                    "- LIVE MARKET PRICES (yfinance, current session) — "
                    "these supersede stale FRED values when they differ materially. "
                    "USE LIVE VALUES in your analysis:\n    "
                    + "\n    ".join(divergence_lines)
                )

        # --- Business Cycle: Custom Leading Indicator ---
        bc = regime_dict.get("components", {}).get("business_cycle", {})
        cli = bc.get("custom_leading_indicator", {})
        cli_val = cli.get("composite_value")
        cli_trend = cli.get("trend", "unknown")
        if cli_val is not None:
            notes.append(
                f"- Custom Leading Indicator: {cli_val:.2f} (trend: {cli_trend}). "
                f"This is a CUSTOM composite built from claims, permits, sentiment, and hours. "
                f"It is NOT the Conference Board LEI. Refer to it as 'custom leading indicator' "
                f"not 'LEI' or 'leading economic indicator' or 'leading index.'"
            )

        # --- Business Cycle: ISM Proxy ---
        mfg = bc.get("manufacturing_activity", {})
        mfg_val = mfg.get("value")
        is_proxy = mfg.get("is_proxy_NOT_actual_ISM", True)
        if mfg_val is not None and is_proxy:
            notes.append(
                f"- Manufacturing Activity: {mfg_val:.1f} (Philly Fed-derived PROXY). "
                f"This is NOT the official ISM Manufacturing PMI. It is derived from the "
                f"Philadelphia Fed regional survey. Say 'manufacturing activity proxy at {mfg_val:.1f}' "
                f"not 'ISM Manufacturing at {mfg_val:.1f}.'"
            )

        # --- Monetary Policy: Net Liquidity ---
        mp = regime_dict.get("components", {}).get("monetary_policy", {})
        liq = mp.get("liquidity", {})
        net_liq_m = liq.get("net_liquidity_millions")
        net_liq_t = liq.get("net_liquidity_trillions")
        if net_liq_t is not None:
            notes.append(
                f"- Net Liquidity: ${net_liq_t:.3f} trillion (= ${net_liq_m:,.0f} million). "
                f"Always express in TRILLIONS, not millions."
            )

        # --- Monetary Policy: Balance Sheet Direction ---
        bs = mp.get("balance_sheet", {})
        pace = bs.get("monthly_pace_millions")
        if pace is not None:
            if pace > 0:
                notes.append(
                    f"- Fed Balance Sheet: GROWING by ~${abs(pace/1000):.1f}B/month. "
                    f"This is NOT quantitative tightening — the balance sheet is expanding."
                )
            elif pace < -5:
                notes.append(
                    f"- Fed Balance Sheet: SHRINKING by ~${abs(pace/1000):.1f}B/month (QT active)."
                )

        # --- Cross-Asset: EM Relative Performance ---
        em = ca.get("em", {})
        em_val = em.get("relative_return_vs_spy_20d_percentage_points")
        if em_val is not None:
            notes.append(
                f"- EM vs SPY: EEM {'outperformed' if em_val > 0 else 'underperformed'} SPY by "
                f"{abs(em_val):.1f} percentage points over 20 days."
            )

        # --- Breadth: SPY/RSP ---
        br = regime_dict.get("components", {}).get("breadth", {})
        spy_rsp = br.get("spy_rsp", {})
        rsp_change = spy_rsp.get("ratio_change_20d")
        if rsp_change is not None:
            if rsp_change > 0.005:
                notes.append(
                    f"- SPY/RSP: Cap-weighted slightly outperforming equal-weight over 20 days "
                    f"(ratio change: {rsp_change:.4f}). Do NOT cite the absolute ratio level."
                )
            elif rsp_change < -0.005:
                notes.append(
                    f"- SPY/RSP: Equal-weight outperforming cap-weighted over 20 days "
                    f"(ratio change: {rsp_change:.4f}). Broad breadth improving."
                )

        # --- Sector Absolute Returns ---
        sector_rets = br.get("sector_returns", {})
        if sector_rets and len(sector_rets) >= 5:
            # Sort by 20d return descending
            sorted_sectors = sorted(
                sector_rets.items(),
                key=lambda x: x[1].get("20d", 0),
                reverse=True,
            )

            NAMES = {
                "XLB": "Materials", "XLC": "Comm Svcs", "XLE": "Energy",
                "XLF": "Financials", "XLI": "Industrials", "XLK": "Technology",
                "XLP": "Staples", "XLRE": "Real Estate", "XLU": "Utilities",
                "XLV": "Healthcare", "XLY": "Cons Disc",
            }

            lines = ["- SECTOR PERFORMANCE (Polygon ETFs, absolute returns):"]
            for sym, rets in sorted_sectors:
                name = NAMES.get(sym, sym)
                r5 = f"{rets.get('5d', 0):+.1f}%" if '5d' in rets else "N/A"
                r20 = f"{rets.get('20d', 0):+.1f}%" if '20d' in rets else "N/A"
                r60 = f"{rets.get('60d', 0):+.1f}%" if '60d' in rets else "N/A"

                # Label leading/lagging
                leading = br.get("sector_rotation", {}).get("leading", [])
                lagging = br.get("sector_rotation", {}).get("lagging", [])
                label = " ★LEADING" if sym in leading else " ▼LAGGING" if sym in lagging else ""

                lines.append(f"    {name:12s} ({sym}): 5d={r5}, 20d={r20}, 60d={r60}{label}")

            notes.append("\n".join(lines))

        # --- SKEW Index context ---
        vol = regime_dict.get("components", {}).get("volatility", {})
        skew_data = vol.get("skew", {})
        skew_val = skew_data.get("current")
        skew_regime = skew_data.get("regime")
        if skew_val is not None:
            vix_val = vol.get("vix", {}).get("current")
            if skew_regime in ("elevated", "extreme"):
                if vix_val and vix_val < 25:
                    notes.append(
                        f"- SKEW-VIX DIVERGENCE: SKEW at {skew_val:.0f} ({skew_regime}) while VIX "
                        f"only {vix_val:.1f}. This means institutions are quietly hedging tail risk "
                        f"even though headline volatility appears moderate. Watch for delayed VIX catch-up."
                    )
                else:
                    notes.append(
                        f"- SKEW INDEX: {skew_val:.0f} ({skew_regime}). Elevated demand for "
                        f"out-of-the-money put protection, confirming stress visible in VIX."
                    )
            elif skew_regime == "complacent" and vix_val and vix_val > 25:
                notes.append(
                    f"- SKEW-VIX DIVERGENCE: VIX at {vix_val:.1f} (elevated) but SKEW only "
                    f"{skew_val:.0f} (complacent). Institutions are NOT hedging despite elevated "
                    f"headline vol — suggests they view current volatility as transient."
                )

        # --- GDPNow Staleness Check ---
        bc = regime_dict.get("components", {}).get("business_cycle", {})
        gdpnow_data = bc.get("gdp_nowcast", {})
        gdpnow_val = gdpnow_data.get("value") if isinstance(gdpnow_data, dict) else gdpnow_data
        if gdpnow_val is not None:
            # GDPNow on FRED is quarterly — it shows the FINAL estimate for the prior quarter,
            # NOT the current quarter's real-time nowcast. The Atlanta Fed updates the real
            # GDPNow multiple times per week, but FRED only captures the quarterly final.
            notes.append(
                f"- GDPNow: {gdpnow_val:.1f}% — CRITICAL WARNING: The FRED GDPNow series updates "
                f"only QUARTERLY (final estimate per quarter). This value is likely the PRIOR "
                f"quarter's final estimate, NOT the current quarter's real-time nowcast. "
                f"The actual current-quarter GDPNow from the Atlanta Fed may be significantly "
                f"different. USE WEB SEARCH to find 'Atlanta Fed GDPNow current estimate' for "
                f"the real-time value. Do NOT present this FRED value as the current growth estimate "
                f"without verifying via web search."
            )

        # --- Recession Probability Disclaimer ---
        recession_prob = regime_dict.get("key_levels", {}).get("recession_prob", 0)
        if recession_prob is not None and recession_prob > 0:
            notes.append(
                f"- Recession Probability: {recession_prob*100:.0f}% — This is a HEURISTIC estimate "
                f"based on weighted combination of Sahm Rule, yield curve, claims, and cycle phase. "
                f"It is NOT a calibrated statistical probability from a trained model. A future "
                f"logistic regression model will replace this. For now, treat as directional guidance "
                f"(low/moderate/elevated risk) rather than a precise percentage."
            )

        # --- Recession Probability Drivers ---
        rec_drivers = bc.get("recession_drivers")
        if rec_drivers and isinstance(rec_drivers, dict) and rec_drivers.get("drivers"):
            prob = rec_drivers.get("probability", 0)
            base = rec_drivers.get("base_rate", 0)
            driver_list = rec_drivers.get("drivers", [])

            lines = [
                f"- RECESSION PROBABILITY DECOMPOSITION (calibrated logistic model, "
                f"AUC=0.99): Current probability {prob:.1%}, base rate {base:.1%}."
            ]
            for d in driver_list[:3]:  # Top 3 drivers only
                feat = d["feature"].replace("_", " ").title()
                val = d["value"]
                std = d["std_devs_from_mean"]
                contrib = d["contribution"]
                direction = "pushes probability UP" if contrib > 0 else "pushes probability DOWN"
                lines.append(
                    f"    {feat}: {val:.2f} ({std:+.1f} std devs from mean, "
                    f"contribution {contrib:+.3f}, {direction})"
                )

            notes.append("\n".join(lines))

        # --- Regime Classification Context ---
        primary = regime_dict.get("regime", {})
        regime_label = primary.get("primary", "unknown") if isinstance(primary, dict) else str(primary)
        if regime_label == "transition":
            notes.append(
                f"- Regime: 'transition' is the BASELINE state — the system classified 96% of "
                f"196 backtest observations as 'transition'. Do NOT interpret this as uncertain "
                f"or temporary. It is the normal state. Use the forward_bias field "
                f"(constructive/neutral/cautious) for directional assessment, not the regime label."
            )

        # --- Data Quality Warnings ---
        quality_report = getattr(regime, '_quality_report', None)
        if quality_report and quality_report.critical_issues:
            quality_notes = quality_report.get_quality_warnings_for_notes()
            for qn in quality_notes:
                notes.append(f"- {qn}")

        # --- Data Gaps Warning ---
        # This is populated by the query() method if any FRED series or Polygon symbols are missing
        data_gaps = getattr(regime, '_data_gaps', None)
        if data_gaps:
            missing_fred = data_gaps.get("fred_missing", [])
            missing_polygon = data_gaps.get("polygon_missing", [])
            if missing_fred or missing_polygon:
                gap_parts = []
                if missing_fred:
                    gap_parts.append(f"{len(missing_fred)} FRED series ({', '.join(missing_fred[:5])}{'...' if len(missing_fred) > 5 else ''})")
                if missing_polygon:
                    gap_parts.append(f"{len(missing_polygon)} Polygon symbols ({', '.join(missing_polygon[:5])})")
                notes.append(
                    f"- DATA GAPS: Missing {'; '.join(gap_parts)}. "
                    f"Assessment may be incomplete for affected domains. Reduce confidence accordingly."
                )

        # --- Computation Failures ---
        failures = (data_gaps or {}).get("computation_failures", [])
        if failures:
            notes.append(
                f"- COMPUTATION FAILURES: The following modules failed and are excluded from this assessment: "
                f"{', '.join(failures)}. Reduce confidence accordingly and note the gap in your analysis."
            )

        # --- Trajectory: Rate Decision History ---
        traj = regime_dict.get("trajectory", {})
        rate_info = traj.get("rate_decisions", {})
        decisions = rate_info.get("decisions_12m", [])
        cumulative = rate_info.get("cumulative_change_bp", 0)
        trajectory_str = rate_info.get("policy_trajectory", "unknown")
        if decisions:
            last_decision = decisions[-1]
            notes.append(
                f"- Fed Rate History: The Fed has made {len(decisions)} rate change(s) over the past 12 months, "
                f"totaling {cumulative:+.0f}bp. Policy trajectory: {trajectory_str}. "
                f"Last change: {last_decision['change_bp']:+.0f}bp on {last_decision['date']} "
                f"(rate after: {last_decision['rate_after']:.2f}%)."
            )
        elif trajectory_str == "holding":
            months = rate_info.get("months_since_last_change", 0)
            notes.append(
                f"- Fed Rate History: No rate changes in the past 12 months. "
                f"The Fed has been on hold for {months:.0f} months."
            )

        # --- Trajectory: Inflation Surprise ---
        surprise_info = traj.get("inflation_surprise", {})
        surprise_pp = surprise_info.get("surprise_pp", 0)
        surprise_dir = surprise_info.get("direction", "neutral")
        if abs(surprise_pp) > 0.1:
            notes.append(
                f"- Inflation Surprise: Core PCE exceeds 5Y breakeven by {surprise_pp:+.2f}pp — "
                f"{surprise_dir} surprise. {'Market underpricing inflation persistence.' if surprise_dir == 'hawkish' else 'Inflation coming in below expectations.'}"
            )

        # --- Trajectory: Forward Bias ---
        bias_info = traj.get("forward_bias", {})
        bias = bias_info.get("bias", "neutral")
        confidence = bias_info.get("confidence", "low")
        signals = traj.get("trajectory_signals", {})
        infl_traj = signals.get("inflation_trajectory", "unknown")
        pos_sig = signals.get("positioning_signal", "neutral")
        sp500_pctl = signals.get("sp500_net_spec_percentile", 50.0)
        shifted = signals.get("binding_shifted", False)
        shift_dir = signals.get("shift_direction", "none")

        if bias != "neutral" or infl_traj != "unknown":
            parts = []
            if infl_traj != "unknown":
                parts.append(f"inflation {infl_traj}")
            if pos_sig != "neutral":
                parts.append(f"positioning {pos_sig} (S&P 500 at {sp500_pctl:.0f}th percentile)")
            if shifted:
                parts.append(f"binding constraint shifted {shift_dir}")
            notes.append(
                f"- Forward Macro Bias: {bias} ({confidence} confidence). "
                f"Signals: {', '.join(parts) if parts else 'none active'}."
            )

        # --- Trajectory: Sector Guidance ---
        sector_info = traj.get("sector_guidance", {})
        overweights = sector_info.get("overweight", [])
        underweights = sector_info.get("underweight", [])
        rationale = sector_info.get("rationale", "")
        if overweights:
            from finias.agents.macro_strategist.computations.trajectory import SECTOR_NAMES
            ow_names = [SECTOR_NAMES.get(s, s) for s in overweights]
            uw_names = [SECTOR_NAMES.get(s, s) for s in underweights]
            notes.append(
                f"- Empirical Sector Guidance: Overweight {', '.join(ow_names)}. "
                f"Underweight {', '.join(uw_names)}. "
                f"Based on historical sector returns during similar macro conditions (196 weeks of data)."
            )

        # --- Position Sizing ---
        sizing = traj.get("position_sizing", {})
        max_pos = sizing.get("max_single_position_pct")
        beta = sizing.get("portfolio_beta_target")
        cash = sizing.get("cash_target_pct")
        reduce = sizing.get("reduce_overall_exposure", False)
        if max_pos is not None:
            notes.append(
                f"- Position Sizing: Max single position {max_pos}%, max sector {sizing.get('max_sector_exposure_pct')}%, "
                f"target portfolio beta {beta}, cash target {cash}%. "
                f"{'REDUCE OVERALL EXPOSURE — conditions warrant defensive positioning.' if reduce else 'Standard exposure limits apply.'}"
            )

        # --- Event Calendar ---
        events = traj.get("event_calendar", {})
        upcoming = events.get("upcoming_events", [])
        multiplier = events.get("pre_event_sizing_multiplier", 1.0)
        if upcoming:
            next_event = upcoming[0]
            notes.append(
                f"- Upcoming Event: {next_event['event']} in {next_event['days_away']} days ({next_event['date']}). "
                f"Pre-event sizing multiplier: {multiplier}x. "
                f"{'Position sizes reduced due to upcoming high-impact event.' if multiplier < 1.0 else 'No pre-event reduction needed.'}"
            )

        # --- Velocity ---
        velocity = traj.get("velocity", {})
        urgency = velocity.get("urgency", "normal")
        if urgency != "normal":
            vel_parts = []
            if velocity.get("vix") in ("spiking", "rising_fast"):
                vel_parts.append(f"VIX {velocity['vix']}")
            if velocity.get("credit_spreads") == "rapid_widening":
                vel_parts.append("credit spreads rapidly widening")
            if velocity.get("breadth") == "collapsing":
                vel_parts.append("breadth collapsing")
            if velocity.get("liquidity") == "draining":
                vel_parts.append("liquidity draining")
            notes.append(
                f"- URGENCY {urgency.upper()}: {', '.join(vel_parts) if vel_parts else 'multiple indicators deteriorating rapidly'}. "
                f"Conditions are changing faster than normal — monitor closely."
            )

            # --- Velocity-Aware Staleness Warning ---
            # In high-velocity environments, fast-moving metrics can become materially
            # stale within days. Tell Claude which specific values to verify via web search.
            stale_metrics = []

            if velocity.get("vix") in ("spiking", "rising_fast"):
                vix_val = regime_dict.get("components", {}).get("volatility", {}).get("vix", {}).get("level")
                if vix_val is not None:
                    stale_metrics.append(f"VIX (computed: {vix_val})")
                else:
                    stale_metrics.append("VIX (computed value unavailable)")

            if velocity.get("credit_spreads") in ("rapid_widening", "widening"):
                hy_val = ca.get("credit", {}).get("hy_spread")
                if hy_val is not None:
                    stale_metrics.append(f"HY spread (computed: {hy_val}%)")
                else:
                    stale_metrics.append("HY spread (computed value unavailable)")

            if velocity.get("dollar") in ("surging", "strengthening"):
                dxy_val = ca.get("dollar", {}).get("dxy")
                if dxy_val is not None:
                    stale_metrics.append(f"Dollar/DXY (computed: {dxy_val})")
                else:
                    stale_metrics.append("Dollar/DXY (computed value unavailable)")

            # Always include oil and SPY in high-velocity environments
            oil_wti = ca.get("oil", {}).get("wti_price")
            oil_brent = ca.get("oil", {}).get("brent_price")
            spread = ca.get("oil", {}).get("wti_brent_spread")
            spread_wide = ca.get("oil", {}).get("wti_brent_spread_widening", False)

            if oil_wti is not None:
                oil_note = f"Oil/WTI (computed: ${oil_wti:.2f})"
                if oil_brent is not None:
                    oil_note += f", Brent (computed: ${oil_brent:.2f})"
                    if spread is not None:
                        oil_note += f", spread: ${spread:.2f}"
                if spread_wide:
                    oil_note += (
                        " — NOTE: WTI-Brent spread exceeds $5, indicating geopolitical "
                        "supply disruption affecting global benchmarks more than domestic. "
                        "Use Brent as the primary oil reference for global impact assessment."
                    )
                stale_metrics.append(oil_note)
            else:
                stale_metrics.append("Oil/WTI (computed value unavailable)")

            stale_metrics.append("SPY level")

            notes.append(
                f"- STALE DATA WARNING — High velocity environment means computed values may be "
                f"materially outdated. USE WEB SEARCH to verify current levels before referencing "
                f"these in your analysis: {', '.join(stale_metrics)}. "
                f"Cite the web-searched values alongside computed values when they differ materially."
            )

        # --- Oil: WTI vs Brent context ---
        oil_data = ca.get("oil", {})
        wti_p = oil_data.get("wti_price")
        brent_p = oil_data.get("brent_price")
        spread_val = oil_data.get("wti_brent_spread")
        if wti_p is not None and brent_p is not None and spread_val is not None:
            notes.append(
                f"- OIL PRICES: WTI ${wti_p:.2f}, Brent ${brent_p:.2f}, "
                f"spread ${spread_val:.2f} (WTI minus Brent). "
                f"{'SPREAD WIDENED >$5 — geopolitical supply premium on global benchmark. ' if abs(spread_val) > 5 else ''}"
                f"Use Brent for global impact, WTI for domestic impact."
            )

        # --- Scenario Triggers (with timeframe and momentum) ---
        triggers = traj.get("scenario_triggers", [])
        # Show fast/critical triggers prominently, slow triggers with context
        fast_triggers = [t for t in triggers if t.get("timeframe") == "fast" and t.get("distance", 999) < 10]
        medium_triggers = [t for t in triggers if t.get("timeframe") == "medium" and t.get("distance", 999) < 2]
        slow_triggers = [t for t in triggers if t.get("timeframe") == "slow" and t.get("momentum") == "toward_threshold"]

        for t in fast_triggers:
            momentum_str = f", momentum: {t.get('momentum', 'unknown')}" if t.get('momentum') != 'unknown' else ""
            notes.append(
                f"- FAST TRIGGER: {t['metric']} at {t['current']}, threshold {t['threshold']} "
                f"(distance: {t['distance']}{momentum_str}). "
                f"{t.get('framing_note', '')} If breached: {t['consequence']}."
            )

        for t in medium_triggers:
            momentum_str = f", momentum: {t.get('momentum', 'unknown')}" if t.get('momentum') != 'unknown' else ""
            notes.append(
                f"- MEDIUM TRIGGER: {t['metric']} at {t['current']}, threshold {t['threshold']} "
                f"(distance: {t['distance']}{momentum_str}). "
                f"{t.get('framing_note', '')} If breached: {t['consequence']}."
            )

        for t in slow_triggers:
            notes.append(
                f"- SLOW TRIGGER (deteriorating): {t['metric']} at {t['current']}, threshold {t['threshold']} "
                f"(distance: {t['distance']}, momentum: toward_threshold, change: {t.get('change', 'N/A')}). "
                f"{t.get('framing_note', '')} If breached: {t['consequence']}."
            )

        # --- Cross-Asset Correlations ---
        corr_data = ca.get("correlations")
        if corr_data and corr_data.get("pairs"):
            corr_notes = _generate_correlation_notes_from_dict(corr_data)
            notes.extend(corr_notes)

        # --- CFTC Positioning ---
        positioning = getattr(self, '_positioning_analysis', None)
        if positioning is not None:
            from finias.agents.macro_strategist.computations.positioning import generate_positioning_data_notes
            pos_notes = generate_positioning_data_notes(positioning)
            notes.extend(pos_notes)

        if not notes:
            return ""

        return (
            "IMPORTANT DATA NOTES — Read these BEFORE interpreting the JSON data:\n"
            + "\n".join(notes)
            + "\n\n"
        )

    async def _build_historical_context(self) -> str:
        """
        Build temporal context from recent regime assessments.

        Queries the last 4 assessments and computes per-metric direction
        and streak count. This gives Claude trend awareness beyond the
        single-point snapshot — "VIX has dropped from 35 to 25 over the
        last 2 assessments" vs just "VIX is 25."

        Returns a formatted string to prepend to the interpretation prompt,
        or empty string if insufficient history.
        """
        try:
            rows = await self.cache.db.fetch(
                """
                SELECT id, vix_level, sahm_value, hy_spread, core_pce_yoy,
                       net_liquidity_trillion, fed_funds_rate, nfci,
                       composite_score, stress_index, binding_constraint,
                       primary_regime, assessed_at, full_regime_json
                FROM regime_assessments
                ORDER BY id DESC LIMIT 5
                """
            )
        except Exception as e:
            logger.warning(f"Could not fetch historical context: {e}")
            return ""

        if len(rows) < 2:
            return ""

        # Rows are newest-first. Reverse for chronological order.
        rows = list(reversed(rows))

        # Current is the last row, prior assessments are everything before it
        # But we want to compare the CURRENT computation (not yet stored) against stored history
        # So we use all fetched rows as the historical sequence

        # Define the metrics to track
        metrics = {
            "VIX": {"column": "vix_level", "format": ".1f", "unit": ""},
            "Sahm Rule": {"column": "sahm_value", "format": ".3f", "unit": ""},
            "HY Spread": {"column": "hy_spread", "format": ".2f", "unit": "%"},
            "Core PCE YoY": {"column": "core_pce_yoy", "format": ".2f", "unit": "%"},
            "Net Liquidity": {"column": "net_liquidity_trillion", "format": ".2f", "unit": "T"},
            "Composite Score": {"column": "composite_score", "format": ".3f", "unit": ""},
            "Stress Index": {"column": "stress_index", "format": ".3f", "unit": ""},
            "NFCI": {"column": "nfci", "format": ".3f", "unit": ""},
        }

        context_lines = []
        context_lines.append("HISTORICAL CONTEXT — Recent regime assessment trend (last {} assessments):".format(len(rows)))

        significant_changes = []

        for name, info in metrics.items():
            col = info["column"]
            fmt = info["format"]
            unit = info["unit"]

            # Extract values (skip None)
            values = []
            for r in rows:
                val = r[col]
                if val is not None:
                    values.append(float(val))

            if len(values) < 2:
                continue

            # Compute direction from sequential comparisons
            changes = []
            for i in range(1, len(values)):
                diff = values[i] - values[i-1]
                if abs(diff) < 0.001:  # Essentially flat
                    changes.append(0)
                elif diff > 0:
                    changes.append(1)
                else:
                    changes.append(-1)

            # Compute streak (consecutive same-direction moves from most recent)
            if not changes:
                continue

            latest_dir = changes[-1]
            streak = 0
            for c in reversed(changes):
                if c == latest_dir:
                    streak += 1
                else:
                    break

            # Classify direction
            if latest_dir > 0:
                direction = "rising"
            elif latest_dir < 0:
                direction = "falling"
            else:
                direction = "flat"

            # Format the values as a sequence
            val_strs = [f"{v:{fmt}}{unit}" for v in values]
            sequence = " → ".join(val_strs)

            # Compute total change from first to last
            total_change = values[-1] - values[0]

            line = f"  {name}: {sequence} ({direction}, streak: {streak})"
            context_lines.append(line)

            # Track significant moves for the summary
            if name == "VIX" and abs(total_change) > 3:
                significant_changes.append(
                    f"VIX {'dropped' if total_change < 0 else 'rose'} "
                    f"{abs(total_change):.1f} points over {len(values)-1} assessments"
                )
            elif name == "HY Spread" and abs(total_change) > 0.3:
                significant_changes.append(
                    f"HY spread {'tightened' if total_change < 0 else 'widened'} "
                    f"{abs(total_change):.2f}% over {len(values)-1} assessments"
                )
            elif name == "Sahm Rule" and abs(total_change) > 0.03:
                significant_changes.append(
                    f"Sahm Rule {'improved' if total_change < 0 else 'deteriorated'} "
                    f"from {values[0]:.3f} to {values[-1]:.3f}"
                )
            elif name == "Stress Index" and abs(total_change) > 0.05:
                significant_changes.append(
                    f"Stress {'declined' if total_change < 0 else 'rose'} "
                    f"from {values[0]:.3f} to {values[-1]:.3f}"
                )

        # Binding constraint history
        binding_history = [r["binding_constraint"] for r in rows if r["binding_constraint"]]
        if binding_history:
            unique_bindings = []
            for b in binding_history:
                if not unique_bindings or unique_bindings[-1] != b:
                    unique_bindings.append(b)

            if len(unique_bindings) == 1:
                context_lines.append(f"  Binding constraint: {unique_bindings[0]} (unchanged across all assessments)")
            else:
                context_lines.append(f"  Binding constraint sequence: {' → '.join(unique_bindings)}")

        # Regime history
        regime_history = [r["primary_regime"] for r in rows if r["primary_regime"]]
        if regime_history:
            unique_regimes = []
            for r in regime_history:
                if not unique_regimes or unique_regimes[-1] != r:
                    unique_regimes.append(r)

            if len(unique_regimes) == 1:
                context_lines.append(f"  Regime: {unique_regimes[0]} (stable across all assessments)")
            else:
                context_lines.append(f"  Regime sequence: {' → '.join(unique_regimes)}")

        # SKEW history from full_regime_json (yfinance-sourced, stored in volatility component)
        skew_values = []
        for r in rows:
            try:
                frj = r.get("full_regime_json")
                if frj:
                    if isinstance(frj, str):
                        frj = json.loads(frj)
                    skew_val = frj.get("components", {}).get("volatility", {}).get("skew", {}).get("current")
                    if skew_val is not None:
                        skew_values.append(float(skew_val))
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                pass

        if len(skew_values) >= 2:
            skew_strs = [f"{v:.0f}" for v in skew_values]
            skew_dir = "rising" if skew_values[-1] > skew_values[0] + 2 else "falling" if skew_values[-1] < skew_values[0] - 2 else "stable"
            context_lines.append(f"  SKEW: {' → '.join(skew_strs)} ({skew_dir})")

            if abs(skew_values[-1] - skew_values[0]) > 10:
                significant_changes.append(
                    f"SKEW {'rose' if skew_values[-1] > skew_values[0] else 'fell'} "
                    f"from {skew_values[0]:.0f} to {skew_values[-1]:.0f} — "
                    f"{'institutions increasing' if skew_values[-1] > skew_values[0] else 'institutions reducing'} tail hedging"
                )

        # Summary of significant changes
        if significant_changes:
            context_lines.append(f"\n  NOTABLE CHANGES: {'; '.join(significant_changes)}")

        context_lines.append(
            "\n  Use this trend context to frame your analysis. Note what has CHANGED "
            "since prior assessments, not just where things stand now."
        )

        return "\n".join(context_lines) + "\n\n"

    async def _build_prior_assessment_context(self) -> str:
        """
        Build continuity context from the prior interpretation.

        Fetches the most recent interpretation_json and extracts
        watch_items and risks. This gives Claude analytical continuity:
        "Your last assessment flagged X. Has it materialized or resolved?"

        Only passes watch_items and risks — NOT the full summary or
        key_findings — to prevent error propagation. If Claude fabricated
        a binding shift in the prior interpretation, we don't want it
        echoed forward.

        Returns a formatted string to prepend to the interpretation prompt,
        or empty string if no prior interpretation exists.
        """
        try:
            row = await self.cache.db.fetchrow(
                """
                SELECT interpretation_json, assessed_at
                FROM regime_assessments
                ORDER BY id DESC LIMIT 1
                """
            )
        except Exception as e:
            logger.warning(f"Could not fetch prior interpretation: {e}")
            return ""

        if not row or not row["interpretation_json"]:
            return ""

        try:
            prior = json.loads(row["interpretation_json"])
        except (json.JSONDecodeError, TypeError):
            return ""

        watch_items = prior.get("watch_items", [])
        risks = prior.get("risks", [])

        # Skip if prior had empty fields (parsing failure)
        if not watch_items and not risks:
            return ""

        # Calculate age
        assessed_at = row.get("assessed_at")
        age_str = ""
        if assessed_at:
            try:
                from datetime import timezone
                age = datetime.now(timezone.utc) - assessed_at
                hours = age.total_seconds() / 3600
                if hours < 1:
                    age_str = f" ({int(age.total_seconds()/60)} minutes ago)"
                elif hours < 24:
                    age_str = f" ({hours:.1f} hours ago)"
                else:
                    age_str = f" ({hours/24:.1f} days ago)"
            except Exception:
                pass

        lines = []
        lines.append(f"YOUR PREVIOUS ASSESSMENT{age_str} — for analytical continuity:")

        if risks:
            lines.append("  Previous risks flagged:")
            for i, risk in enumerate(risks, 1):
                if isinstance(risk, str):
                    lines.append(f"    {i}. {risk}")

        if watch_items:
            lines.append("  Previous watch items:")
            for i, item in enumerate(watch_items, 1):
                if isinstance(item, str):
                    lines.append(f"    {i}. {item}")

        lines.append(
            "\n  CONTINUITY INSTRUCTIONS: Note whether each previous risk/watch item has "
            "materialized, worsened, improved, or resolved. If conditions are similar to "
            "your last assessment, focus on what has CHANGED or EVOLVED rather than "
            "repeating the same analysis. If a previous watch item has been resolved, "
            "acknowledge it and replace with a new one."
        )

        return "\n".join(lines) + "\n\n"

    def _validate_interpretation(self, interpretation: dict, regime_assessment, live_prices: dict = None) -> dict:
        """
        Post-hoc validation of Claude's interpretation against computed data.

        Compares Claude's claimed values against the authoritative computed
        regime data. Auto-corrects fields where the computed value is
        definitively correct (binding_constraint, forward_bias). Flags
        fields that are suspicious but may have valid explanations (Claude
        may use live prices from web search instead of FRED values).

        Produces a _validation audit trail stored with the interpretation
        for tracking fabrication frequency over time.

        This is pure Python comparison — zero API calls, <1ms execution.

        Args:
            interpretation: The parsed interpretation dict from _interpret()
            regime_assessment: The computed RegimeAssessment object
            live_prices: Optional dict of yfinance live prices for dual-source checking

        Returns:
            The interpretation dict, potentially modified, with _validation field added
        """
        if not interpretation or not interpretation.get("summary"):
            return interpretation

        lp = live_prices or {}
        kl = regime_assessment.key_levels or {}
        traj = regime_assessment.trajectory or {}
        forward_bias_data = traj.get("forward_bias", {})
        metrics = interpretation.get("key_metrics", {})

        corrections = []
        warnings = []
        passed = 0

        # === TIER 1: Auto-correct (computed value is definitively correct) ===

        # 1. Binding constraint
        computed_binding = regime_assessment.binding_constraint
        claude_binding = interpretation.get("binding_constraint", "")
        if computed_binding and claude_binding:
            # Fuzzy match: Claude writes "Inflation persistence", computed says "inflation"
            if computed_binding.lower() not in claude_binding.lower():
                corrections.append({
                    "field": "binding_constraint",
                    "claude_value": claude_binding,
                    "computed_value": computed_binding,
                    "action": "corrected",
                })
                interpretation["binding_constraint"] = computed_binding
                logger.warning(
                    f"VALIDATION: Corrected binding_constraint from "
                    f"'{claude_binding}' to '{computed_binding}'"
                )
            else:
                passed += 1

        # 2. Forward bias in key_metrics
        computed_bias = forward_bias_data.get("bias", "neutral")
        claude_bias = metrics.get("forward_bias", "")
        if claude_bias and computed_bias:
            if claude_bias.lower().strip() != computed_bias.lower().strip():
                corrections.append({
                    "field": "key_metrics.forward_bias",
                    "claude_value": claude_bias,
                    "computed_value": computed_bias,
                    "action": "corrected",
                })
                metrics["forward_bias"] = computed_bias
                logger.warning(
                    f"VALIDATION: Corrected forward_bias from "
                    f"'{claude_bias}' to '{computed_bias}'"
                )
            else:
                passed += 1

        # 3. Composite score in key_metrics
        computed_composite = regime_assessment.composite_score
        claude_composite = metrics.get("composite_score")
        if claude_composite is not None and computed_composite is not None:
            try:
                claude_val = float(claude_composite)
                if abs(claude_val - computed_composite) > 0.02:
                    corrections.append({
                        "field": "key_metrics.composite_score",
                        "claude_value": claude_val,
                        "computed_value": round(computed_composite, 3),
                        "action": "corrected",
                    })
                    metrics["composite_score"] = round(computed_composite, 3)
                    logger.warning(
                        f"VALIDATION: Corrected composite_score from "
                        f"{claude_val} to {computed_composite:.3f}"
                    )
                else:
                    passed += 1
            except (ValueError, TypeError):
                warnings.append({
                    "field": "key_metrics.composite_score",
                    "claude_value": str(claude_composite),
                    "note": "Non-numeric value",
                    "action": "flagged",
                })

        # === TIER 2: Tolerance checks with dual-source validation ===

        # Helper for dual-source checking
        def _check_metric(field_name, claude_val, fred_val, live_val, tolerance, unit=""):
            nonlocal passed
            if claude_val is None:
                return  # Claude didn't provide this metric
            try:
                cv = float(claude_val)
            except (ValueError, TypeError):
                warnings.append({
                    "field": f"key_metrics.{field_name}",
                    "claude_value": str(claude_val),
                    "note": "Non-numeric value",
                    "action": "flagged",
                })
                return

            # Check against FRED value
            fred_match = fred_val is not None and abs(cv - fred_val) <= tolerance
            # Check against live value
            live_match = live_val is not None and abs(cv - live_val) <= tolerance

            if fred_match or live_match:
                passed += 1
            else:
                note_parts = []
                if fred_val is not None:
                    note_parts.append(f"FRED={fred_val}{unit}")
                if live_val is not None:
                    note_parts.append(f"live={live_val}{unit}")
                warnings.append({
                    "field": f"key_metrics.{field_name}",
                    "claude_value": cv,
                    "computed_value": fred_val,
                    "live_value": live_val,
                    "tolerance": tolerance,
                    "action": "flagged",
                    "note": f"Does not match {' or '.join(note_parts)} within tolerance {tolerance}",
                })
                logger.warning(
                    f"VALIDATION: key_metrics.{field_name}={cv} does not match "
                    f"FRED={fred_val} or live={live_val} (tolerance={tolerance})"
                )

        # VIX — accept either FRED or live
        _check_metric("vix", metrics.get("vix"),
                      kl.get("vix"), lp.get("vix"), tolerance=2.0)

        # Core PCE YoY — FRED only (no live equivalent)
        _check_metric("core_pce_yoy", metrics.get("core_pce_yoy"),
                      kl.get("core_pce_yoy"), None, tolerance=0.05, unit="%")

        # Core PCE 3m annualized — FRED only
        _check_metric("core_pce_3m_annualized", metrics.get("core_pce_3m_annualized"),
                      kl.get("core_pce_3m_ann"), None, tolerance=0.05, unit="%")

        # HY Spread — FRED only
        _check_metric("hy_spread", metrics.get("hy_spread"),
                      kl.get("hy_spread"), None, tolerance=0.1, unit="%")

        # Oil WTI — accept either FRED or live
        fred_oil = None
        oil_dict = regime_assessment.cross_asset if isinstance(regime_assessment.cross_asset, dict) else {}
        if isinstance(oil_dict, dict):
            fred_oil = oil_dict.get("oil", {}).get("wti_price") if isinstance(oil_dict.get("oil"), dict) else None
        if fred_oil is None:
            fred_oil = kl.get("oil_wti")
        _check_metric("oil_wti", metrics.get("oil_wti"),
                      fred_oil, lp.get("wti"), tolerance=3.0, unit="$")

        # Fed Funds — FRED only
        _check_metric("fed_funds", metrics.get("fed_funds"),
                      kl.get("fed_funds"), None, tolerance=0.05, unit="%")

        # Sahm — FRED only, tight tolerance
        _check_metric("sahm_value", metrics.get("sahm_value"),
                      kl.get("sahm_value"), None, tolerance=0.005)

        # Net Liquidity — FRED only
        computed_net_liq_t = None
        net_liq_raw = kl.get("net_liquidity")
        if net_liq_raw is not None:
            computed_net_liq_t = net_liq_raw / 1_000_000 if net_liq_raw > 1000 else net_liq_raw
        _check_metric("net_liquidity_trillion", metrics.get("net_liquidity_trillion"),
                      computed_net_liq_t, None, tolerance=0.05, unit="T")

        # === TIER 3: Qualitative check — regime alignment ===

        computed_regime = regime_assessment.primary_regime.value
        claude_regime = interpretation.get("macro_regime", "")
        if computed_regime and claude_regime:
            if computed_regime.lower() in claude_regime.lower():
                passed += 1
            else:
                warnings.append({
                    "field": "macro_regime",
                    "claude_value": claude_regime,
                    "computed_value": computed_regime,
                    "action": "flagged",
                    "note": f"Computed regime '{computed_regime}' not found in Claude's regime description",
                })
                logger.warning(
                    f"VALIDATION: macro_regime '{claude_regime}' does not contain "
                    f"computed regime '{computed_regime}'"
                )

        # === TIER 3 (continued): Structural checks on new fields ===

        # Scenarios should have required fields
        for i, scenario in enumerate(interpretation.get("scenarios", [])):
            if isinstance(scenario, dict):
                if not scenario.get("probability"):
                    warnings.append({
                        "field": f"scenarios[{i}]",
                        "note": "Missing probability assessment",
                        "action": "flagged",
                    })
                if not scenario.get("trigger"):
                    warnings.append({
                        "field": f"scenarios[{i}]",
                        "note": "Missing trigger condition",
                        "action": "flagged",
                    })

        # Catalysts should have dates
        for i, catalyst in enumerate(interpretation.get("catalysts", [])):
            if isinstance(catalyst, dict):
                if not catalyst.get("date"):
                    warnings.append({
                        "field": f"catalysts[{i}]",
                        "note": "Missing date",
                        "action": "flagged",
                    })

        # === Build validation summary ===
        from datetime import datetime, timezone
        interpretation["_validation"] = {
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "corrections": corrections,
            "warnings": warnings,
            "passed": passed,
            "corrected": len(corrections),
            "warned": len(warnings),
            "total_checks": passed + len(corrections) + len(warnings),
        }

        if corrections:
            logger.info(
                f"VALIDATION COMPLETE: {passed} passed, {len(corrections)} corrected, "
                f"{len(warnings)} warned. Corrections: "
                f"{', '.join(c['field'] for c in corrections)}"
            )
        else:
            logger.info(
                f"VALIDATION COMPLETE: {passed} passed, 0 corrected, "
                f"{len(warnings)} warned."
            )

        return interpretation

    def _regime_to_direction(self, regime: MarketRegime, score: float) -> SignalDirection:
        """
        Map regime assessment to a signal direction.

        Uses the trajectory layer's forward_bias when available (validated by backtest).
        Falls back to NEUTRAL when trajectory signals aren't available.

        The composite score is NOT used for direction because backtesting proved
        it has -0.096 correlation with forward returns (anti-predictive).
        """
        # Use trajectory forward_bias if available
        trajectory = getattr(self, '_last_trajectory', None)
        if trajectory is not None:
            bias = trajectory.forward_bias
            confidence = trajectory.forward_bias_confidence

            if bias == "constructive" and confidence == "high":
                return SignalDirection.BULLISH
            elif bias == "constructive":
                return SignalDirection.SLIGHTLY_BULLISH
            elif bias == "cautious" and confidence == "high":
                return SignalDirection.BEARISH
            elif bias == "cautious":
                return SignalDirection.SLIGHTLY_BEARISH
            else:
                return SignalDirection.NEUTRAL

        # Fallback: crisis override only (stress > 0.8 is structurally valid)
        if regime == MarketRegime.CRISIS:
            return SignalDirection.STRONGLY_BEARISH

        return SignalDirection.NEUTRAL

    def _score_to_confidence(self, score: float) -> ConfidenceLevel:
        """Map a 0-1 confidence score to a ConfidenceLevel enum."""
        if score >= 0.85:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.7:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MODERATE
        elif score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    async def _persist_opinion(self, regime, query: AgentQuery, interpretation: dict):
        """Store the regime assessment and opinion in PostgreSQL for history."""
        import uuid

        # Strip citation tags from interpretation before storing
        clean_interpretation = {}
        for key, value in interpretation.items():
            if isinstance(value, str):
                clean_interpretation[key] = _strip_cite_tags(value)
            elif isinstance(value, list):
                clean_interpretation[key] = [
                    _strip_cite_tags(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                clean_interpretation[key] = value

        # Store in regime_assessments
        key_levels = regime.key_levels
        await self.cache.db.execute(
            """
            INSERT INTO regime_assessments (
                primary_regime, cycle_phase, liquidity_regime, volatility_regime, inflation_regime,
                growth_cycle_score, monetary_liquidity_score, inflation_score, market_signals_score,
                weight_growth, weight_monetary, weight_inflation, weight_market,
                composite_score, confidence, stress_index, binding_constraint,
                vix_level, fed_funds_rate, net_liquidity_trillion, spread_2s10s,
                core_pce_yoy, unemployment_rate, sahm_value, ism_manufacturing,
                hy_spread, nfci,
                full_regime_json, interpretation_json, data_quality_json
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10, $11, $12, $13,
                $14, $15, $16, $17,
                $18, $19, $20, $21,
                $22, $23, $24, $25,
                $26, $27,
                $28, $29, $30
            )
            """,
            regime.primary_regime.value, regime.cycle_phase,
            regime.liquidity_regime, regime.volatility_regime, regime.inflation_regime,
            regime.growth_cycle_score, regime.monetary_liquidity_score,
            regime.inflation_score, regime.market_signals_score,
            regime.weight_growth, regime.weight_monetary,
            regime.weight_inflation, regime.weight_market,
            regime.composite_score, regime.confidence,
            regime.stress_index, regime.binding_constraint,
            key_levels.get("vix"), key_levels.get("fed_funds"),
            (key_levels.get("net_liquidity") or 0) / 1_000_000, key_levels.get("spread_2s10s"),
            key_levels.get("core_pce_yoy"), key_levels.get("unemployment"),
            key_levels.get("sahm_value"), key_levels.get("ism_manufacturing"),
            key_levels.get("hy_spread"), key_levels.get("nfci"),
            json.dumps(regime.to_dict(), default=str),  # full_regime_json — ALL 200+ fields
            json.dumps(clean_interpretation, default=str),     # interpretation_json — Claude's full analysis
            json.dumps(
                getattr(regime, '_quality_report', None).to_dict()
                if getattr(regime, '_quality_report', None) else None,
                default=str,
            ),  # data_quality_json — quality report at assessment time
        )

        # Store in agent_opinions
        opinion_id = str(uuid.uuid4())
        direction = self._regime_to_direction(regime.primary_regime, regime.composite_score)
        confidence = self._score_to_confidence(regime.confidence)

        await self.cache.db.execute(
            """
            INSERT INTO agent_opinions (
                opinion_id, agent_name, agent_layer,
                direction, confidence, regime,
                summary, key_findings, data_points,
                methodology, risks_to_view, watch_items,
                data_freshness, computation_ms,
                query_question, query_context
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
            )
            """,
            opinion_id, self.name, self.layer.value,
            direction.value, confidence.value, regime.primary_regime.value,
            clean_interpretation.get("summary", ""),
            json.dumps(clean_interpretation.get("key_findings", [])),
            json.dumps(regime.to_dict(), default=str),
            f"Hierarchical regime model. Binding constraint: {regime.binding_constraint}.",
            json.dumps(clean_interpretation.get("risks", [])),
            json.dumps(clean_interpretation.get("watch_items", [])),
            datetime.now(timezone.utc), 0,
            query.question, json.dumps(query.context, default=str),
        )

        logger.info(f"Persisted opinion {opinion_id} and regime assessment to database")
