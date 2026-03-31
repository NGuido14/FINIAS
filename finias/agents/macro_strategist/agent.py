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
from finias.agents.macro_strategist.prompts.interpretation import MACRO_INTERPRETATION_PROMPT

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
    (JSON-LD, API responses, etc.). This function looks for a JSON block
    containing the expected interpretation keys rather than naively taking
    the outermost {}.

    Falls back to the outermost {} approach if targeted extraction fails.
    """
    # Strategy 1: Find JSON blocks and look for one with expected keys
    expected_keys = {"summary", "key_findings", "risks", "watch_items"}

    # Try to find JSON blocks by matching balanced braces
    depth = 0
    start_positions = []
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start_positions.append(i)
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start_positions:
                candidate = text[start_positions[-1]:i+1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and expected_keys.intersection(parsed.keys()):
                        return parsed
                except json.JSONDecodeError:
                    continue

    # Strategy 2: Fallback to outermost {} (original approach)
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        try:
            return json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            pass

    # Strategy 3: Return empty dict (caller handles defaults)
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
            "PPIACO", "CES0500000003", "DCOILWTICO",
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
            vol_analysis = analyze_volatility(
                vix_series=fred_data.get("VIXCLS", []),
                spx_prices=spx_prices,
                vix3m_series=fred_data.get("VXVCLS", []),
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

        trajectory = compute_trajectory(
            regime_assessment=regime_assessment,
            fed_target_upper=fred_data.get("DFEDTARU", []),
            prior_regime_assessment=prior_regime,
        )
        regime_assessment.trajectory = trajectory.to_dict()
        self._last_trajectory = trajectory

        # === Claude interpretation ===
        interpretation = await self._interpret(regime_assessment, query.question)

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
            data_points=regime_assessment.to_dict(),
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

    async def _interpret(self, regime: Any, question: str) -> dict[str, Any]:
        """
        Ask Claude to interpret the regime assessment.

        This is the 10% of the work that requires genuine intelligence.
        The Python did all the math. Claude explains what it means.

        Claude has access to web search to:
        - Research WHY indicators are moving (e.g., oil up 36% → Iran conflict)
        - Verify stale data (e.g., check current GDPNow from Atlanta Fed)
        - Get current market narrative and geopolitical context
        - Cross-check any data point that seems outdated
        """
        settings = get_settings()

        regime_data = json.dumps(regime.to_dict(), indent=2, default=str)

        # Build plain-English notes for fields Claude tends to misinterpret
        data_notes = self._build_data_notes(regime)
        date_context = f"TODAY'S DATE: {date.today().isoformat()}. All analysis and forward-looking statements should reference dates relative to today.\n\n"

        prompt = date_context + data_notes + MACRO_INTERPRETATION_PROMPT.format(
            regime_data=regime_data,
            question=question,
        )

        from finias.core.utils.retry import retry_claude_call

        response = await retry_claude_call(
            lambda: self._client.messages.create(
                model=settings.claude_model,
                max_tokens=3000,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                }],
                messages=[{"role": "user", "content": prompt}],
            )
        )

        # Extract text from potentially multi-block response
        # When Claude uses web search, response contains tool_use and tool_result
        # blocks alongside text blocks. We need all text blocks.
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        text = "\n".join(text_parts)

        # Claude returns JSON with macro_regime, binding_constraint, summary, key_findings, risks, watch_items
        try:
            result = _extract_interpretation_json(text)
            if not result:
                # Try parsing the full text directly before falling back
                try:
                    text_stripped = text.strip()
                    if text_stripped.startswith("{"):
                        result = json.loads(text_stripped)
                        if not isinstance(result, dict) or "summary" not in result:
                            result = {}
                except (json.JSONDecodeError, ValueError):
                    result = {}

                if not result:
                    raise json.JSONDecodeError("No valid interpretation JSON found", text, 0)

            # Ensure all expected keys exist
            result.setdefault("summary", "")
            result.setdefault("key_findings", [])
            result.setdefault("risks", [])
            result.setdefault("watch_items", [])
            result.setdefault("macro_regime", "")
            result.setdefault("binding_constraint", "")
            result.setdefault("key_metrics", {})

            # Prepend binding constraint to summary for downstream visibility
            if result["binding_constraint"] and result["binding_constraint"] not in result["summary"]:
                result["summary"] = (
                    f"Binding constraint: {result['binding_constraint']}. "
                    + result["summary"]
                )
        except json.JSONDecodeError:
            result = {
                "summary": text,
                "key_findings": [],
                "risks": [],
                "watch_items": [],
                "macro_regime": "",
                "binding_constraint": "",
                "key_metrics": {},
            }

            # Secondary attempt: the raw text itself might be valid JSON
            # (happens when web search content confuses the primary parser)
            try:
                text_stripped = text.strip()
                if text_stripped.startswith("{"):
                    parsed = json.loads(text_stripped)
                    if isinstance(parsed, dict) and "summary" in parsed:
                        result = parsed
                        result.setdefault("key_findings", [])
                        result.setdefault("risks", [])
                        result.setdefault("watch_items", [])
                        result.setdefault("macro_regime", "")
                        result.setdefault("binding_constraint", "")
                        result.setdefault("key_metrics", {})
            except (json.JSONDecodeError, ValueError):
                pass  # Keep the fallback result

        return result

    def _build_data_notes(self, regime) -> str:
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
        stress_sig = signals.get("stress_contrarian", "neutral")
        shifted = signals.get("binding_shifted", False)
        shift_dir = signals.get("shift_direction", "none")

        if bias != "neutral" or infl_traj != "unknown":
            parts = []
            if infl_traj != "unknown":
                parts.append(f"inflation {infl_traj}")
            if stress_sig != "neutral":
                parts.append(f"stress {stress_sig}")
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
            oil_val = ca.get("oil", {}).get("price")
            if oil_val is not None:
                stale_metrics.append(
                    f"Oil/WTI (computed: ${oil_val:.2f} — NOTE: this is WTI, not Brent; "
                    f"in geopolitical supply disruptions Brent can trade $10-30 above WTI)"
                )
            else:
                stale_metrics.append("Oil/WTI (computed value unavailable)")

            stale_metrics.append("SPY level")

            notes.append(
                f"- STALE DATA WARNING — High velocity environment means computed values may be "
                f"materially outdated. USE WEB SEARCH to verify current levels before referencing "
                f"these in your analysis: {', '.join(stale_metrics)}. "
                f"Cite the web-searched values alongside computed values when they differ materially."
            )

        # --- Scenario Triggers ---
        triggers = traj.get("scenario_triggers", [])
        critical_triggers = [t for t in triggers if t.get("severity") == "critical" and t.get("distance", 999) < 5]
        if critical_triggers:
            for t in critical_triggers:
                notes.append(
                    f"- CRITICAL TRIGGER NEARBY: {t['metric']} at {t['current']}, threshold {t['threshold']} "
                    f"(distance: {t['distance']}). If breached: {t['consequence']}."
                )

        # --- Cross-Asset Correlations ---
        corr_data = ca.get("correlations")
        if corr_data and corr_data.get("pairs"):
            corr_notes = _generate_correlation_notes_from_dict(corr_data)
            notes.extend(corr_notes)

        if not notes:
            return ""

        return (
            "IMPORTANT DATA NOTES — Read these BEFORE interpreting the JSON data:\n"
            + "\n".join(notes)
            + "\n\n"
        )

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
                full_regime_json, interpretation_json
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10, $11, $12, $13,
                $14, $15, $16, $17,
                $18, $19, $20, $21,
                $22, $23, $24, $25,
                $26, $27,
                $28, $29
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
