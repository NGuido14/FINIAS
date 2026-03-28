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
from finias.agents.macro_strategist.computations.yield_curve import analyze_yield_curve
from finias.agents.macro_strategist.computations.volatility import analyze_volatility
from finias.agents.macro_strategist.computations.breadth import analyze_breadth
from finias.agents.macro_strategist.computations.cross_asset import analyze_cross_assets
from finias.agents.macro_strategist.computations.regime import detect_regime
from finias.agents.macro_strategist.computations.monetary_policy import analyze_monetary_policy
from finias.agents.macro_strategist.computations.business_cycle import analyze_business_cycle
from finias.agents.macro_strategist.computations.inflation import analyze_inflation
from finias.agents.macro_strategist.prompts.interpretation import MACRO_INTERPRETATION_PROMPT

logger = logging.getLogger("finias.agent.macro_strategist")


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
            "NFCI", "ANFCI", "STLFSI2",
            "TOTBKCR", "TOTALSL", "M2SL",
            # Volatility
            "VIXCLS",
            # Cross-Asset
            "BAMLH0A0HYM2", "DTWEXBGS",
            # Inflation
            "CPIAUCSL", "CPILFESL", "CUSR0000SEHC", "CUSR0000SAS",
            "PCEPI", "PCEPILFE",
            "STICKCPIM157SFRBATL", "FLEXCPIM157SFRBATL",
            "PCETRIM12M159SFRBDAL",
            "T5YIE", "T10YIE", "T5YIFR",
            "PPIACO", "CES0500000003", "DCOILWTICO",
            # Business Cycle
            "USSLIND", "UNRATE", "U6RATE", "ICSA", "CCSA",
            "JTSJOL", "JTSQUR",
            "TEMPHELPS", "AWHAETP",
            "PERMIT", "HOUST", "RSAFS",
            "UMCSENT", "INDPRO", "TCU", "CFNAI",
            "PI", "DGORDER", "PAYEMS",
            "GACDISA066MSFRBPHI", "CIVPART", "LNS11300060",
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

        # === Run ALL computation modules ===

        # 1. Yield Curve (enhanced)
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

        # 2. Volatility (enhanced)
        vol_analysis = analyze_volatility(
            vix_series=fred_data.get("VIXCLS", []),
            spx_prices=spx_prices,
        )
        # Add correlation if we have sector data
        if len(sector_prices) >= 5:
            from finias.agents.macro_strategist.computations.volatility import (
                compute_sector_correlation, classify_correlation_regime
            )
            avg_corr = compute_sector_correlation(sector_prices)
            vol_analysis.sector_correlation = avg_corr
            vol_analysis.correlation_regime = classify_correlation_regime(avg_corr)

        # 3. Breadth
        breadth_analysis = analyze_breadth(spx_prices=spx_prices)

        # 4. Cross-Asset
        ca_analysis = analyze_cross_assets(
            dxy_series=fred_data.get("DTWEXBGS", []),
            hy_spread_series=fred_data.get("BAMLH0A0HYM2", []),
            breakeven_5y=fred_data.get("T5YIE", []),
            breakeven_10y=fred_data.get("T10YIE", []),
        )

        # 5. Monetary Policy (NEW)
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
            stress_series=fred_data.get("STLFSI2", []),
            bank_credit=fred_data.get("TOTBKCR", []),
            consumer_credit=fred_data.get("TOTALSL", []),
            m2_series=fred_data.get("M2SL", []),
        )

        # 6. Business Cycle (NEW)
        cycle_analysis = analyze_business_cycle(
            lei_series=fred_data.get("USSLIND", []),
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
            philly_fed=fred_data.get("GACDISA066MSFRBPHI", []),
        )

        # 7. Inflation (NEW)
        infl_analysis = analyze_inflation(
            cpi_all=fred_data.get("CPIAUCSL", []),
            cpi_core=fred_data.get("CPILFESL", []),
            cpi_shelter=fred_data.get("CUSR0000SEHC", []),
            cpi_services=fred_data.get("CUSR0000SAS", []),
            pce=fred_data.get("PCEPI", []),
            core_pce=fred_data.get("PCEPILFE", []),
            sticky_cpi=fred_data.get("STICKCPIM157SFRBATL", []),
            flexible_cpi=fred_data.get("FLEXCPIM157SFRBATL", []),
            trimmed_mean=fred_data.get("PCETRIM12M159SFRBDAL", []),
            breakeven_5y=fred_data.get("T5YIE", []),
            breakeven_10y=fred_data.get("T10YIE", []),
            forward_5y5y=fred_data.get("T5YIFR", []),
            ppi=fred_data.get("PPIACO", []),
            ahe=fred_data.get("CES0500000003", []),
            oil=fred_data.get("DCOILWTICO", []),
        )

        # === Detect regime with full hierarchy ===
        regime_assessment = detect_regime(
            yield_curve=yc_analysis,
            volatility=vol_analysis,
            breadth=breadth_analysis,
            cross_asset=ca_analysis,
            monetary_policy=mp_analysis,
            business_cycle=cycle_analysis,
            inflation_analysis=infl_analysis,
        )

        # === Claude interpretation ===
        interpretation = await self._interpret(regime_assessment, query.question)

        # === Publish to shared state ===
        await self.state.set_regime(regime_assessment.to_dict())
        await self.state.publish_opinion(self.name, regime_assessment.to_dict())

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
        """
        settings = get_settings()

        regime_data = json.dumps(regime.to_dict(), indent=2, default=str)

        prompt = MACRO_INTERPRETATION_PROMPT.format(
            regime_data=regime_data,
            question=question,
        )

        response = await self._client.messages.create(
            model=settings.claude_model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse Claude's response
        text = response.content[0].text

        # Claude returns JSON with summary, key_findings, risks, watch_items
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: use the raw text as summary
            result = {
                "summary": text,
                "key_findings": [],
                "risks": [],
                "watch_items": [],
            }

        return result

    def _regime_to_direction(self, regime: MarketRegime, score: float) -> SignalDirection:
        """Map regime to a signal direction."""
        mapping = {
            MarketRegime.RISK_ON: SignalDirection.BULLISH,
            MarketRegime.RISK_OFF: SignalDirection.BEARISH,
            MarketRegime.TRANSITION: SignalDirection.NEUTRAL,
            MarketRegime.CRISIS: SignalDirection.STRONGLY_BEARISH,
            MarketRegime.LOW_VOLATILITY: SignalDirection.SLIGHTLY_BULLISH,
            MarketRegime.HIGH_VOLATILITY: SignalDirection.SLIGHTLY_BEARISH,
        }
        base = mapping.get(regime, SignalDirection.NEUTRAL)

        # Adjust intensity based on composite score
        if abs(score) > 0.6:
            if base == SignalDirection.BULLISH:
                return SignalDirection.STRONGLY_BULLISH
            elif base == SignalDirection.BEARISH:
                return SignalDirection.STRONGLY_BEARISH

        return base

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
