"""
Hierarchical Market Regime Detection Model

Synthesizes ALL macro analyses into a multi-dimensional regime classification.

Architecture:
    Raw Data (100+ series)
        ↓
    Domain Computations (8 modules)
        ↓
    Category Scores (4 categories, each -1 to +1):
        1. GROWTH & CYCLE      ← Business Cycle + LEI + Labor signals
        2. MONETARY & LIQUIDITY ← Fed Policy + Net Liquidity + Financial Conditions
        3. INFLATION            ← Inflation Dynamics (standalone)
        4. MARKET SIGNALS       ← Volatility + Breadth + Cross-Asset + Yield Curve
        ↓
    Dynamic Weighting (based on what's driving markets NOW)
        ↓
    Regime Assessment (multi-dimensional classification with confidence)
        ↓
    AgentOpinion (what every other agent sees)

Each category compresses its domains into a single score.
The final regime uses only 4 weighted inputs — not 8.
This prevents signal dilution while maintaining depth.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone
import numpy as np

from finias.core.agents.models import MarketRegime

# Import all analysis types
from finias.agents.macro_strategist.computations.yield_curve import YieldCurveAnalysis
from finias.agents.macro_strategist.computations.volatility import VolatilityAnalysis
from finias.agents.macro_strategist.computations.breadth import BreadthAnalysis
from finias.agents.macro_strategist.computations.cross_asset import CrossAssetAnalysis
from finias.agents.macro_strategist.computations.monetary_policy import MonetaryPolicyAnalysis
from finias.agents.macro_strategist.computations.business_cycle import BusinessCycleAnalysis
from finias.agents.macro_strategist.computations.inflation import InflationAnalysis


@dataclass
class RegimeAssessment:
    """Multi-dimensional regime assessment with hierarchical scoring."""

    # Multi-dimensional regime classification
    primary_regime: MarketRegime                    # risk_on, risk_off, transition, crisis
    cycle_phase: str = "unknown"                    # early_cycle, mid_cycle, late_cycle, recession
    liquidity_regime: str = "unknown"               # ample, adequate, tightening, scarce
    volatility_regime: str = "unknown"              # compressed, normal, elevated, extreme
    inflation_regime: str = "unknown"               # disinflation, stable, rising, stagflation

    # Category scores (-1 to +1)
    growth_cycle_score: float = 0.0
    monetary_liquidity_score: float = 0.0
    inflation_score: float = 0.0
    market_signals_score: float = 0.0

    # Dynamic weights (sum to 1.0)
    weight_growth: float = 0.25
    weight_monetary: float = 0.25
    weight_inflation: float = 0.25
    weight_market: float = 0.25

    # Composite
    composite_score: float = 0.0
    confidence: float = 0.5
    stress_index: float = 0.0
    binding_constraint: str = "none"                # Which category matters most

    # Component analyses (full detail for Director/Claude interpretation)
    yield_curve: dict = field(default_factory=dict)
    volatility: dict = field(default_factory=dict)
    breadth: dict = field(default_factory=dict)
    cross_asset: dict = field(default_factory=dict)
    monetary_policy: dict = field(default_factory=dict)
    business_cycle: dict = field(default_factory=dict)
    inflation: dict = field(default_factory=dict)

    # Key levels for quick reference
    key_levels: dict = field(default_factory=dict)

    assessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "regime": {
                "primary": self.primary_regime.value,
                "cycle_phase": self.cycle_phase,
                "liquidity": self.liquidity_regime,
                "volatility": self.volatility_regime,
                "inflation": self.inflation_regime,
            },
            "category_scores": {
                "growth_cycle": self.growth_cycle_score,
                "monetary_liquidity": self.monetary_liquidity_score,
                "inflation": self.inflation_score,
                "market_signals": self.market_signals_score,
            },
            "weights": {
                "growth": self.weight_growth,
                "monetary": self.weight_monetary,
                "inflation": self.weight_inflation,
                "market": self.weight_market,
            },
            "composite_score": self.composite_score,
            "confidence": self.confidence,
            "stress_index": self.stress_index,
            "binding_constraint": self.binding_constraint,
            "key_levels": self.key_levels,
            "components": {
                "yield_curve": self.yield_curve,
                "volatility": self.volatility,
                "breadth": self.breadth,
                "cross_asset": self.cross_asset,
                "monetary_policy": self.monetary_policy,
                "business_cycle": self.business_cycle,
                "inflation": self.inflation,
            },
            "assessed_at": self.assessed_at.isoformat(),
        }


# Default weights — used when dynamic weighting data isn't available yet
DEFAULT_WEIGHTS = {
    "growth": 0.25,
    "monetary": 0.30,
    "inflation": 0.20,
    "market": 0.25,
}


def detect_regime(
    yield_curve: YieldCurveAnalysis,
    volatility: VolatilityAnalysis,
    breadth: BreadthAnalysis,
    cross_asset: CrossAssetAnalysis,
    monetary_policy: Optional[MonetaryPolicyAnalysis] = None,
    business_cycle: Optional[BusinessCycleAnalysis] = None,
    inflation_analysis: Optional[InflationAnalysis] = None,
    spx_returns: Optional[np.ndarray] = None,
    historical_category_scores: Optional[dict] = None,
) -> RegimeAssessment:
    """
    Hierarchical regime detection.

    Step 1: Compute 4 category scores from domain analyses
    Step 2: Determine dynamic weights (or use defaults)
    Step 3: Compute weighted composite
    Step 4: Classify regime with confidence
    Step 5: Compute stress index
    Step 6: Identify binding constraint
    """

    # === Step 1: Category Scores ===

    # Category 1: Growth & Cycle
    growth_score = _compute_growth_cycle_score(business_cycle, yield_curve)

    # Category 2: Monetary & Liquidity
    monetary_score = _compute_monetary_liquidity_score(monetary_policy, yield_curve)

    # Category 3: Inflation
    infl_score = _compute_inflation_category_score(inflation_analysis)

    # Category 4: Market Signals
    market_score = _compute_market_signals_score(volatility, breadth, cross_asset, yield_curve)

    # === Step 2: Dynamic Weights ===
    weights = _compute_dynamic_weights(
        growth_score, monetary_score, infl_score, market_score,
        spx_returns, historical_category_scores
    )

    # === Step 3: Weighted Composite ===
    composite = (
        weights["growth"] * growth_score +
        weights["monetary"] * monetary_score +
        weights["inflation"] * infl_score +
        weights["market"] * market_score
    )

    # === Step 4: Classify ===
    stress = _compute_stress_index(
        volatility, cross_asset, monetary_policy, business_cycle, yield_curve
    )
    primary_regime, confidence = _classify_primary_regime(composite, stress, volatility)

    # === Step 5: Binding Constraint ===
    binding = _identify_binding_constraint(
        growth_score, monetary_score, infl_score, market_score, weights
    )

    # === Step 6: Sub-regime classifications ===
    cycle = business_cycle.cycle_phase if business_cycle else "unknown"
    liq = monetary_policy.liquidity_regime if monetary_policy else "unknown"
    vol_regime = volatility.vol_regime if hasattr(volatility, 'vol_regime') else "unknown"
    infl_regime = inflation_analysis.inflation_regime if inflation_analysis else "unknown"

    # === Key Levels ===
    key_levels = _extract_key_levels(
        volatility, yield_curve, cross_asset,
        monetary_policy, business_cycle, inflation_analysis
    )

    return RegimeAssessment(
        primary_regime=primary_regime,
        cycle_phase=cycle,
        liquidity_regime=liq,
        volatility_regime=vol_regime,
        inflation_regime=infl_regime,
        growth_cycle_score=growth_score,
        monetary_liquidity_score=monetary_score,
        inflation_score=infl_score,
        market_signals_score=market_score,
        weight_growth=weights["growth"],
        weight_monetary=weights["monetary"],
        weight_inflation=weights["inflation"],
        weight_market=weights["market"],
        composite_score=composite,
        confidence=confidence,
        stress_index=stress,
        binding_constraint=binding,
        yield_curve=yield_curve.to_dict(),
        volatility=volatility.to_dict(),
        breadth=breadth.to_dict(),
        cross_asset=cross_asset.to_dict(),
        monetary_policy=monetary_policy.to_dict() if monetary_policy else {},
        business_cycle=business_cycle.to_dict() if business_cycle else {},
        inflation=inflation_analysis.to_dict() if inflation_analysis else {},
        key_levels=key_levels,
    )


# === Category Score Computations ===

def _compute_growth_cycle_score(
    cycle: Optional[BusinessCycleAnalysis],
    yc: YieldCurveAnalysis,
) -> float:
    """
    Growth & Cycle category: -1 (recession) to +1 (strong expansion).

    Primary inputs: business cycle composite, LEI, Sahm Rule.
    Secondary: yield curve recession signal.
    """
    if cycle is not None:
        # Use the business cycle composite as the primary driver
        score = cycle.composite_leading * 0.6

        # Recession probability as drag
        score -= cycle.recession_probability * 0.3

        # Sahm Rule proximity
        if cycle.sahm_triggered:
            score -= 0.4
        elif cycle.sahm_value > 0.35:
            score -= 0.2

        return max(-1.0, min(1.0, score))

    # Fallback: yield curve only (Sprint 0 compatibility)
    score = 0.0
    if yc.curve_shape == "normal":
        score += 0.3
    elif yc.curve_shape == "inverted":
        score -= 0.4
    score -= yc.recession_signal_score * 0.4
    return max(-1.0, min(1.0, score))


def _compute_monetary_liquidity_score(
    mp: Optional[MonetaryPolicyAnalysis],
    yc: YieldCurveAnalysis,
) -> float:
    """
    Monetary & Liquidity category: -1 (very tight) to +1 (very loose).

    Primary inputs: net liquidity trend, policy stance, financial conditions.
    """
    if mp is not None:
        # Weighted combination of policy and liquidity scores
        score = mp.policy_score * 0.5 + mp.liquidity_score * 0.5
        return max(-1.0, min(1.0, score))

    # Fallback: infer from yield curve and cross-asset
    score = 0.0
    if yc.curve_shape == "normal":
        score += 0.2  # Normal curve suggests adequate liquidity
    return max(-1.0, min(1.0, score))


def _compute_inflation_category_score(infl):
    """
    Inflation category contribution to composite.

    The inflation module's score is: positive = overheating, negative = deflation.
    For the composite regime score: both are bad (both should be bearish).
    We negate so that overheating inflation DRAGS the composite down.
    """
    if infl is not None:
        return -infl.inflation_score
    return 0.0


def _compute_market_signals_score(
    vol: VolatilityAnalysis,
    breadth: BreadthAnalysis,
    ca: CrossAssetAnalysis,
    yc: YieldCurveAnalysis,
) -> float:
    """
    Market Signals category: -1 (risk-off) to +1 (risk-on).

    Combines: volatility regime, breadth health, cross-asset signals.
    """
    # Volatility: low risk = bullish, high risk = bearish
    vol_score = 1.0 - (vol.vol_risk_score * 2.0)

    # Breadth: healthy = bullish
    breadth_score = (breadth.breadth_score * 2.0) - 1.0
    if breadth.breadth_divergence:
        breadth_score -= 0.3

    # Cross-asset
    ca_score = ca.cross_asset_score

    # Yield curve market signal (term structure as market indicator)
    yc_score = getattr(yc, 'yield_curve_score', 0.0)

    # Weighted combination
    score = (
        vol_score * 0.35 +
        breadth_score * 0.20 +
        ca_score * 0.25 +
        yc_score * 0.20
    )

    return max(-1.0, min(1.0, score))


# === Dynamic Weighting ===

def _compute_dynamic_weights(
    growth: float, monetary: float, inflation: float, market: float,
    spx_returns: Optional[np.ndarray],
    historical_scores: Optional[dict],
) -> dict[str, float]:
    """
    Dynamically weight categories based on what's driving markets.

    When we have sufficient historical data, compute rolling correlations
    between each category score and SPX returns. Higher correlation =
    higher weight (that category is explaining more of market behavior).

    Without historical data, fall back to defaults.
    """
    if spx_returns is None or historical_scores is None:
        return DEFAULT_WEIGHTS.copy()

    # Need at least 30 observations for meaningful correlation
    if len(spx_returns) < 30:
        return DEFAULT_WEIGHTS.copy()

    try:
        correlations = {}
        for cat_name in ["growth", "monetary", "inflation", "market"]:
            if cat_name in historical_scores and len(historical_scores[cat_name]) >= 30:
                cat_scores = np.array(historical_scores[cat_name][-30:])
                ret = spx_returns[-30:]
                if len(cat_scores) == len(ret):
                    corr = abs(np.corrcoef(cat_scores, ret)[0, 1])
                    correlations[cat_name] = corr if not np.isnan(corr) else 0.0
                else:
                    correlations[cat_name] = 0.25
            else:
                correlations[cat_name] = 0.25

        # Normalize to sum to 1.0
        total = sum(correlations.values())
        if total > 0:
            weights = {k: v / total for k, v in correlations.items()}
        else:
            weights = DEFAULT_WEIGHTS.copy()

        # Floor: no category below 10% weight
        min_weight = 0.10
        for k in weights:
            weights[k] = max(min_weight, weights[k])

        # Renormalize after floor
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights

    except Exception:
        return DEFAULT_WEIGHTS.copy()


# === Regime Classification ===

def _classify_primary_regime(
    composite: float, stress: float, vol: VolatilityAnalysis
) -> tuple[MarketRegime, float]:
    """Map composite score and stress to primary regime."""

    # Crisis overrides
    if stress > 0.8:
        return MarketRegime.CRISIS, min(1.0, 0.5 + stress * 0.5)

    # Standard mapping with confidence
    if composite > 0.3:
        confidence = min(1.0, 0.5 + composite * 0.5)
        return MarketRegime.RISK_ON, confidence
    elif composite < -0.3:
        confidence = min(1.0, 0.5 + abs(composite) * 0.5)
        return MarketRegime.RISK_OFF, confidence
    else:
        # Transition — confidence inversely related to how close to zero
        confidence = max(0.3, 0.6 - abs(composite) * 0.5)
        return MarketRegime.TRANSITION, confidence


def _compute_stress_index(
    vol: VolatilityAnalysis,
    ca: CrossAssetAnalysis,
    mp: Optional[MonetaryPolicyAnalysis],
    cycle: Optional[BusinessCycleAnalysis],
    yc: YieldCurveAnalysis,
) -> float:
    """
    Systemic stress index: 0 (calm) to 1 (crisis).

    Separate from direction — stress can be high even when
    composite direction is unclear (transition periods).
    """
    stress = 0.0

    # Volatility
    stress += vol.vol_risk_score * 0.30

    # Credit
    if ca.credit_stress:
        stress += 0.20
    elif ca.hy_spread is not None and ca.hy_spread > 4.0:
        stress += 0.10

    # Yield curve recession signal
    stress += yc.recession_signal_score * 0.15

    # Monetary tightening
    if mp is not None and mp.liquidity_regime in ("tightening", "scarce"):
        stress += 0.15

    # Recession probability
    if cycle is not None:
        stress += cycle.recession_probability * 0.20

    return min(1.0, stress)


def _identify_binding_constraint(
    growth: float, monetary: float, inflation: float, market: float,
    weights: dict,
) -> str:
    """
    Identify which category is the binding constraint.

    The binding constraint is the category with the largest NEGATIVE
    weighted contribution — i.e., what's holding the regime back most.
    If all are positive, the binding constraint is whichever is least positive.
    """
    contributions = {
        "growth_cycle": growth * weights["growth"],
        "monetary_liquidity": monetary * weights["monetary"],
        "inflation": inflation * weights["inflation"],
        "market_signals": market * weights["market"],
    }

    # Most negative contribution
    return min(contributions, key=contributions.get)


def _extract_key_levels(
    vol: VolatilityAnalysis,
    yc: YieldCurveAnalysis,
    ca: CrossAssetAnalysis,
    mp: Optional[MonetaryPolicyAnalysis],
    cycle: Optional[BusinessCycleAnalysis],
    infl: Optional[InflationAnalysis],
) -> dict:
    """Extract the most important current levels for quick reference."""
    levels = {
        "vix": vol.vix_current,
        "spread_2s10s": yc.spread_2s10s,
        "hy_spread": ca.hy_spread,
    }

    if mp:
        levels["fed_funds"] = mp.fed_funds_current
        levels["net_liquidity"] = mp.net_liquidity
        levels["nfci"] = mp.nfci

    if cycle:
        levels["sahm_value"] = cycle.sahm_value
        levels["ism_manufacturing"] = cycle.ism_manufacturing
        levels["recession_prob"] = cycle.recession_probability

    if infl:
        levels["core_pce_yoy"] = infl.core_pce_yoy
        levels["core_cpi_3m_ann"] = infl.core_cpi_3m_annualized
        levels["breakeven_5y"] = infl.breakeven_5y

    return levels
