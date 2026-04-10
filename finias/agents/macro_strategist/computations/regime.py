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
    consistency_warnings: list = field(default_factory=list)
    trajectory: dict = field(default_factory=dict)    # TrajectoryAssessment.to_dict()
    positioning: dict = field(default_factory=dict)   # PositioningAnalysis.to_dict()

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
            "consistency_warnings": self.consistency_warnings,
            "trajectory": self.trajectory,
            "positioning": self.positioning,
        }

    def to_downstream_context(self) -> "MacroContext":
        """
        Generate a clean macro context for downstream agent consumption.

        This is the primary interface other agents use to understand
        the macro environment. It extracts actionable fields from
        the full regime assessment.
        """
        # Determine sector guidance from trajectory data (empirical) with heuristic fallback
        traj = self.trajectory if isinstance(self.trajectory, dict) else {}
        traj_sector = traj.get("sector_guidance", {})
        overweights = traj_sector.get("overweight", [])
        underweights = traj_sector.get("underweight", [])

        CYCLICAL_ETFS = {"XLF", "XLI", "XLY", "XLK", "XLB", "XLRE"}
        DEFENSIVE_ETFS = {"XLP", "XLU", "XLV"}

        if overweights:
            favor_cyclicals = any(s in CYCLICAL_ETFS for s in overweights)
            favor_defensives = any(s in DEFENSIVE_ETFS for s in overweights)
        else:
            # Fallback: heuristic when trajectory data unavailable
            favor_cyclicals = (
                self.cycle_phase in ("early_cycle", "mid_cycle") and
                self.primary_regime != MarketRegime.RISK_OFF and
                self.liquidity_regime in ("ample", "adequate")
            )
            favor_defensives = (
                self.cycle_phase in ("late_cycle", "recession") or
                self.primary_regime in (MarketRegime.RISK_OFF, MarketRegime.CRISIS) or
                self.stress_index > 0.5
            )

        # Rate environment from monetary policy data
        fed_funds = self.key_levels.get("fed_funds", 0)
        neutral_rate = 2.75
        rate_gap = fed_funds - neutral_rate if fed_funds else 0
        if rate_gap > 1.0:
            rate_env = "restrictive"
        elif rate_gap < -0.5:
            rate_env = "accommodative"
        else:
            rate_env = "neutral"

        # Implied rate direction from forward rates
        yc_data = self.yield_curve
        fwd_rates = yc_data.get("forward_rates", {}) if isinstance(yc_data, dict) else {}
        implied_change_bp = fwd_rates.get("implied_policy_change_1y_bp", 0) or 0

        if implied_change_bp < -50:
            implied_direction = "cuts_expected"
        elif implied_change_bp > 50:
            implied_direction = "hikes_expected"
        else:
            implied_direction = "stable"

        favor_duration = implied_direction == "cuts_expected"

        # Liquidity
        mp_data = self.monetary_policy if isinstance(self.monetary_policy, dict) else {}
        liq_data = mp_data.get("liquidity", {})
        liq_trend = liq_data.get("trend", "unknown")
        net_liq = liq_data.get("net_liquidity_millions", 0) or liq_data.get("net_liquidity", 0) or 0
        liquidity_supportive = liq_trend in ("expanding", "stable")

        # Volatility
        vol_data = self.volatility if isinstance(self.volatility, dict) else {}
        ts_data = vol_data.get("term_structure", {})
        vrp_data = vol_data.get("vrp", {})
        vol_persistent = ts_data.get("shape", "unknown") == "backwardation"
        vrp_regime = vrp_data.get("vrp_regime", "unknown")

        # Risk flags
        ca_data = self.cross_asset if isinstance(self.cross_asset, dict) else {}
        credit_data = ca_data.get("credit", {})
        sb_data = ca_data.get("stock_bond_correlation", {})

        # Business cycle
        bc_data = self.business_cycle if isinstance(self.business_cycle, dict) else {}
        sahm_data = bc_data.get("sahm_rule", {})
        activity_data = bc_data.get("activity", {})

        # Breadth
        br_data = self.breadth if isinstance(self.breadth, dict) else {}

        # Positioning data
        pos_data = self.positioning if isinstance(self.positioning, dict) else {}
        pos_agg = pos_data.get("aggregate", {})
        pos_contracts = pos_data.get("contracts", {})
        sp500_pos = pos_contracts.get("sp500", {})

        # Run consistency checks
        warnings = _validate_consistency(
            self.composite_score,
            self.growth_cycle_score,
            self.monetary_liquidity_score,
            self.inflation_score,
            self.market_signals_score,
            self.primary_regime,
            self.cycle_phase,
            self.key_levels,
        )

        # Trajectory signals (from regime.trajectory dict)
        traj = self.trajectory if isinstance(self.trajectory, dict) else {}
        traj_signals = traj.get("trajectory_signals", {})
        traj_rate = traj.get("rate_decisions", {})
        traj_surprise = traj.get("inflation_surprise", {})
        traj_sector = traj.get("sector_guidance", {})
        traj_bias = traj.get("forward_bias", {})
        traj_sizing = traj.get("position_sizing", {})
        traj_velocity = traj.get("velocity", {})
        traj_events = traj.get("event_calendar", {})
        traj_geo = traj.get("geopolitical", {})

        return MacroContext(
            regime=self.primary_regime.value,
            cycle_phase=self.cycle_phase,
            binding_constraint=self.binding_constraint,
            composite_score=self.composite_score,
            confidence=self.confidence,
            stress_index=self.stress_index,
            favor_cyclicals=favor_cyclicals,
            favor_defensives=favor_defensives,
            favor_duration=favor_duration,
            rate_environment=rate_env,
            implied_rate_direction=implied_direction,
            implied_policy_change_bp=implied_change_bp,
            liquidity_supportive=liquidity_supportive,
            net_liquidity_trillion=net_liq / 1_000_000 if net_liq > 1000 else net_liq,
            volatility_regime=self.volatility_regime,
            vol_persistent=vol_persistent,
            vrp_regime=vrp_regime,
            credit_stress=credit_data.get("stress", False),
            risk_parity_stress=sb_data.get("risk_parity_stress", False),
            recession_probability=self.key_levels.get("recession_prob", 0) or 0,
            sahm_distance_to_trigger=sahm_data.get("distance_to_trigger", 0) or 0,
            breadth_health=br_data.get("breadth_health", "unknown"),
            vix_level=self.key_levels.get("vix", 0) or 0,
            hy_spread=self.key_levels.get("hy_spread", 0) or 0,
            fed_funds=fed_funds or 0,
            core_pce_yoy=self.key_levels.get("core_pce_yoy", 0) or 0,
            gdp_nowcast=activity_data.get("gdp_nowcast"),
            inflation_trajectory=traj_signals.get("inflation_trajectory", "unknown"),
            inflation_surprise_direction=traj_surprise.get("direction", "neutral"),
            inflation_surprise_pp=traj_surprise.get("surprise_pp", 0.0),
            stress_contrarian_signal=traj_signals.get("stress_contrarian", "neutral"),
            binding_constraint_shifted=traj_signals.get("binding_shifted", False),
            binding_shift_direction=traj_signals.get("shift_direction", "none"),
            policy_trajectory=traj_rate.get("policy_trajectory", "unknown"),
            cumulative_rate_change_12m_bp=traj_rate.get("cumulative_change_bp", 0.0),
            forward_bias=traj_bias.get("bias", "neutral"),
            forward_bias_score=traj_bias.get("score", 0.0),
            forward_bias_confidence=traj_bias.get("confidence", "low"),
            sector_overweights=traj_sector.get("overweight", []),
            sector_underweights=traj_sector.get("underweight", []),
            sector_rationale=traj_sector.get("rationale", ""),
            max_single_position_pct=traj_sizing.get("max_single_position_pct", 5.0),
            max_sector_exposure_pct=traj_sizing.get("max_sector_exposure_pct", 30.0),
            portfolio_beta_target=traj_sizing.get("portfolio_beta_target", 1.0),
            cash_target_pct=traj_sizing.get("cash_target_pct", 5.0),
            reduce_overall_exposure=traj_sizing.get("reduce_overall_exposure", False),
            pre_event_sizing_multiplier=traj_events.get("pre_event_sizing_multiplier", 1.0),
            scenario_triggers=traj.get("scenario_triggers", []),
            vix_velocity=traj_velocity.get("vix", "unknown"),
            spread_velocity=traj_velocity.get("credit_spreads", "unknown"),
            breadth_velocity=traj_velocity.get("breadth", "unknown"),
            dollar_velocity=traj_velocity.get("dollar", "unknown"),
            liquidity_velocity=traj_velocity.get("liquidity", "unknown"),
            urgency=traj_velocity.get("urgency", "normal"),
            upcoming_events=traj_events.get("upcoming_events", []),
            nearest_high_impact_days=traj_events.get("nearest_high_impact_days"),
            active_geopolitical_risks=traj_geo.get("active_risks", []),
            geopolitical_risk_level=traj_geo.get("risk_level", "unknown"),
            narrative_regime=traj_geo.get("narrative_regime", "unknown"),
            data_freshness_warnings=traj.get("data_freshness", {}).get("warnings", []),
            sp500_positioning_percentile=sp500_pos.get("net_spec_percentile", 50.0),
            sp500_positioning_crowding=sp500_pos.get("crowding", "neutral"),
            positioning_aggregate_score=pos_agg.get("score", 0.0),
            positioning_crowding_count=pos_agg.get("crowding_alert_count", 0),
            consistency_warnings=warnings,
        )

    def to_director_summary(self) -> dict:
        """
        Generate a reduced regime summary for the Director's tool_result.

        The full to_dict() produces 200+ fields including every intermediate
        computation. The Director only needs the interpretation, key levels,
        trajectory signals, and regime classification to synthesize a
        user-facing response.

        This reduces the Director's input tokens by roughly 50%, eliminating
        rate limit collisions and reducing cost.
        """
        traj = self.trajectory or {}

        return {
            "regime": {
                "primary": self.primary_regime.value,
                "cycle_phase": self.cycle_phase,
                "liquidity": self.liquidity_regime,
                "volatility": self.volatility_regime,
                "inflation": self.inflation_regime,
            },
            "scores": {
                "composite": self.composite_score,
                "growth": self.growth_cycle_score,
                "monetary": self.monetary_liquidity_score,
                "inflation": self.inflation_score,
                "market": self.market_signals_score,
            },
            "confidence": self.confidence,
            "stress_index": self.stress_index,
            "binding_constraint": self.binding_constraint,
            "key_levels": self.key_levels,
            "trajectory": {
                "forward_bias": traj.get("forward_bias", {}),
                "position_sizing": traj.get("position_sizing", {}),
                "velocity": traj.get("velocity", {}),
                "event_calendar": traj.get("event_calendar", {}),
                "scenario_triggers": traj.get("scenario_triggers", []),
                "sector_guidance": traj.get("sector_guidance", {}),
                "rate_decisions": traj.get("rate_decisions", {}),
                "inflation_surprise": traj.get("inflation_surprise", {}),
                "trajectory_signals": traj.get("trajectory_signals", {}),
            },
            "cross_asset_summary": {
                "credit": self.cross_asset.get("credit", {}),
                "oil": self.cross_asset.get("oil", {}),
                "dollar": self.cross_asset.get("dollar", {}),
                "risk_appetite": self.cross_asset.get("risk_appetite", {}),
                "stock_bond_correlation": self.cross_asset.get("stock_bond_correlation", {}),
                "correlations_aggregate": self.cross_asset.get("correlations", {}).get("aggregate", {}),
            },
            "breadth_summary": {
                "health": self.breadth.get("breadth_health", "unknown"),
                "score": self.breadth.get("breadth_score", 0),
                "participation": self.breadth.get("sector_participation", {}),
                "rotation": self.breadth.get("sector_rotation", {}),
            },
            "positioning_summary": self.positioning.get("aggregate", {}) if self.positioning else {},
        }


@dataclass
class MacroContext:
    """
    Simplified macro context for downstream agent consumption.

    This is the interface contract between the Macro Strategist and all other agents.
    Instead of parsing 2000+ lines of raw regime JSON, downstream agents consume
    this clean, structured summary of what matters for their decision-making.

    Generated by RegimeAssessment.to_downstream_context() — always derived from
    the full assessment, never stored or computed separately.
    """
    # Regime classification
    regime: str                         # risk_on, risk_off, transition, crisis
    cycle_phase: str                    # early_cycle, mid_cycle, late_cycle, recession
    binding_constraint: str             # inflation, growth_cycle, monetary_liquidity, market_signals
    composite_score: float              # -1 to +1 (negative = bearish, positive = bullish)
    confidence: float                   # 0 to 1
    stress_index: float                 # 0 to 1 (>0.5 elevated, >0.8 crisis)

    # Actionable context for sector/stock selection
    favor_cyclicals: bool               # True if early/mid cycle with adequate liquidity
    favor_defensives: bool              # True if late cycle, risk-off, or high stress
    favor_duration: bool                # True if rates expected to fall (cuts priced)

    # Rate & liquidity environment
    rate_environment: str               # restrictive, neutral, accommodative
    implied_rate_direction: str         # cuts_expected, stable, hikes_expected
    implied_policy_change_bp: float     # Forward rate minus fed funds (positive = hikes priced)
    liquidity_supportive: bool          # True if net liquidity trend is expanding or stable
    net_liquidity_trillion: float       # Current net liquidity in trillions

    # Volatility context
    volatility_regime: str              # low, normal, elevated, extreme
    vol_persistent: bool                # True if VIX term structure in backwardation
    vrp_regime: str                     # normal, compressed, flat, negative

    # Risk flags
    credit_stress: bool                 # True if HY spread > 500bp
    risk_parity_stress: bool            # True if stocks and bonds falling together
    recession_probability: float        # 0 to 1
    sahm_distance_to_trigger: float     # How far from 0.50 threshold
    breadth_health: str                 # strong, healthy, weakening, poor

    # Key levels for risk management thresholds
    vix_level: float
    hy_spread: float
    fed_funds: float
    core_pce_yoy: float
    gdp_nowcast: Optional[float]        # None if not available

    # === TRAJECTORY SIGNALS (new) ===
    inflation_trajectory: str = "unknown"               # easing, stable, tightening
    inflation_surprise_direction: str = "neutral"       # hawkish, neutral, dovish
    inflation_surprise_pp: float = 0.0                  # actual minus expected
    stress_contrarian_signal: str = "neutral"            # opportunity, neutral, caution
    binding_constraint_shifted: bool = False
    binding_shift_direction: str = "none"
    policy_trajectory: str = "unknown"                   # cutting, holding, hiking
    cumulative_rate_change_12m_bp: float = 0.0
    forward_bias: str = "neutral"                        # constructive, neutral, cautious
    forward_bias_score: float = 0.0
    forward_bias_confidence: str = "low"
    sector_overweights: list = field(default_factory=list)
    sector_underweights: list = field(default_factory=list)
    sector_rationale: str = ""

    # === POSITION SIZING (for Risk Officer / Execution Agent) ===
    max_single_position_pct: float = 5.0
    max_sector_exposure_pct: float = 30.0
    portfolio_beta_target: float = 1.0
    cash_target_pct: float = 5.0
    reduce_overall_exposure: bool = False
    pre_event_sizing_multiplier: float = 1.0

    # === SCENARIO TRIGGERS (for Thesis Monitor) ===
    scenario_triggers: list = field(default_factory=list)

    # === VELOCITY (for all agents — urgency assessment) ===
    vix_velocity: str = "unknown"
    spread_velocity: str = "unknown"
    breadth_velocity: str = "unknown"
    dollar_velocity: str = "unknown"
    liquidity_velocity: str = "unknown"
    urgency: str = "normal"

    # === EVENT CALENDAR (for Risk Officer / Trade Decision Agent) ===
    upcoming_events: list = field(default_factory=list)
    nearest_high_impact_days: Optional[int] = None

    # === GEOPOLITICAL CONTEXT (placeholder — populated by future News agent) ===
    active_geopolitical_risks: list = field(default_factory=list)
    geopolitical_risk_level: str = "unknown"
    narrative_regime: str = "unknown"

    # === DATA QUALITY (for all agents — confidence calibration) ===
    data_freshness_warnings: list = field(default_factory=list)

    # === POSITIONING (for Trade Decision Agent / Risk Officer) ===
    sp500_positioning_percentile: float = 50.0              # 0-100 (CFTC COT)
    sp500_positioning_crowding: str = "neutral"             # crowded_long, crowded_short, neutral
    positioning_aggregate_score: float = 0.0                # -1 to +1
    positioning_crowding_count: int = 0                     # How many contracts at extremes

    # Consistency
    consistency_warnings: list = field(default_factory=list)  # Any internal contradictions detected

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "cycle_phase": self.cycle_phase,
            "binding_constraint": self.binding_constraint,
            "composite_score": self.composite_score,
            "confidence": self.confidence,
            "stress_index": self.stress_index,
            "sector_guidance": {
                "favor_cyclicals": self.favor_cyclicals,
                "favor_defensives": self.favor_defensives,
                "favor_duration": self.favor_duration,
                "overweight": self.sector_overweights,
                "underweight": self.sector_underweights,
                "rationale": self.sector_rationale,
            },
            "rates_and_liquidity": {
                "rate_environment": self.rate_environment,
                "implied_rate_direction": self.implied_rate_direction,
                "implied_policy_change_bp": self.implied_policy_change_bp,
                "liquidity_supportive": self.liquidity_supportive,
                "net_liquidity_trillion": self.net_liquidity_trillion,
            },
            "volatility": {
                "regime": self.volatility_regime,
                "persistent": self.vol_persistent,
                "vrp_regime": self.vrp_regime,
            },
            "risk_flags": {
                "credit_stress": self.credit_stress,
                "risk_parity_stress": self.risk_parity_stress,
                "recession_probability": self.recession_probability,
                "sahm_distance_to_trigger": self.sahm_distance_to_trigger,
                "breadth_health": self.breadth_health,
            },
            "key_levels": {
                "vix": self.vix_level,
                "hy_spread": self.hy_spread,
                "fed_funds": self.fed_funds,
                "core_pce_yoy": self.core_pce_yoy,
                "gdp_nowcast": self.gdp_nowcast,
            },
            "trajectory": {
                "inflation_trajectory": self.inflation_trajectory,
                "inflation_surprise": {
                    "direction": self.inflation_surprise_direction,
                    "surprise_pp": self.inflation_surprise_pp,
                },
                "stress_contrarian": self.stress_contrarian_signal,
                "binding_shifted": self.binding_constraint_shifted,
                "binding_shift_direction": self.binding_shift_direction,
                "policy_trajectory": self.policy_trajectory,
                "cumulative_rate_change_12m_bp": self.cumulative_rate_change_12m_bp,
                "forward_bias": self.forward_bias,
                "forward_bias_score": self.forward_bias_score,
                "forward_bias_confidence": self.forward_bias_confidence,
            },
            "position_sizing": {
                "max_single_position_pct": self.max_single_position_pct,
                "max_sector_exposure_pct": self.max_sector_exposure_pct,
                "portfolio_beta_target": self.portfolio_beta_target,
                "cash_target_pct": self.cash_target_pct,
                "reduce_overall_exposure": self.reduce_overall_exposure,
                "pre_event_sizing_multiplier": self.pre_event_sizing_multiplier,
            },
            "scenario_triggers": self.scenario_triggers,
            "velocity": {
                "vix": self.vix_velocity,
                "credit_spreads": self.spread_velocity,
                "breadth": self.breadth_velocity,
                "dollar": self.dollar_velocity,
                "liquidity": self.liquidity_velocity,
                "urgency": self.urgency,
            },
            "event_calendar": {
                "upcoming_events": self.upcoming_events,
                "nearest_high_impact_days": self.nearest_high_impact_days,
            },
            "geopolitical": {
                "active_risks": self.active_geopolitical_risks,
                "risk_level": self.geopolitical_risk_level,
                "narrative_regime": self.narrative_regime,
            },
            "data_freshness": {
                "warnings": self.data_freshness_warnings,
            },
            "consistency_warnings": self.consistency_warnings,
        }


def _validate_consistency(
    composite: float,
    growth: float,
    monetary: float,
    inflation: float,
    market: float,
    regime: MarketRegime,
    cycle_phase: str,
    key_levels: dict,
) -> list[str]:
    """
    Check regime assessment for internal contradictions.

    Returns a list of warning strings. Empty list = internally consistent.
    These warnings are informational — they don't change the assessment.
    They tell downstream consumers how much to trust the composite.
    """
    warnings = []

    scores = {"growth": growth, "monetary": monetary, "inflation": inflation, "market": market}

    # Check 1: Composite direction vs category majority
    positive_count = sum(1 for s in scores.values() if s > 0.05)
    negative_count = sum(1 for s in scores.values() if s < -0.05)

    if composite > 0.1 and negative_count >= 3:
        warnings.append(
            f"Composite is positive ({composite:.3f}) but {negative_count}/4 categories are negative. "
            f"Positive composite may be driven by a single strong category."
        )
    elif composite < -0.1 and positive_count >= 3:
        warnings.append(
            f"Composite is negative ({composite:.3f}) but {positive_count}/4 categories are positive. "
            f"Negative composite may be driven by a single strongly negative category."
        )

    # Check 2: Regime vs Sahm Rule
    sahm = key_levels.get("sahm_value", 0) or 0
    recession_prob = key_levels.get("recession_prob", 0) or 0
    if sahm > 0.40 and regime == MarketRegime.RISK_ON:
        warnings.append(
            f"Regime is risk_on but Sahm Rule at {sahm:.3f} (approaching 0.50 trigger). "
            f"Risk-on classification may be premature."
        )

    # Check 3: Growth score vs GDPNow
    gdp = key_levels.get("gdp_nowcast")
    if gdp is not None:
        if gdp > 3.0 and growth < -0.3:
            warnings.append(
                f"GDPNow at {gdp:.1f}% (strong growth) but growth_cycle_score is {growth:.3f} (bearish). "
                f"Sahm Rule penalty may be overriding positive growth signals."
            )
        elif gdp < 1.0 and growth > 0.3:
            warnings.append(
                f"GDPNow at {gdp:.1f}% (weak growth) but growth_cycle_score is {growth:.3f} (bullish). "
                f"Growth score may be lagging reality."
            )

    # Check 4: Cycle phase vs recession probability
    if cycle_phase in ("early_cycle", "mid_cycle") and recession_prob > 0.40:
        warnings.append(
            f"Classified as {cycle_phase} but recession probability is {recession_prob:.0%}. "
            f"Cycle classification may be stale."
        )

    # Check 5: Inflation binding but inflation regime is "stable" or "disinflation"
    infl_regime_key = key_levels.get("core_pce_yoy", 0) or 0
    if inflation < -0.3 and infl_regime_key < 2.5:
        warnings.append(
            f"Inflation score is strongly negative ({inflation:.3f}) suggesting overheating, "
            f"but core PCE at {infl_regime_key:.1f}% is near target. Check inflation score logic."
        )

    # Check 6: High stress but risk_on regime
    stress = key_levels.get("vix", 0) or 0
    if stress > 30 and regime == MarketRegime.RISK_ON:
        warnings.append(
            f"VIX at {stress:.1f} (above 30) but regime is risk_on. "
            f"Stress may not be fully reflected in the composite."
        )

    return warnings


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

    # === Step 7: Consistency Validation ===
    consistency_warnings = _validate_consistency(
        composite, growth_score, monetary_score, infl_score, market_score,
        primary_regime, cycle, key_levels,
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
        consistency_warnings=consistency_warnings,
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

    # IMPLEMENTATION NOTE (2026-04):
    # Thresholds recalibrated from ±0.3 to +0.10/-0.15 based on
    # actual composite score distribution (range: -0.4 to +0.12).
    # Old thresholds produced transition 96% of the time (useless).
    # New thresholds produce ~4 meaningful regimes:
    #   crisis (stress > 0.7): ~2-3% of observations
    #   risk_off (composite < -0.15): ~15-20%
    #   risk_on (composite > 0.10): ~10-15%
    #   transition (everything else): ~60-70%
    # This is calibration to the data range, not optimization to returns.
    # Crisis threshold lowered from 0.8 to 0.7 (stress-drawdown
    # correlation of -0.256 confirmed in 198-observation backtest).

    # Crisis overrides — lowered from 0.8 to 0.7 to catch more crisis events
    # Backtest showed stress > 0.7 correlated with -5.48% avg max drawdown
    if stress > 0.7:
        return MarketRegime.CRISIS, min(1.0, 0.5 + stress * 0.5)

    # Recalibrated thresholds based on actual composite score distribution:
    # - Historical range: roughly -0.4 to +0.12
    # - Old thresholds (±0.3): risk_on NEVER triggered, risk_off only 4%
    # - New thresholds: risk_on ~10-15%, risk_off ~15-20%, transition ~60-70%
    # This is calibration to the actual data range, not overfitting to returns.
    if composite > 0.10:
        confidence = min(1.0, 0.5 + composite * 2.0)
        return MarketRegime.RISK_ON, confidence
    elif composite < -0.15:
        confidence = min(1.0, 0.5 + abs(composite) * 1.5)
        return MarketRegime.RISK_OFF, confidence
    else:
        # Transition — confidence inversely related to how close to zero
        confidence = max(0.3, 0.6 - abs(composite) * 2.0)
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
        levels["core_pce_3m_ann"] = infl.core_pce_3m_annualized
        levels["breakeven_5y"] = infl.breakeven_5y

    return levels
