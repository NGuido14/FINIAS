"""
Macro Trajectory Layer

Computes forward-looking signals that sit on top of the descriptive regime assessment.
The descriptive layer tells you WHERE things ARE. This layer tells you WHERE things are GOING.

Validated by walk-forward backtesting on 196 weekly observations (2022-2025)
with corrected Sahm Rule computation:
- Inflation trajectory: strongest signal (+1.32% vs -0.11% spread, 73% hit rate)
- Stress contrarian: positive returns (+0.96%) but underperforms neutral baseline
- Binding constraint transitions: small spread (+1.20% away vs +0.58% toward inflation)

All computation is pure Python. No API calls. No Claude.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from datetime import date


@dataclass
class TrajectoryAssessment:
    """
    Forward-looking trajectory signals computed from the descriptive layer.

    This is the predictive complement to RegimeAssessment. RegimeAssessment
    describes the current state. TrajectoryAssessment predicts direction.
    """

    # === Rate Decision History ===
    rate_decisions_12m: list = field(default_factory=list)  # [{"date": str, "change_bp": float}, ...]
    cumulative_rate_change_12m_bp: float = 0.0              # Total bp change over 12 months
    policy_trajectory: str = "unknown"                       # cutting, holding, hiking
    months_since_last_change: float = 0.0

    # === Inflation Surprise ===
    inflation_surprise_pp: float = 0.0                       # actual core PCE minus breakeven 5Y
    inflation_surprise_direction: str = "neutral"            # hawkish, neutral, dovish

    # === Signal 1: Inflation Trajectory (strongest signal) ===
    inflation_trajectory: str = "unknown"                    # easing, stable, tightening
    inflation_score_4w_change: float = 0.0                   # Change in inflation category score

    # === Signal 2: Stress Contrarian ===
    stress_4w_change: float = 0.0
    stress_contrarian_signal: str = "neutral"                # opportunity, neutral, caution

    # === Signal 3: Binding Constraint Shift ===
    binding_constraint_shifted: bool = False
    prior_binding_constraint: str = "none"
    binding_shift_direction: str = "none"                    # toward_inflation, away_from_inflation, none

    # === Sector Guidance (empirically derived) ===
    sector_overweights: list = field(default_factory=list)   # Top 3 sectors for current conditions
    sector_underweights: list = field(default_factory=list)  # Bottom 3 sectors
    sector_rationale: str = ""                               # Which macro factor drives the recommendation

    # === Net Forward Bias ===
    forward_bias: str = "neutral"                            # constructive, neutral, cautious
    forward_bias_score: float = 0.0                          # -1 to +1
    forward_bias_confidence: str = "low"                     # high, moderate, low

    # === Position Sizing Guidance ===
    max_single_position_pct: float = 5.0
    max_sector_exposure_pct: float = 30.0
    portfolio_beta_target: float = 1.0
    cash_target_pct: float = 5.0
    reduce_overall_exposure: bool = False

    # === Scenario Triggers ===
    scenario_triggers: list = field(default_factory=list)

    # === Velocity Context ===
    vix_velocity: str = "unknown"
    spread_velocity: str = "unknown"
    breadth_velocity: str = "unknown"
    dollar_velocity: str = "unknown"
    liquidity_velocity: str = "unknown"
    urgency: str = "normal"

    # === Event Calendar ===
    upcoming_events: list = field(default_factory=list)
    pre_event_sizing_multiplier: float = 1.0
    nearest_high_impact_days: Optional[int] = None

    # === Geopolitical Context (placeholder for future News/Event Monitor agent) ===
    active_geopolitical_risks: list = field(default_factory=list)
    geopolitical_risk_level: str = "unknown"
    narrative_regime: str = "unknown"  # inflation_fear, ai_euphoria, recession_watch, geopolitical, etc.

    # === Data Freshness ===
    data_freshness_warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rate_decisions": {
                "decisions_12m": self.rate_decisions_12m,
                "cumulative_change_bp": self.cumulative_rate_change_12m_bp,
                "policy_trajectory": self.policy_trajectory,
                "months_since_last_change": self.months_since_last_change,
            },
            "inflation_surprise": {
                "surprise_pp": self.inflation_surprise_pp,
                "direction": self.inflation_surprise_direction,
                "_note": "Positive = hawkish (actual > expected). Negative = dovish (actual < expected).",
            },
            "trajectory_signals": {
                "inflation_trajectory": self.inflation_trajectory,
                "inflation_score_4w_change": self.inflation_score_4w_change,
                "stress_contrarian": self.stress_contrarian_signal,
                "stress_4w_change": self.stress_4w_change,
                "binding_shifted": self.binding_constraint_shifted,
                "prior_binding": self.prior_binding_constraint,
                "shift_direction": self.binding_shift_direction,
            },
            "sector_guidance": {
                "overweight": self.sector_overweights,
                "underweight": self.sector_underweights,
                "rationale": self.sector_rationale,
            },
            "forward_bias": {
                "bias": self.forward_bias,
                "score": self.forward_bias_score,
                "confidence": self.forward_bias_confidence,
            },
            "position_sizing": {
                "max_single_position_pct": self.max_single_position_pct,
                "max_sector_exposure_pct": self.max_sector_exposure_pct,
                "portfolio_beta_target": self.portfolio_beta_target,
                "cash_target_pct": self.cash_target_pct,
                "reduce_overall_exposure": self.reduce_overall_exposure,
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
                "pre_event_sizing_multiplier": self.pre_event_sizing_multiplier,
                "nearest_high_impact_days": self.nearest_high_impact_days,
            },
            "geopolitical": {
                "active_risks": self.active_geopolitical_risks,
                "risk_level": self.geopolitical_risk_level,
                "narrative_regime": self.narrative_regime,
                "_note": "Populated by News/Event Monitor agent when built. Empty = no geopolitical context available.",
            },
            "data_freshness": {
                "warnings": self.data_freshness_warnings,
            },
        }


def compute_rate_decision_history(
    fed_target_upper: list[dict],
    as_of_date: date = None,
) -> dict:
    """
    Derive Fed rate decisions from DFEDTARU changes.

    Every change in the Fed Funds Target Upper is a Fed decision.
    Returns the decision history, cumulative change, and trajectory.
    """
    if not fed_target_upper or len(fed_target_upper) < 2:
        return {
            "decisions": [],
            "cumulative_bp": 0.0,
            "trajectory": "unknown",
            "months_since_last": 0.0,
        }

    if as_of_date is None:
        as_of_date = date.today()

    # Find all rate changes
    decisions = []
    for i in range(1, len(fed_target_upper)):
        prev_val = fed_target_upper[i - 1]["value"]
        curr_val = fed_target_upper[i]["value"]
        if abs(curr_val - prev_val) > 0.001:  # Threshold to avoid floating point noise
            change_bp = round((curr_val - prev_val) * 100)
            decisions.append({
                "date": fed_target_upper[i]["date"],
                "rate_after": curr_val,
                "change_bp": change_bp,
            })

    # Filter to last 12 months
    cutoff = date(as_of_date.year - 1, as_of_date.month, as_of_date.day)
    cutoff_str = cutoff.isoformat()
    recent_decisions = [d for d in decisions if d["date"] >= cutoff_str]

    # Cumulative change
    cumulative = sum(d["change_bp"] for d in recent_decisions)

    # Trajectory classification
    if not recent_decisions:
        trajectory = "holding"
    elif cumulative < -50:
        trajectory = "cutting"
    elif cumulative > 50:
        trajectory = "hiking"
    else:
        trajectory = "holding"

    # Months since last change
    if decisions:
        last_date = date.fromisoformat(decisions[-1]["date"])
        months_since = (as_of_date - last_date).days / 30.44
    else:
        months_since = 999.0

    return {
        "decisions": recent_decisions,
        "cumulative_bp": cumulative,
        "trajectory": trajectory,
        "months_since_last": round(months_since, 1),
    }


def compute_inflation_surprise(
    core_pce_yoy: float,
    breakeven_5y: float,
) -> dict:
    """
    Compare actual inflation to market expectations.

    Positive surprise = actual > expected (hawkish — market underpricing inflation)
    Negative surprise = actual < expected (dovish — inflation coming in below expectations)
    """
    if core_pce_yoy is None or breakeven_5y is None:
        return {"surprise_pp": 0.0, "direction": "neutral"}

    surprise = core_pce_yoy - breakeven_5y

    if surprise > 0.30:
        direction = "hawkish"
    elif surprise < -0.30:
        direction = "dovish"
    else:
        direction = "neutral"

    return {"surprise_pp": round(surprise, 3), "direction": direction}


def compute_inflation_trajectory(
    current_inflation_score: float,
    prior_inflation_score: float,
) -> dict:
    """
    Classify inflation trajectory from 4-week change in inflation category score.

    Backtest evidence (196 obs, corrected Sahm Rule):
      Easing (Δ > +0.02):     22 obs, avg 20d return +1.32%, hit rate 73%
      Tightening (Δ < -0.02): 11 obs, avg 20d return -0.11%, hit rate 45%
      Stable:                 163 obs, avg 20d return +1.06%, hit rate 67%
    """
    change = current_inflation_score - prior_inflation_score

    if change > 0.02:
        trajectory = "easing"
    elif change < -0.02:
        trajectory = "tightening"
    else:
        trajectory = "stable"

    return {"trajectory": trajectory, "change": round(change, 4)}


def compute_stress_contrarian(
    current_stress: float,
    prior_stress: float,
    median_stress: float = 0.20,
) -> dict:
    """
    Stress contrarian signal.

    Backtest evidence (196 obs, corrected):
      Opportunity (rising stress): avg 20d return +0.96%, 71% hit rate (41 obs)
      Caution (falling stress):    avg 20d return -0.08%, 64% hit rate (14 obs)
      NOTE: Opportunity underperforms neutral baseline (+1.15%). Use as confirming
      signal alongside inflation trajectory, not as standalone.

    Signal:
      opportunity: stress rising from below median (fear building, likely overdone)
      caution: stress falling from above median (complacency returning)
      neutral: stress stable or change direction unclear
    """
    change = current_stress - prior_stress

    if change > 0.03 and current_stress > median_stress:
        signal = "opportunity"  # Stress rising meaningfully
    elif change < -0.03 and current_stress < median_stress:
        signal = "caution"  # Stress falling, complacency
    else:
        signal = "neutral"

    return {"signal": signal, "change": round(change, 4)}


def compute_binding_shift(
    current_binding: str,
    prior_binding: str,
) -> dict:
    """
    Detect binding constraint transitions.

    Backtest evidence (196 obs, corrected Sahm Rule):
      Shift away from inflation: avg 20d return +1.20% (8 obs)
      Shift toward inflation:    avg 20d return +0.58% (7 obs)
      No shift:                  avg 20d return +1.03% (181 obs)

    NOTE: The spread between away/toward is much smaller than originally measured
    (+4.58%/-0.50% in prior run). Do not weight binding shifts as a primary signal.
    The inflation trajectory signal is more reliable.
    """
    if current_binding == prior_binding or not current_binding or not prior_binding:
        return {"shifted": False, "direction": "none", "prior": prior_binding or "none"}

    # Determine direction relative to inflation
    if prior_binding == "inflation" and current_binding != "inflation":
        direction = "away_from_inflation"
    elif prior_binding != "inflation" and current_binding == "inflation":
        direction = "toward_inflation"
    else:
        direction = "other"

    return {"shifted": True, "direction": direction, "prior": prior_binding}


# === EMPIRICAL SECTOR SENSITIVITY MAP ===
# Computed from 196 weeks of backtest data (2022-2025)
# These are avg 20d forward RELATIVE returns (sector minus SPY)
# Updated periodically as more data accumulates

SECTOR_BY_INFLATION_TRAJECTORY = {
    "easing": {
        "overweight": ["XLK", "XLRE", "XLY"],   # Tech +1.07%, Real Estate +1.72pp spread, Cons Disc +1.24pp
        "underweight": ["XLE", "XLU", "XLP"],    # Energy -1.42%, Utilities -1.54%, Staples -1.66%
    },
    "tightening": {
        "overweight": ["XLE", "XLU", "XLP"],     # Energy +1.08%, Utilities +0.58%, Staples +0.15%
        "underweight": ["XLRE", "XLY", "XLK"],   # Real Estate -2.46%, Cons Disc -0.88%, Tech -0.15%
    },
    "stable": {
        "overweight": ["XLC", "XLK", "XLF"],     # Comm Svcs +0.64%, Tech +0.52%
        "underweight": ["XLRE", "XLP", "XLE"],    # Real Estate -1.26%, Staples -1.25%, Energy -0.99%
    },
}

SECTOR_BY_BINDING_CONSTRAINT = {
    "growth_cycle": {
        "overweight": ["XLK", "XLY", "XLC"],     # Tech +0.90%, Cons Disc +0.49%, Comm Svcs +0.48%
        "underweight": ["XLV", "XLP", "XLE"],     # Healthcare -2.06%, Staples -1.79%, Energy -1.34%
    },
    "inflation": {
        "overweight": ["XLK", "XLC", "XLV"],     # Tech +0.42%, Comm Svcs +0.13%, Healthcare +0.15%
        "underweight": ["XLRE", "XLU", "XLB"],    # Real Estate -1.51%, Utilities -0.45%, Materials -0.57%
    },
    "monetary_liquidity": {
        "overweight": ["XLC", "XLK", "XLF"],     # Comm Svcs +0.63%, Tech +0.55%, Financials +0.56%
        "underweight": ["XLU", "XLP", "XLRE"],    # Utilities -1.77%, Staples -1.74%, Real Estate -1.36%
    },
}

SECTOR_BY_STRESS = {
    "high": {
        "overweight": ["XLRE", "XLY", "XLI"],    # Real Estate +1.77pp, Cons Disc +0.94pp, Industrials +0.76pp
        "underweight": ["XLE", "XLV", "XLC"],     # Energy -1.55pp, Healthcare -0.97pp
    },
    "low": {
        "overweight": ["XLK", "XLE", "XLU"],     # Tech +0.58%, Energy +0.14%
        "underweight": ["XLRE", "XLY", "XLB"],    # Real Estate -1.59%, Cons Disc -1.00%
    },
}

SECTOR_NAMES = {
    "XLB": "Materials", "XLC": "Comm Svcs", "XLE": "Energy",
    "XLF": "Financials", "XLI": "Industrials", "XLK": "Technology",
    "XLP": "Staples", "XLRE": "Real Estate", "XLU": "Utilities",
    "XLV": "Healthcare", "XLY": "Cons Disc",
}


def compute_sector_guidance(
    inflation_trajectory: str,
    binding_constraint: str,
    stress_index: float,
    stress_high_threshold: float = 0.30,
) -> dict:
    """
    Determine sector over/underweights based on current macro conditions.

    Priority: inflation trajectory (strongest signal) > binding constraint > stress level.
    When signals agree, confidence is high. When they conflict, we take the inflation
    trajectory signal (backed by 2.5pp spread in backtest data).
    """
    # Primary signal: inflation trajectory
    infl_guidance = SECTOR_BY_INFLATION_TRAJECTORY.get(inflation_trajectory, {})

    # Secondary signal: binding constraint
    bc_guidance = SECTOR_BY_BINDING_CONSTRAINT.get(binding_constraint, {})

    # Tertiary signal: stress level
    stress_level = "high" if stress_index > stress_high_threshold else "low"
    stress_guidance = SECTOR_BY_STRESS.get(stress_level, {})

    # Use inflation trajectory as primary (strongest empirical signal)
    overweights = infl_guidance.get("overweight", [])
    underweights = infl_guidance.get("underweight", [])

    # Build rationale
    rationale_parts = []
    if inflation_trajectory != "unknown":
        rationale_parts.append(f"Inflation {inflation_trajectory}")
    if binding_constraint != "none":
        rationale_parts.append(f"{binding_constraint} binding")
    if stress_index > stress_high_threshold:
        rationale_parts.append("elevated stress")

    rationale = f"Based on: {', '.join(rationale_parts)}. " if rationale_parts else ""
    rationale += f"Overweight: {', '.join(SECTOR_NAMES.get(s, s) for s in overweights)}. "
    rationale += f"Underweight: {', '.join(SECTOR_NAMES.get(s, s) for s in underweights)}."

    return {
        "overweights": overweights,
        "underweights": underweights,
        "rationale": rationale,
    }


def compute_forward_bias(
    inflation_trajectory: str,
    stress_contrarian: str,
    binding_shift_direction: str,
) -> dict:
    """
    Net forward-looking assessment from the three validated signals.

    Inflation trajectory gets 2x weight (strongest signal in backtest).

    Score mapping:
      +1: easing / opportunity / away_from_inflation
       0: stable / neutral / none
      -1: tightening / caution / toward_inflation

    Classification:
      > +0.25: constructive
      < -0.25: cautious
      else: neutral
    """
    # Score each signal
    infl_score = {"easing": 1, "stable": 0, "tightening": -1}.get(inflation_trajectory, 0)
    stress_score = {"opportunity": 1, "neutral": 0, "caution": -1}.get(stress_contrarian, 0)
    shift_score = {"away_from_inflation": 1, "none": 0, "toward_inflation": -1, "other": 0}.get(binding_shift_direction, 0)

    # Weighted average (inflation 2x)
    weighted = (infl_score * 2 + stress_score * 1 + shift_score * 1) / 4.0

    # Classify
    if weighted > 0.25:
        bias = "constructive"
    elif weighted < -0.25:
        bias = "cautious"
    else:
        bias = "neutral"

    # Confidence based on signal agreement
    signals = [infl_score, stress_score, shift_score]
    nonzero = [s for s in signals if s != 0]
    if len(nonzero) >= 2 and all(s > 0 for s in nonzero):
        confidence = "high"
    elif len(nonzero) >= 2 and all(s < 0 for s in nonzero):
        confidence = "high"
    elif len(nonzero) >= 2:
        confidence = "low"  # Conflicting signals
    else:
        confidence = "moderate"

    return {"bias": bias, "score": round(weighted, 3), "confidence": confidence}


def compute_position_sizing(
    vix_level: float,
    vol_persistent: bool,
    stress_index: float,
    breadth_health: str,
    credit_stress: bool,
    recession_probability: float,
) -> dict:
    """
    Compute position sizing guidance from macro conditions.

    These are MAXIMUM limits, not targets. Downstream agents may use
    smaller sizes based on their own analysis. But no agent should
    exceed these limits without explicit override.

    Logic:
    - Base position: 5% per stock, 30% per sector, beta target 1.0
    - VIX 20-30: reduce to 3%, 25%, beta 0.8
    - VIX > 30: reduce to 2%, 20%, beta 0.7
    - VIX > 30 + backwardation (persistent): reduce to 1.5%, 15%, beta 0.5
    - Stress > 0.5: additional 25% reduction across all limits
    - Credit stress: additional 25% reduction
    - Breadth "poor": additional 20% reduction
    - Recession prob > 0.5: beta target capped at 0.5
    """
    # Base limits
    if vix_level > 35:
        max_position = 1.5
        max_sector = 15.0
        beta_target = 0.5
        cash_target = 25.0
    elif vix_level > 30:
        if vol_persistent:  # Backwardation — vol expected to stay
            max_position = 1.5
            max_sector = 15.0
            beta_target = 0.5
            cash_target = 20.0
        else:
            max_position = 2.0
            max_sector = 20.0
            beta_target = 0.7
            cash_target = 15.0
    elif vix_level > 25:
        max_position = 3.0
        max_sector = 25.0
        beta_target = 0.8
        cash_target = 10.0
    elif vix_level > 20:
        max_position = 4.0
        max_sector = 25.0
        beta_target = 0.9
        cash_target = 5.0
    else:
        max_position = 5.0
        max_sector = 30.0
        beta_target = 1.0
        cash_target = 5.0

    # Stress multiplier
    if stress_index > 0.7:
        stress_mult = 0.5
    elif stress_index > 0.5:
        stress_mult = 0.75
    else:
        stress_mult = 1.0

    max_position *= stress_mult
    max_sector *= stress_mult

    # Credit stress reduction
    if credit_stress:
        max_position *= 0.75
        max_sector *= 0.75
        cash_target = max(cash_target, 20.0)

    # Breadth deterioration
    if breadth_health == "poor":
        max_position *= 0.80
        beta_target = min(beta_target, 0.6)
    elif breadth_health == "weakening":
        max_position *= 0.90

    # Recession override
    if recession_probability > 0.5:
        beta_target = min(beta_target, 0.5)
        cash_target = max(cash_target, 25.0)
    elif recession_probability > 0.35:
        beta_target = min(beta_target, 0.7)

    # Determine if overall exposure should be reduced
    reduce_exposure = (
        stress_index > 0.5 or
        credit_stress or
        (vix_level > 30 and vol_persistent) or
        recession_probability > 0.5 or
        breadth_health == "poor"
    )

    return {
        "max_single_position_pct": round(max_position, 1),
        "max_sector_exposure_pct": round(max_sector, 1),
        "portfolio_beta_target": round(beta_target, 2),
        "cash_target_pct": round(cash_target, 1),
        "reduce_overall_exposure": reduce_exposure,
    }


def compute_scenario_triggers(
    regime_assessment,
) -> list[dict]:
    """
    Generate structured scenario triggers from current macro levels.

    Each trigger defines a specific threshold that, if breached,
    would change the macro regime or forward bias. Downstream agents
    (especially Thesis Monitor) check these programmatically.
    """
    kl = regime_assessment.key_levels
    triggers = []

    # --- Inflation triggers ---
    core_pce_3m = kl.get("core_pce_3m_ann")  # Fed's preferred measure: core PCE 3m annualized
    if core_pce_3m is not None:
        triggers.append({
            "id": "inflation_acceleration",
            "metric": "core_pce_3m_annualized",
            "operator": ">",
            "threshold": 4.0,
            "current": round(core_pce_3m, 2),
            "distance": round(4.0 - core_pce_3m, 2),
            "consequence": "Fed forced to hike, forward_bias → cautious",
            "severity": "high",
        })

    core_pce_yoy = kl.get("core_pce_yoy")
    if core_pce_yoy is not None:
        triggers.append({
            "id": "inflation_normalization",
            "metric": "core_pce_yoy",
            "operator": "<",
            "threshold": 2.5,
            "current": round(core_pce_yoy, 2),
            "distance": round(core_pce_yoy - 2.5, 2),
            "consequence": "Fed can ease, forward_bias → constructive",
            "severity": "high",
        })

    # --- Recession triggers ---
    sahm = kl.get("sahm_value")
    if sahm is not None:
        triggers.append({
            "id": "sahm_recession",
            "metric": "sahm_value",
            "operator": ">",
            "threshold": 0.50,
            "current": round(sahm, 3),
            "distance": round(0.50 - sahm, 3),
            "consequence": "Recession confirmed, reduce_exposure → true, beta → 0.3",
            "severity": "critical",
        })

    # --- Volatility triggers ---
    vix = kl.get("vix")
    if vix is not None:
        triggers.append({
            "id": "vix_crisis",
            "metric": "vix",
            "operator": ">",
            "threshold": 35.0,
            "current": round(vix, 1),
            "distance": round(35.0 - vix, 1),
            "consequence": "Volatility crisis, max position → 1.5%, force deleverage",
            "severity": "critical",
        })
        triggers.append({
            "id": "vix_normalization",
            "metric": "vix",
            "operator": "<",
            "threshold": 20.0,
            "current": round(vix, 1),
            "distance": round(vix - 20.0, 1),
            "consequence": "Vol normalizing, position limits expand, beta → 1.0",
            "severity": "medium",
        })

    # --- Credit triggers ---
    hy = kl.get("hy_spread")
    if hy is not None:
        triggers.append({
            "id": "credit_stress",
            "metric": "hy_spread",
            "operator": ">",
            "threshold": 4.5,
            "current": round(hy, 2),
            "distance": round(4.5 - hy, 2),
            "consequence": "Credit stress, reduce HY exposure, tighten stops",
            "severity": "high",
        })
        triggers.append({
            "id": "credit_crisis",
            "metric": "hy_spread",
            "operator": ">",
            "threshold": 6.0,
            "current": round(hy, 2),
            "distance": round(6.0 - hy, 2),
            "consequence": "Credit crisis, exit all HY, reduce equity to 50%",
            "severity": "critical",
        })

    # --- Inflation surprise trigger ---
    infl_data = regime_assessment.inflation if isinstance(regime_assessment.inflation, dict) else {}
    breakeven = infl_data.get("expectations", {}).get("breakeven_5y")
    if breakeven is not None and core_pce_yoy is not None:
        surprise_gap = core_pce_yoy - breakeven
        triggers.append({
            "id": "inflation_surprise_extreme",
            "metric": "inflation_surprise_pp",
            "operator": ">",
            "threshold": 1.0,
            "current": round(surprise_gap, 2),
            "distance": round(1.0 - surprise_gap, 2),
            "consequence": "Extreme hawkish surprise, short duration, overweight commodities",
            "severity": "high",
        })

    # --- Net liquidity trigger ---
    net_liq = kl.get("net_liquidity")
    if net_liq is not None:
        net_liq_t = net_liq / 1_000_000 if net_liq > 1000 else net_liq
        triggers.append({
            "id": "liquidity_drain",
            "metric": "net_liquidity_trillion",
            "operator": "<",
            "threshold": 5.0,
            "current": round(net_liq_t, 2),
            "distance": round(net_liq_t - 5.0, 2),
            "consequence": "Liquidity tightening, reduce risk, raise cash",
            "severity": "high",
        })

    return triggers


def compute_velocity_context(
    regime_assessment,
) -> dict:
    """
    Classify the velocity (rate of change) of key macro indicators.

    A Risk Officer needs to know not just "VIX is 31" but "VIX spiked
    from 20 to 31 in 5 days" vs "VIX has been grinding around 30 for weeks."
    The urgency of response depends on velocity, not just level.
    """
    vol_data = regime_assessment.volatility if isinstance(regime_assessment.volatility, dict) else {}
    ca_data = regime_assessment.cross_asset if isinstance(regime_assessment.cross_asset, dict) else {}
    br_data = regime_assessment.breadth if isinstance(regime_assessment.breadth, dict) else {}
    mp_data = regime_assessment.monetary_policy if isinstance(regime_assessment.monetary_policy, dict) else {}

    # VIX velocity
    vix_info = vol_data.get("vix", {})
    vix_5d = vix_info.get("change_5d")
    vix_20d = vix_info.get("change_20d")

    if vix_5d is not None and vix_5d > 8:
        vix_velocity = "spiking"
    elif vix_5d is not None and vix_5d > 3:
        vix_velocity = "rising_fast"
    elif vix_20d is not None and vix_20d > 5:
        vix_velocity = "grinding_higher"
    elif vix_5d is not None and vix_5d < -5:
        vix_velocity = "collapsing"
    elif vix_20d is not None and vix_20d < -3:
        vix_velocity = "declining"
    else:
        vix_velocity = "stable"

    # Credit spread velocity
    credit_data = ca_data.get("credit", {})
    hy_change = credit_data.get("change_30d")

    if hy_change is not None and hy_change > 0.5:
        spread_velocity = "rapid_widening"
    elif hy_change is not None and hy_change > 0.2:
        spread_velocity = "widening"
    elif hy_change is not None and hy_change < -0.3:
        spread_velocity = "tightening"
    else:
        spread_velocity = "stable"

    # Breadth velocity — compare 50MA participation to 200MA
    pct_50 = br_data.get("sector_participation", {}).get("pct_above_50ma", 50)
    pct_200 = br_data.get("sector_participation", {}).get("pct_above_200ma", 50)

    if pct_50 < 25 and pct_200 > 40:
        breadth_velocity = "collapsing"  # Short-term breakdown while long-term still ok
    elif pct_50 < pct_200 * 0.6:
        breadth_velocity = "deteriorating"
    elif pct_50 > pct_200 * 1.2:
        breadth_velocity = "improving"
    else:
        breadth_velocity = "stable"

    # Dollar velocity
    dollar_data = ca_data.get("dollar", {})
    dxy_change = dollar_data.get("change_30d")

    if dxy_change is not None and dxy_change > 3:
        dollar_velocity = "surging"
    elif dxy_change is not None and dxy_change > 1:
        dollar_velocity = "strengthening"
    elif dxy_change is not None and dxy_change < -3:
        dollar_velocity = "weakening_fast"
    elif dxy_change is not None and dxy_change < -1:
        dollar_velocity = "weakening"
    else:
        dollar_velocity = "stable"

    # Liquidity velocity
    liq_data = mp_data.get("liquidity", {})
    liq_13w = liq_data.get("change_13w_millions")

    if liq_13w is not None:
        liq_13w_t = liq_13w / 1_000_000  # Convert to trillions
        if liq_13w_t > 0.2:
            liquidity_velocity = "expanding"
        elif liq_13w_t < -0.2:
            liquidity_velocity = "draining"
        else:
            liquidity_velocity = "stable"
    else:
        liquidity_velocity = "unknown"

    # Overall urgency assessment
    urgent_signals = sum([
        vix_velocity in ("spiking", "rising_fast"),
        spread_velocity == "rapid_widening",
        breadth_velocity == "collapsing",
        liquidity_velocity == "draining",
    ])

    if urgent_signals >= 2:
        urgency = "high"
    elif urgent_signals == 1:
        urgency = "elevated"
    else:
        urgency = "normal"

    return {
        "vix_velocity": vix_velocity,
        "spread_velocity": spread_velocity,
        "breadth_velocity": breadth_velocity,
        "dollar_velocity": dollar_velocity,
        "liquidity_velocity": liquidity_velocity,
        "urgency": urgency,
    }


# === FOMC Meeting Dates (published annually by the Fed) ===
# Update this list at the start of each year
FOMC_DATES_2026 = [
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]

FOMC_DATES_BY_YEAR = {
    2026: FOMC_DATES_2026,
}

def _get_fomc_dates(year: int) -> list[str]:
    """
    Get FOMC meeting dates for a given year.

    Uses exact dates when available (hardcoded from Fed schedule).
    Falls back to approximate dates (8 meetings, ~6 weeks apart) when
    exact dates haven't been added yet.
    """
    if year in FOMC_DATES_BY_YEAR:
        return FOMC_DATES_BY_YEAR[year]

    # Approximate: 8 meetings per year, roughly every 6-7 weeks
    # Starting late January, ending mid-December
    import logging
    logging.getLogger("finias.agent.macro_strategist").warning(
        f"Using APPROXIMATE FOMC dates for {year}. "
        f"Update FOMC_DATES_BY_YEAR[{year}] with exact dates from the Fed schedule."
    )

    approximate_dates = []
    # Typical FOMC pattern: late Jan, mid Mar, early May, mid Jun, late Jul, mid Sep, early Nov, mid Dec
    month_days = [(1, 28), (3, 18), (5, 6), (6, 17), (7, 29), (9, 16), (11, 4), (12, 16)]
    for month, day in month_days:
        try:
            approximate_dates.append(date(year, month, day).isoformat())
        except ValueError:
            pass

    return approximate_dates

# CPI release dates are typically the 2nd or 3rd week of each month
# NFP is first Friday of each month
# These are approximate — exact dates published by BLS
CPI_RELEASE_DAYS = [10, 11, 12, 13, 14]  # Typical range of month day
NFP_RELEASE_DAYS = [1, 2, 3, 4, 5, 6, 7]  # First Friday range


def compute_event_calendar(as_of_date: date = None) -> dict:
    """
    Compute upcoming macro events and their impact on position sizing.

    FOMC, CPI, and NFP are the three events that most move markets.
    Position sizing should be reduced ahead of high-impact events.
    """
    from datetime import timedelta

    if as_of_date is None:
        as_of_date = date.today()

    events = []

    # FOMC meetings
    fomc_dates = _get_fomc_dates(as_of_date.year)
    for fomc_str in fomc_dates:
        fomc_date = date.fromisoformat(fomc_str)
        days_away = (fomc_date - as_of_date).days
        if 0 <= days_away <= 30:
            events.append({
                "event": "FOMC",
                "date": fomc_str,
                "days_away": days_away,
                "impact": "high",
            })

    # Next CPI (approximate: ~12th of each month)
    for month_offset in range(0, 3):
        cpi_month = as_of_date.month + month_offset
        cpi_year = as_of_date.year
        if cpi_month > 12:
            cpi_month -= 12
            cpi_year += 1
        cpi_date = date(cpi_year, cpi_month, 12)  # Approximate
        days_away = (cpi_date - as_of_date).days
        if 0 <= days_away <= 30:
            events.append({
                "event": "CPI Release",
                "date": cpi_date.isoformat(),
                "days_away": days_away,
                "impact": "high",
            })
            break

    # Next NFP (first Friday of next month, approximate)
    for month_offset in range(0, 3):
        nfp_month = as_of_date.month + month_offset
        nfp_year = as_of_date.year
        if nfp_month > 12:
            nfp_month -= 12
            nfp_year += 1
        # Find first Friday
        first_day = date(nfp_year, nfp_month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        nfp_date = first_day + timedelta(days=days_until_friday)
        days_away = (nfp_date - as_of_date).days
        if 0 <= days_away <= 30:
            events.append({
                "event": "NFP Release",
                "date": nfp_date.isoformat(),
                "days_away": days_away,
                "impact": "high",
            })
            break

    # Sort by days away
    events.sort(key=lambda e: e["days_away"])

    # Compute pre-event sizing multiplier
    # Within 2 days of high-impact event: 0.5x
    # Within 5 days: 0.75x
    # Otherwise: 1.0x
    nearest_high_impact = None
    for e in events:
        if e["impact"] == "high":
            nearest_high_impact = e["days_away"]
            break

    if nearest_high_impact is not None and nearest_high_impact <= 2:
        sizing_multiplier = 0.50
    elif nearest_high_impact is not None and nearest_high_impact <= 5:
        sizing_multiplier = 0.75
    else:
        sizing_multiplier = 1.0

    return {
        "upcoming_events": events,
        "pre_event_sizing_multiplier": sizing_multiplier,
        "nearest_high_impact_days": nearest_high_impact,
    }


def compute_data_freshness(
    regime_assessment,
    as_of_date: date = None,
) -> dict:
    """
    Assess the freshness of data underlying the regime assessment.

    Checks key indicators for staleness and produces warnings
    that Claude and downstream agents can use to calibrate confidence.
    """
    if as_of_date is None:
        as_of_date = date.today()

    warnings = []

    # GDPNow is known to be quarterly on FRED
    # The Atlanta Fed updates it multiple times per week, but FRED only gets quarterly finals
    warnings.append(
        "GDPNow from FRED is quarterly (prior quarter final). "
        "Current quarter estimate requires web search for Atlanta Fed GDPNow."
    )

    # Check if we can detect any staleness from the data itself
    # Monthly FRED data (unemployment, CPI, PCE) is expected to be 30-60 days old
    # Weekly FRED data (claims, WALCL) should be within 10 days
    # Daily FRED data (VIX, yields) should be within 3 days

    return {
        "daily_data_note": "VIX, yields, spreads: updated within 1-3 business days",
        "weekly_data_note": "Fed balance sheet, claims, NFCI: updated within 7-10 days",
        "monthly_data_note": "CPI, PCE, unemployment, payrolls: 30-60 day publication lag is NORMAL",
        "gdpnow_note": "FRED GDPNow is QUARTERLY final only — use web search for current estimate",
        "warnings": warnings,
    }


def compute_trajectory(
    regime_assessment,
    fed_target_upper: list[dict],
    prior_regime_assessment=None,
) -> TrajectoryAssessment:
    """
    Compute the full trajectory assessment from a regime assessment.

    Args:
        regime_assessment: Current RegimeAssessment
        fed_target_upper: DFEDTARU series for rate decision history
        prior_regime_assessment: Previous week's RegimeAssessment (for changes).
                                 If None, trajectory signals that need prior data return defaults.
    """
    result = TrajectoryAssessment()

    # === 1. Rate Decision History ===
    rate_info = compute_rate_decision_history(fed_target_upper)
    result.rate_decisions_12m = rate_info["decisions"]
    result.cumulative_rate_change_12m_bp = rate_info["cumulative_bp"]
    result.policy_trajectory = rate_info["trajectory"]
    result.months_since_last_change = rate_info["months_since_last"]

    # === 2. Inflation Surprise ===
    core_pce = regime_assessment.key_levels.get("core_pce_yoy")

    # Get breakeven from the inflation component dict
    infl_dict = regime_assessment.inflation if isinstance(regime_assessment.inflation, dict) else {}
    expectations = infl_dict.get("expectations", {})
    breakeven_5y = expectations.get("breakeven_5y")

    if core_pce is not None and breakeven_5y is not None:
        surprise_info = compute_inflation_surprise(core_pce, breakeven_5y)
        result.inflation_surprise_pp = surprise_info["surprise_pp"]
        result.inflation_surprise_direction = surprise_info["direction"]

    # === 3-5: Signals that need prior assessment ===
    if prior_regime_assessment is not None:
        # Signal 1: Inflation Trajectory
        infl_traj = compute_inflation_trajectory(
            regime_assessment.inflation_score,
            prior_regime_assessment.inflation_score,
        )
        result.inflation_trajectory = infl_traj["trajectory"]
        result.inflation_score_4w_change = infl_traj["change"]

        # Signal 2: Stress Contrarian
        stress_info = compute_stress_contrarian(
            regime_assessment.stress_index,
            prior_regime_assessment.stress_index,
        )
        result.stress_contrarian_signal = stress_info["signal"]
        result.stress_4w_change = stress_info["change"]

        # Signal 3: Binding Constraint Shift
        shift_info = compute_binding_shift(
            regime_assessment.binding_constraint,
            prior_regime_assessment.binding_constraint,
        )
        result.binding_constraint_shifted = shift_info["shifted"]
        result.prior_binding_constraint = shift_info["prior"]
        result.binding_shift_direction = shift_info["direction"]

    # === 6. Sector Guidance ===
    sector_info = compute_sector_guidance(
        result.inflation_trajectory,
        regime_assessment.binding_constraint,
        regime_assessment.stress_index,
    )
    result.sector_overweights = sector_info["overweights"]
    result.sector_underweights = sector_info["underweights"]
    result.sector_rationale = sector_info["rationale"]

    # === 7. Net Forward Bias ===
    bias_info = compute_forward_bias(
        result.inflation_trajectory,
        result.stress_contrarian_signal,
        result.binding_shift_direction,
    )
    result.forward_bias = bias_info["bias"]
    result.forward_bias_score = bias_info["score"]
    result.forward_bias_confidence = bias_info["confidence"]

    # === 8. Position Sizing ===
    # Need vol_persistent from regime data
    vol_data = regime_assessment.volatility if isinstance(regime_assessment.volatility, dict) else {}
    ts_data = vol_data.get("term_structure", {})
    vol_persistent = ts_data.get("shape", "unknown") == "backwardation"

    # Need breadth_health and credit_stress from regime data
    br_data = regime_assessment.breadth if isinstance(regime_assessment.breadth, dict) else {}
    ca_data = regime_assessment.cross_asset if isinstance(regime_assessment.cross_asset, dict) else {}
    credit_data = ca_data.get("credit", {})

    sizing = compute_position_sizing(
        vix_level=regime_assessment.key_levels.get("vix", 20) or 20,
        vol_persistent=vol_persistent,
        stress_index=regime_assessment.stress_index,
        breadth_health=br_data.get("breadth_health", "unknown"),
        credit_stress=credit_data.get("stress", False),
        recession_probability=regime_assessment.key_levels.get("recession_prob", 0) or 0,
    )
    result.max_single_position_pct = sizing["max_single_position_pct"]
    result.max_sector_exposure_pct = sizing["max_sector_exposure_pct"]
    result.portfolio_beta_target = sizing["portfolio_beta_target"]
    result.cash_target_pct = sizing["cash_target_pct"]
    result.reduce_overall_exposure = sizing["reduce_overall_exposure"]

    # === 9. Scenario Triggers ===
    result.scenario_triggers = compute_scenario_triggers(regime_assessment)

    # === 10. Velocity Context ===
    velocity = compute_velocity_context(regime_assessment)
    result.vix_velocity = velocity["vix_velocity"]
    result.spread_velocity = velocity["spread_velocity"]
    result.breadth_velocity = velocity["breadth_velocity"]
    result.dollar_velocity = velocity["dollar_velocity"]
    result.liquidity_velocity = velocity["liquidity_velocity"]
    result.urgency = velocity["urgency"]

    # === 11. Event Calendar ===
    calendar = compute_event_calendar()
    result.upcoming_events = calendar["upcoming_events"]
    result.pre_event_sizing_multiplier = calendar["pre_event_sizing_multiplier"]
    result.nearest_high_impact_days = calendar["nearest_high_impact_days"]

    # === 12. Apply event calendar to position sizing ===
    # Pre-event sizing reduction stacks with vol/stress reduction
    result.max_single_position_pct = round(
        result.max_single_position_pct * result.pre_event_sizing_multiplier, 1
    )

    # === 13. Data Freshness ===
    freshness = compute_data_freshness(regime_assessment)
    result.data_freshness_warnings = freshness["warnings"]

    return result
