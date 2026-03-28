"""
Business Cycle Positioning Module

Determines where we are in the economic cycle: early, mid, late, or recession.
This is the single most important context for sector allocation, factor exposure,
and risk tolerance.

Uses:
  - Conference Board Leading Economic Index (LEI)
  - ISM Manufacturing PMI
  - Labor market signals (claims, JOLTS)
  - Housing (permits, starts)
  - Consumer data (retail sales, sentiment)
  - Sahm Rule for recession detection

All computation is pure Python. No API calls. No Claude.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class BusinessCycleAnalysis:
    """Complete business cycle assessment."""

    # Cycle phase
    cycle_phase: str = "unknown"                    # early_cycle, mid_cycle, late_cycle, recession
    phase_confidence: float = 0.5

    # Leading Economic Index
    lei_level: Optional[float] = None
    lei_mom_change: Optional[float] = None
    lei_3m_change: Optional[float] = None
    lei_6m_change: Optional[float] = None
    lei_consecutive_negatives: int = 0              # Months of negative MoM
    lei_trend: str = "unknown"                      # improving, stable, deteriorating

    # PMI
    ism_manufacturing: Optional[float] = None
    ism_direction: Optional[str] = None             # rising, stable, falling
    ism_is_proxy: bool = True                       # True when derived from Philly Fed, not actual ISM
    ism_months_below_50: int = 0
    ism_months_above_50: int = 0

    # Sahm Rule
    sahm_value: float = 0.0
    sahm_triggered: bool = False
    sahm_distance_to_trigger: float = 0.0           # How far from 0.50
    sahm_acceleration: Optional[float] = None       # Rate of change

    # Labor leading signals
    initial_claims_4wk: Optional[float] = None
    initial_claims_trend: str = "unknown"
    continuing_claims: Optional[float] = None
    continuing_claims_trend: str = "unknown"
    jolts_openings: Optional[float] = None
    jolts_quits_rate: Optional[float] = None
    temp_employment_trend: str = "unknown"
    avg_weekly_hours: Optional[float] = None
    avg_weekly_hours_trend: str = "unknown"

    # Housing
    building_permits_trend: str = "unknown"
    housing_starts_trend: str = "unknown"

    # Consumer
    retail_sales_yoy: Optional[float] = None
    consumer_sentiment: Optional[float] = None
    consumer_sentiment_trend: str = "unknown"

    # Activity
    industrial_production_yoy: Optional[float] = None
    capacity_utilization: Optional[float] = None
    cfnai: Optional[float] = None                   # Chicago Fed National Activity

    # GDP Nowcast
    gdp_nowcast: Optional[float] = None             # Atlanta Fed GDPNow (annualized %)
    gdp_nowcast_trend: str = "unknown"              # accelerating, stable, decelerating

    # Composite
    composite_leading: float = 0.0                  # -1 to +1
    recession_probability: float = 0.0              # 0 to 1

    # Sector implications
    favored_sectors: list[str] = field(default_factory=list)
    disfavored_sectors: list[str] = field(default_factory=list)
    factor_regime: str = "unknown"                  # growth, value, quality, defensive

    def to_dict(self) -> dict:
        return {
            "cycle_phase": self.cycle_phase,
            "phase_confidence": self.phase_confidence,
            "lei": {
                "level": self.lei_level,
                "mom_change": self.lei_mom_change,
                "3m_change": self.lei_3m_change,
                "6m_change": self.lei_6m_change,
                "consecutive_negatives": self.lei_consecutive_negatives,
                "trend": self.lei_trend,
            },
            "ism": {
                "manufacturing": self.ism_manufacturing,
                "direction": self.ism_direction,
                "months_below_50": self.ism_months_below_50,
                "months_above_50": self.ism_months_above_50,
                "is_proxy": self.ism_is_proxy,
            },
            "sahm_rule": {
                "value": self.sahm_value,
                "triggered": self.sahm_triggered,
                "distance_to_trigger": self.sahm_distance_to_trigger,
                "acceleration": self.sahm_acceleration,
            },
            "labor_leading": {
                "initial_claims_4wk": self.initial_claims_4wk,
                "initial_claims_trend": self.initial_claims_trend,
                "continuing_claims": self.continuing_claims,
                "continuing_claims_trend": self.continuing_claims_trend,
                "jolts_openings": self.jolts_openings,
                "jolts_quits_rate": self.jolts_quits_rate,
                "temp_employment_trend": self.temp_employment_trend,
                "avg_weekly_hours": self.avg_weekly_hours,
            },
            "housing": {
                "permits_trend": self.building_permits_trend,
                "starts_trend": self.housing_starts_trend,
            },
            "consumer": {
                "retail_sales_yoy": self.retail_sales_yoy,
                "sentiment": self.consumer_sentiment,
                "sentiment_trend": self.consumer_sentiment_trend,
            },
            "activity": {
                "indpro_yoy": self.industrial_production_yoy,
                "capacity_util": self.capacity_utilization,
                "cfnai": self.cfnai,
                "gdp_nowcast": self.gdp_nowcast,
                "gdp_nowcast_trend": self.gdp_nowcast_trend,
            },
            "composite_leading": self.composite_leading,
            "recession_probability": self.recession_probability,
            "favored_sectors": self.favored_sectors,
            "disfavored_sectors": self.disfavored_sectors,
            "factor_regime": self.factor_regime,
        }


def analyze_business_cycle(
    lei_series: list[dict],
    unemployment: list[dict],
    initial_claims: list[dict],
    continuing_claims: list[dict],
    jolts_openings: list[dict],
    jolts_quits: list[dict],
    temp_employment: list[dict],
    avg_weekly_hours: list[dict],
    building_permits: list[dict],
    housing_starts: list[dict],
    retail_sales: list[dict],
    consumer_sentiment: list[dict],
    industrial_production: list[dict],
    capacity_utilization: list[dict],
    cfnai_series: list[dict],
    personal_income: list[dict],
    durable_goods: list[dict],
    nfp_series: list[dict],
    philly_fed: list[dict],
    gdp_nowcast_series: list[dict] = None,
) -> BusinessCycleAnalysis:
    """Perform complete business cycle analysis."""

    result = BusinessCycleAnalysis()

    # --- LEI Analysis ---
    if lei_series:
        result.lei_level = _latest(lei_series)
        result.lei_mom_change = _mom_change(lei_series)
        result.lei_3m_change = _cumulative_change(lei_series, 3)
        result.lei_6m_change = _cumulative_change(lei_series, 6)
        result.lei_consecutive_negatives = _count_consecutive_negatives_mom(lei_series)
        result.lei_trend = _classify_lei_trend(lei_series)

    # --- GDP Nowcast (Atlanta Fed GDPNow) ---
    if gdp_nowcast_series and len(gdp_nowcast_series) >= 1:
        result.gdp_nowcast = gdp_nowcast_series[-1]["value"]
        if len(gdp_nowcast_series) >= 2:
            prev = gdp_nowcast_series[-2]["value"]
            diff = result.gdp_nowcast - prev
            if diff > 0.3:
                result.gdp_nowcast_trend = "accelerating"
            elif diff < -0.3:
                result.gdp_nowcast_trend = "decelerating"
            else:
                result.gdp_nowcast_trend = "stable"

    # --- Custom LEI Proxy (replaces Conference Board LEI) ---
    # Build a composite leading indicator from data we have:
    # 1. Initial claims (inverted — falling claims = improving)
    # 2. Building permits (housing leads GDP by ~4 quarters)
    # 3. Consumer sentiment expectations
    # 4. Yield curve spread (2s10s — inverted curve leads recession)
    # 5. Average weekly hours (hours cut before layoffs)
    custom_lei_components = []

    if initial_claims and len(initial_claims) >= 24:
        # Invert: falling claims = positive signal
        # Use 6-month lookback (24 weeks) aligned with other components
        claims_current = np.mean([c["value"] for c in initial_claims[-4:]])
        claims_past = np.mean([c["value"] for c in initial_claims[-24:-20]])
        if claims_past > 0:
            claims_change = -(claims_current / claims_past - 1)  # Inverted
            custom_lei_components.append(("claims", claims_change, 0.25))

    if building_permits and len(building_permits) >= 6:
        permits_current = building_permits[-1]["value"]
        permits_past = building_permits[-6]["value"]
        if permits_past > 0:
            permits_change = permits_current / permits_past - 1
            custom_lei_components.append(("permits", permits_change, 0.20))

    if consumer_sentiment and len(consumer_sentiment) >= 6:
        sent_current = consumer_sentiment[-1]["value"]
        sent_past = consumer_sentiment[-6]["value"]
        if sent_past > 0:
            sent_change = sent_current / sent_past - 1
            custom_lei_components.append(("sentiment", sent_change, 0.15))

    if avg_weekly_hours and len(avg_weekly_hours) >= 6:
        hours_current = avg_weekly_hours[-1]["value"]
        hours_past = avg_weekly_hours[-6]["value"]
        if hours_past > 0:
            hours_change = hours_current / hours_past - 1
            custom_lei_components.append(("hours", hours_change, 0.15))

    # The yield curve spread contribution comes from the yield curve module
    # but we can approximate it from the 2s10s spread in fred data if available

    if custom_lei_components:
        total_weight = sum(w for _, _, w in custom_lei_components)
        composite_lei = sum(v * w for _, v, w in custom_lei_components) / total_weight

        # Map to LEI-like scale: positive = improving, negative = deteriorating
        result.lei_level = composite_lei * 100  # Scale to percentage-like
        result.lei_mom_change = composite_lei * 100  # Approximate
        result.lei_trend = "improving" if composite_lei > 0.01 else ("deteriorating" if composite_lei < -0.01 else "stable")
        result.lei_consecutive_negatives = 0 if composite_lei >= 0 else 3  # Approximate

    # --- ISM Manufacturing (use Philly Fed as proxy when ISM unavailable) ---
    if philly_fed:
        # Philly Fed diffusion index centered on 0; ISM centered on 50
        philly = _latest(philly_fed)
        if philly is not None:
            # Rough mapping: Philly Fed correlates ~0.7 with ISM Mfg
            result.ism_manufacturing = 50 + (philly * 0.3)  # Approximate
            result.ism_direction = _classify_trend_simple(philly_fed, 3)
            result.ism_months_below_50 = _count_consecutive(
                philly_fed, lambda v: (50 + v * 0.3) < 50
            )
            result.ism_months_above_50 = _count_consecutive(
                philly_fed, lambda v: (50 + v * 0.3) >= 50
            )

    # --- Sahm Rule ---
    sahm_val, sahm_trig = _compute_sahm_rule(unemployment)
    result.sahm_value = sahm_val
    result.sahm_triggered = sahm_trig
    result.sahm_distance_to_trigger = max(0, 0.50 - sahm_val)
    result.sahm_acceleration = _compute_sahm_acceleration(unemployment)

    # --- Labor Leading Signals ---
    if initial_claims:
        result.initial_claims_4wk = _moving_average(initial_claims, 4)
        result.initial_claims_trend = _classify_claims_trend(initial_claims)
    if continuing_claims:
        result.continuing_claims = _latest(continuing_claims)
        result.continuing_claims_trend = _classify_trend_simple(continuing_claims, 8)
    if jolts_openings:
        result.jolts_openings = _latest(jolts_openings)
    if jolts_quits:
        result.jolts_quits_rate = _latest(jolts_quits)
    if temp_employment:
        result.temp_employment_trend = _classify_trend_simple(temp_employment, 3)
    if avg_weekly_hours:
        result.avg_weekly_hours = _latest(avg_weekly_hours)
        result.avg_weekly_hours_trend = _classify_trend_simple(avg_weekly_hours, 3)

    # --- Housing ---
    result.building_permits_trend = _classify_trend_simple(building_permits, 3)
    result.housing_starts_trend = _classify_trend_simple(housing_starts, 3)

    # --- Consumer ---
    result.retail_sales_yoy = _compute_yoy(retail_sales)
    if consumer_sentiment:
        result.consumer_sentiment = _latest(consumer_sentiment)
        result.consumer_sentiment_trend = _classify_trend_simple(consumer_sentiment, 3)

    # --- Activity ---
    result.industrial_production_yoy = _compute_yoy(industrial_production)
    result.capacity_utilization = _latest(capacity_utilization)
    result.cfnai = _latest(cfnai_series)

    # --- Composite Leading Indicator ---
    result.composite_leading = _compute_composite_leading(result)

    # --- Recession Probability ---
    result.recession_probability = _compute_recession_probability(result)

    # --- Cycle Phase Classification ---
    result.cycle_phase, result.phase_confidence = _classify_cycle_phase(result)

    # --- Sector Implications ---
    result.favored_sectors, result.disfavored_sectors, result.factor_regime = (
        _get_sector_implications(result.cycle_phase)
    )

    return result


# --- Sahm Rule ---

def _compute_sahm_rule(unemployment: list[dict]) -> tuple[float, bool]:
    """
    Sahm Rule: recession signal when the 3-month moving average of the
    unemployment rate rises 0.50 percentage points or more above its
    12-month low.

    Has NEVER produced a false positive in its history.
    """
    if not unemployment or len(unemployment) < 15:
        return 0.0, False

    values = [u["value"] for u in unemployment]

    # 3-month moving average of current unemployment
    current_3m_avg = np.mean(values[-3:])

    # 12-month low (using months before the current 3-month window)
    lookback = values[:-3] if len(values) > 15 else values[:-3]
    twelve_month_window = lookback[-12:] if len(lookback) >= 12 else lookback

    if not twelve_month_window:
        return 0.0, False

    twelve_month_low = min(twelve_month_window)

    sahm_value = current_3m_avg - twelve_month_low
    triggered = sahm_value >= 0.50

    return round(sahm_value, 4), triggered


def _compute_sahm_acceleration(unemployment: list[dict]) -> Optional[float]:
    """How fast is the Sahm value moving? Positive = deteriorating."""
    if not unemployment or len(unemployment) < 18:
        return None

    # Sahm value now vs 3 months ago
    current_sahm, _ = _compute_sahm_rule(unemployment)

    # Sahm value 3 months ago (use data up to -3)
    past_sahm, _ = _compute_sahm_rule(unemployment[:-3])

    return round(current_sahm - past_sahm, 4)


# --- LEI Analysis ---

def _mom_change(series: list[dict]) -> Optional[float]:
    if len(series) < 2:
        return None
    return series[-1]["value"] - series[-2]["value"]


def _cumulative_change(series: list[dict], months: int) -> Optional[float]:
    if len(series) <= months:
        return None
    return series[-1]["value"] - series[-(months + 1)]["value"]


def _count_consecutive_negatives_mom(series: list[dict]) -> int:
    """Count consecutive months of negative MoM change (from most recent)."""
    if len(series) < 2:
        return 0
    count = 0
    for i in range(len(series) - 1, 0, -1):
        if series[i]["value"] < series[i - 1]["value"]:
            count += 1
        else:
            break
    return count


def _classify_lei_trend(lei: list[dict]) -> str:
    if len(lei) < 6:
        return "unknown"
    change_6m = lei[-1]["value"] - lei[-6]["value"]
    mom = lei[-1]["value"] - lei[-2]["value"]

    if change_6m > 0 and mom > 0:
        return "improving"
    elif change_6m < -1.0 or (mom < 0 and _count_consecutive_negatives_mom(lei) >= 3):
        return "deteriorating"
    return "stable"


# --- Helper Functions ---

def _latest(series: list[dict]) -> Optional[float]:
    if not series:
        return None
    return series[-1]["value"]


def _moving_average(series: list[dict], n: int) -> Optional[float]:
    if not series or len(series) < n:
        return None
    return float(np.mean([s["value"] for s in series[-n:]]))


def _classify_trend_simple(series: list[dict], lookback: int) -> str:
    """Simple trend classification: rising, stable, falling."""
    if not series or len(series) <= lookback:
        return "unknown"
    recent = np.mean([s["value"] for s in series[-lookback:]])
    earlier = np.mean([s["value"] for s in series[-(lookback * 2):-lookback]]) if len(series) >= lookback * 2 else series[0]["value"]

    pct = (recent - earlier) / abs(earlier) * 100 if earlier != 0 else 0

    if pct > 2:
        return "rising"
    elif pct < -2:
        return "falling"
    return "stable"


def _count_consecutive(series: list[dict], condition) -> int:
    """Count consecutive observations from end matching condition."""
    count = 0
    for i in range(len(series) - 1, -1, -1):
        if condition(series[i]["value"]):
            count += 1
        else:
            break
    return count


def _compute_yoy(series: list[dict]) -> Optional[float]:
    if not series or len(series) < 13:
        return None
    current = series[-1]["value"]
    year_ago = series[-13]["value"]
    if year_ago == 0:
        return None
    return (current - year_ago) / abs(year_ago) * 100


def _classify_claims_trend(claims: list[dict]) -> str:
    """Classify initial claims trend. Rising claims = worsening labor market."""
    if len(claims) < 8:
        return "unknown"
    recent = np.mean([c["value"] for c in claims[-4:]])
    earlier = np.mean([c["value"] for c in claims[-8:-4]])
    pct = (recent - earlier) / earlier * 100 if earlier != 0 else 0

    if pct > 5:
        return "rising"
    elif pct < -5:
        return "falling"
    return "stable"


# --- Composite & Classification ---

def _compute_composite_leading(result: BusinessCycleAnalysis) -> float:
    """
    Compute composite leading indicator: -1 (deep recession signal) to +1 (strong expansion).

    Combines:
    - LEI direction (most comprehensive leading indicator)
    - ISM direction (real-time activity)
    - Claims trend (most timely labor signal)
    - Housing trend (leads by ~4 quarters)
    - Consumer sentiment trend
    - Sahm Rule proximity
    """
    scores = []
    weights = []

    # LEI (weight: 0.25)
    if result.lei_trend != "unknown":
        lei_score = {"improving": 0.6, "stable": 0.0, "deteriorating": -0.6}[result.lei_trend]
        if result.lei_consecutive_negatives >= 6:
            lei_score -= 0.3
        elif result.lei_consecutive_negatives >= 3:
            lei_score -= 0.15
        scores.append(lei_score)
        weights.append(0.25)

    # ISM (weight: 0.20)
    if result.ism_manufacturing is not None:
        ism_score = (result.ism_manufacturing - 50) / 15  # Normalize: 35=-1, 50=0, 65=+1
        ism_score = max(-1, min(1, ism_score))
        scores.append(ism_score)
        weights.append(0.20)

    # Claims (weight: 0.20)
    if result.initial_claims_trend != "unknown":
        claims_score = {"falling": 0.5, "stable": 0.0, "rising": -0.5}[result.initial_claims_trend]
        scores.append(claims_score)
        weights.append(0.20)

    # Housing (weight: 0.15)
    if result.building_permits_trend != "unknown":
        housing_score = {"rising": 0.5, "stable": 0.0, "falling": -0.5}[result.building_permits_trend]
        scores.append(housing_score)
        weights.append(0.15)

    # Sentiment (weight: 0.10)
    if result.consumer_sentiment_trend != "unknown":
        sent_score = {"rising": 0.4, "stable": 0.0, "falling": -0.4}[result.consumer_sentiment_trend]
        scores.append(sent_score)
        weights.append(0.10)

    # Sahm proximity (weight: 0.10)
    sahm_score = -result.sahm_value * 2  # 0.5 → -1.0
    sahm_score = max(-1, min(0.3, sahm_score))
    scores.append(sahm_score)
    weights.append(0.10)

    if not scores:
        return 0.0

    total_weight = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_weight


def _compute_recession_probability(result: BusinessCycleAnalysis) -> float:
    """
    Composite recession probability from multiple indicators.
    """
    prob = 0.0

    # Sahm Rule (most reliable single indicator)
    if result.sahm_triggered:
        prob += 0.50  # If triggered, minimum 50% probability
    elif result.sahm_value > 0.35:
        prob += 0.25
    elif result.sahm_value > 0.25:
        prob += 0.10

    # LEI
    if result.lei_consecutive_negatives >= 6:
        prob += 0.20
    elif result.lei_consecutive_negatives >= 3:
        prob += 0.10

    # ISM
    if result.ism_manufacturing is not None:
        if result.ism_manufacturing < 45:
            prob += 0.15
        elif result.ism_manufacturing < 48:
            prob += 0.08

    # Claims
    if result.initial_claims_trend == "rising":
        prob += 0.08

    # CFNAI (below -0.7 = recession territory per Chicago Fed)
    if result.cfnai is not None and result.cfnai < -0.70:
        prob += 0.12

    return min(1.0, prob)


def _classify_cycle_phase(result: BusinessCycleAnalysis) -> tuple[str, float]:
    """
    Classify current business cycle phase.

    Returns (phase, confidence).
    """
    composite = result.composite_leading
    recession_prob = result.recession_probability

    # Clear recession
    if recession_prob > 0.6 or result.sahm_triggered:
        return "recession", min(0.95, 0.5 + recession_prob)

    # Recession warning
    if recession_prob > 0.35:
        return "late_cycle", 0.6

    # Use composite and specific indicators
    if composite > 0.3:
        # Expansion — but early or mid?
        if result.lei_trend == "improving" and (result.ism_manufacturing or 50) > 52:
            if result.capacity_utilization is not None and result.capacity_utilization < 77:
                return "early_cycle", 0.65
            return "mid_cycle", 0.65
        return "mid_cycle", 0.55

    elif composite > -0.1:
        # Moderate — could be mid or late cycle
        if result.lei_trend == "deteriorating":
            return "late_cycle", 0.6
        return "mid_cycle", 0.5

    else:
        # Negative composite
        if recession_prob > 0.2:
            return "late_cycle", 0.65
        return "late_cycle", 0.55


def _get_sector_implications(
    phase: str,
) -> tuple[list[str], list[str], str]:
    """
    Return sector implications for the current cycle phase.
    Based on historical sector performance by cycle phase.
    """
    implications = {
        "early_cycle": (
            ["financials", "consumer_discretionary", "industrials", "real_estate", "small_caps"],
            ["utilities", "consumer_staples", "healthcare"],
            "growth"
        ),
        "mid_cycle": (
            ["technology", "industrials", "communication_services", "broad_market"],
            ["utilities", "consumer_staples"],
            "growth"
        ),
        "late_cycle": (
            ["healthcare", "consumer_staples", "utilities", "energy", "quality"],
            ["consumer_discretionary", "small_caps", "speculative"],
            "quality"
        ),
        "recession": (
            ["treasuries", "utilities", "consumer_staples", "gold", "cash"],
            ["cyclicals", "small_caps", "financials", "high_beta"],
            "defensive"
        ),
    }

    return implications.get(phase, ([], [], "unknown"))
