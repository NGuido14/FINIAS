"""
Inflation Dynamics Analysis Module

Granular inflation analysis — not just headline CPI, but the components
that drive Fed policy and asset allocation. Distinguishes between
sticky vs flexible prices, goods vs services, and monitors the
wage-price spiral risk.

Key insight: the 3-month annualized rate of core measures is what the
Fed actually watches — it shows current trajectory better than YoY
which carries base effects.

All computation is pure Python. No API calls. No Claude.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class InflationAnalysis:
    """Complete inflation assessment."""

    # Headline
    cpi_yoy: Optional[float] = None
    core_cpi_yoy: Optional[float] = None
    core_cpi_3m_annualized: Optional[float] = None  # Fed's preferred trend
    pce_yoy: Optional[float] = None
    core_pce_yoy: Optional[float] = None
    core_pce_3m_annualized: Optional[float] = None

    # Components
    shelter_yoy: Optional[float] = None
    services_yoy: Optional[float] = None

    # Sticky vs Flexible (Atlanta Fed)
    sticky_cpi_yoy: Optional[float] = None
    flexible_cpi_yoy: Optional[float] = None
    sticky_flexible_gap: Optional[float] = None

    # Trimmed measures
    trimmed_mean_pce: Optional[float] = None        # Dallas Fed Trimmed Mean

    # Market expectations
    breakeven_5y: Optional[float] = None
    breakeven_10y: Optional[float] = None
    forward_5y5y: Optional[float] = None
    expectations_anchored: bool = True

    # Wages
    ahe_yoy: Optional[float] = None
    wage_pressure: str = "unknown"                  # low, moderate, elevated, high

    # Producer prices (leading indicator)
    ppi_yoy: Optional[float] = None
    oil_price: Optional[float] = None
    oil_change_3m: Optional[float] = None

    # Classification
    inflation_regime: str = "unknown"               # disinflation, stable, rising, stagflation
    inflation_trend: str = "unknown"                # accelerating, decelerating, stable
    fed_target_distance: Optional[float] = None     # Core PCE minus 2.0%

    # Risk
    spiral_risk: float = 0.0                        # 0-1 wage-price spiral probability

    # Score
    inflation_score: float = 0.0                    # -1 (deflationary) to +1 (overheating)

    def to_dict(self) -> dict:
        return {
            "headline": {
                "cpi_yoy": self.cpi_yoy,
                "core_cpi_yoy": self.core_cpi_yoy,
                "core_cpi_3m_ann": self.core_cpi_3m_annualized,
                "pce_yoy": self.pce_yoy,
                "core_pce_yoy": self.core_pce_yoy,
                "core_pce_3m_ann": self.core_pce_3m_annualized,
            },
            "components": {
                "shelter_yoy": self.shelter_yoy,
                "services_yoy": self.services_yoy,
            },
            "sticky_flexible": {
                "sticky_cpi_yoy": self.sticky_cpi_yoy,
                "flexible_cpi_yoy": self.flexible_cpi_yoy,
                "gap": self.sticky_flexible_gap,
            },
            "trimmed_mean_pce": self.trimmed_mean_pce,
            "expectations": {
                "breakeven_5y": self.breakeven_5y,
                "breakeven_10y": self.breakeven_10y,
                "forward_5y5y": self.forward_5y5y,
                "anchored": self.expectations_anchored,
            },
            "wages": {
                "ahe_yoy": self.ahe_yoy,
                "pressure": self.wage_pressure,
            },
            "producer": {
                "ppi_yoy": self.ppi_yoy,
                "oil_price": self.oil_price,
                "oil_change_3m": self.oil_change_3m,
            },
            "inflation_regime": self.inflation_regime,
            "inflation_trend": self.inflation_trend,
            "fed_target_distance": self.fed_target_distance,
            "spiral_risk": self.spiral_risk,
            "inflation_score": self.inflation_score,
        }


def analyze_inflation(
    cpi_all: list[dict],
    cpi_core: list[dict],
    cpi_shelter: list[dict],
    cpi_services: list[dict],
    pce: list[dict],
    core_pce: list[dict],
    sticky_cpi: list[dict],
    flexible_cpi: list[dict],
    trimmed_mean: list[dict],
    breakeven_5y: list[dict],
    breakeven_10y: list[dict],
    forward_5y5y: list[dict],
    ppi: list[dict],
    ahe: list[dict],
    oil: list[dict],
) -> InflationAnalysis:
    """Perform complete inflation analysis."""

    result = InflationAnalysis()

    # --- Headline Measures ---
    result.cpi_yoy = _compute_yoy_index(cpi_all)
    result.core_cpi_yoy = _compute_yoy_index(cpi_core)
    result.core_cpi_3m_annualized = _compute_3m_annualized(cpi_core)
    result.pce_yoy = _compute_yoy_index(pce)
    result.core_pce_yoy = _compute_yoy_index(core_pce)
    result.core_pce_3m_annualized = _compute_3m_annualized(core_pce)

    # --- Components ---
    result.shelter_yoy = _compute_yoy_index(cpi_shelter)
    result.services_yoy = _compute_yoy_index(cpi_services)

    # --- Sticky vs Flexible ---
    result.sticky_cpi_yoy = _latest(sticky_cpi)
    result.flexible_cpi_yoy = _latest(flexible_cpi)
    if result.sticky_cpi_yoy is not None and result.flexible_cpi_yoy is not None:
        result.sticky_flexible_gap = result.sticky_cpi_yoy - result.flexible_cpi_yoy

    # --- Trimmed Mean ---
    result.trimmed_mean_pce = _latest(trimmed_mean)

    # --- Expectations ---
    result.breakeven_5y = _latest(breakeven_5y)
    result.breakeven_10y = _latest(breakeven_10y)
    result.forward_5y5y = _latest(forward_5y5y)
    result.expectations_anchored = _check_expectations_anchored(forward_5y5y)

    # --- Wages ---
    result.ahe_yoy = _compute_yoy_index(ahe)
    result.wage_pressure = _classify_wage_pressure(result.ahe_yoy)

    # --- Producer Prices & Oil ---
    result.ppi_yoy = _compute_yoy_index(ppi)
    result.oil_price = _latest(oil)
    result.oil_change_3m = _pct_change(oil, 60) if oil else None  # ~60 trading days

    # --- Fed Target Distance ---
    if result.core_pce_yoy is not None:
        result.fed_target_distance = result.core_pce_yoy - 2.0

    # --- Spiral Risk ---
    result.spiral_risk = _compute_spiral_risk(result)

    # --- Trend ---
    result.inflation_trend = _classify_trend(result)

    # --- Regime ---
    result.inflation_regime = _classify_inflation_regime(result)

    # --- Score ---
    result.inflation_score = _compute_inflation_score(result)

    return result


def _latest(series: list[dict]) -> Optional[float]:
    if not series:
        return None
    return series[-1]["value"]


def _compute_yoy_index(series: list[dict]) -> Optional[float]:
    """
    Compute YoY change for an INDEX series (like CPI, PCE).
    These are reported as index levels, so YoY = (current/year_ago - 1) * 100.
    """
    if not series or len(series) < 13:
        return None
    current = series[-1]["value"]
    year_ago = series[-13]["value"]  # ~12 months for monthly data
    if year_ago == 0:
        return None
    return (current / year_ago - 1) * 100


def _compute_3m_annualized(series: list[dict]) -> Optional[float]:
    """
    Compute 3-month annualized rate of change.

    This is the Fed's preferred near-term inflation trend indicator.
    Formula: ((current / 3_months_ago) ^ 4 - 1) * 100
    """
    if not series or len(series) < 4:
        return None
    current = series[-1]["value"]
    three_months_ago = series[-4]["value"]
    if three_months_ago == 0:
        return None
    return ((current / three_months_ago) ** 4 - 1) * 100


def _pct_change(series: list[dict], n: int) -> Optional[float]:
    if not series or len(series) <= n:
        return None
    current = series[-1]["value"]
    past = series[-(n + 1)]["value"]
    if past == 0:
        return None
    return (current - past) / abs(past) * 100


def _check_expectations_anchored(forward_5y5y: list[dict]) -> bool:
    """
    Inflation expectations are anchored if 5Y5Y forward is between 2.0-2.6%.
    This is the market's long-run inflation expectation — the Fed watches
    this closely. Above 2.6% = unanchoring concern. Below 1.8% = deflation concern.
    """
    if not forward_5y5y:
        return True  # Assume anchored if no data
    val = forward_5y5y[-1]["value"]
    return 1.8 <= val <= 2.6


def _classify_wage_pressure(ahe_yoy: Optional[float]) -> str:
    if ahe_yoy is None:
        return "unknown"
    if ahe_yoy > 5.0:
        return "high"
    elif ahe_yoy > 4.0:
        return "elevated"
    elif ahe_yoy > 3.0:
        return "moderate"
    return "low"


def _compute_spiral_risk(result: InflationAnalysis) -> float:
    """
    Wage-price spiral probability.

    Three conditions for spiral:
    1. Wages accelerating (AHE > 4%)
    2. Services prices following wages up (core services elevated)
    3. Inflation expectations unanchoring (5Y5Y > 2.5%)
    """
    risk = 0.0

    # Wage acceleration
    if result.ahe_yoy is not None:
        if result.ahe_yoy > 5.0:
            risk += 0.35
        elif result.ahe_yoy > 4.0:
            risk += 0.20
        elif result.ahe_yoy > 3.5:
            risk += 0.10

    # Services inflation persistent
    if result.sticky_cpi_yoy is not None and result.sticky_cpi_yoy > 4.0:
        risk += 0.25
    elif result.sticky_cpi_yoy is not None and result.sticky_cpi_yoy > 3.0:
        risk += 0.10

    # Expectations unanchoring
    if not result.expectations_anchored:
        risk += 0.30
    if result.forward_5y5y is not None and result.forward_5y5y > 2.8:
        risk += 0.15

    return min(1.0, risk)


def _classify_trend(result: InflationAnalysis) -> str:
    """Is inflation accelerating, decelerating, or stable?"""
    # Compare 3-month annualized to YoY
    # If 3m > YoY: accelerating (recent months hotter than average)
    # If 3m < YoY: decelerating (recent months cooler)
    core_3m = result.core_pce_3m_annualized or result.core_cpi_3m_annualized
    core_yoy = result.core_pce_yoy or result.core_cpi_yoy

    if core_3m is None or core_yoy is None:
        return "unknown"

    diff = core_3m - core_yoy
    if diff > 0.5:
        return "accelerating"
    elif diff < -0.5:
        return "decelerating"
    return "stable"


def _classify_inflation_regime(result: InflationAnalysis) -> str:
    """Classify the inflation environment."""
    core = result.core_pce_yoy or result.core_cpi_yoy

    if core is None:
        return "unknown"

    # Check for stagflation (would need growth data — use spiral risk as proxy)
    if core > 3.5 and result.spiral_risk > 0.4:
        return "stagflation"

    if core > 3.5:
        return "rising"
    elif core > 1.5:
        if result.inflation_trend == "decelerating":
            return "disinflation"
        return "stable"
    elif core > 0:
        return "disinflation"
    else:
        return "deflation_risk"


def _compute_inflation_score(result: InflationAnalysis) -> float:
    """
    -1 (deflationary concern) to +1 (overheating concern).

    For the regime model: positive score = inflation is a problem (bearish for bonds/equities).
    Negative score = disinflation/deflation (bullish for bonds, mixed for equities).
    """
    score = 0.0

    core = result.core_pce_yoy or result.core_cpi_yoy or 2.0

    # Distance from target
    distance = core - 2.0
    score += distance * 0.2  # Each 1% above target adds 0.2

    # Trend contribution
    if result.inflation_trend == "accelerating":
        score += 0.2
    elif result.inflation_trend == "decelerating":
        score -= 0.15

    # Spiral risk
    score += result.spiral_risk * 0.3

    # Expectations
    if not result.expectations_anchored:
        score += 0.15

    return max(-1.0, min(1.0, score))
