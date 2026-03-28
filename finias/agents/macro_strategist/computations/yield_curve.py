"""
Yield Curve Analysis Module

Analyzes the US Treasury yield curve for regime signals.
Key metrics:
  - 2s10s spread (most watched recession indicator)
  - 3m10y spread (Fed's preferred recession indicator)
  - Curve steepness and shape
  - Rate of change (is inversion deepening or healing?)
  - Historical context for current levels

All computation is pure Python. No API calls. No Claude.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class YieldCurveAnalysis:
    """Complete yield curve assessment."""
    # Current levels
    t3m: Optional[float]           # 3-month T-bill
    t2y: Optional[float]           # 2-year yield
    t5y: Optional[float]           # 5-year yield
    t10y: Optional[float]          # 10-year yield
    t30y: Optional[float]          # 30-year yield

    # Spreads
    spread_2s10s: Optional[float]  # 10Y - 2Y
    spread_3m10y: Optional[float]  # 10Y - 3M
    spread_2s30s: Optional[float]  # 30Y - 2Y

    # Dynamics
    spread_2s10s_change_30d: Optional[float]  # How spread moved over 30 days
    spread_2s10s_change_90d: Optional[float]  # How spread moved over 90 days

    # Inversion tracking
    is_2s10s_inverted: bool
    is_3m10y_inverted: bool
    inversion_depth_2s10s: float   # How deep (negative = inverted)
    days_inverted_2s10s: int       # Consecutive days inverted

    # Shape classification
    curve_shape: str               # normal, flat, inverted, bear_steepening, bull_flattening

    # Scores
    recession_signal_score: float  # 0 to 1 — higher = more recession risk

    # Real yields (TIPS)
    real_yield_5y: Optional[float] = None
    real_yield_10y: Optional[float] = None
    real_yield_10y_change_30d: Optional[float] = None

    # Term premium
    term_premium_10y: Optional[float] = None
    term_premium_trend: Optional[str] = None     # rising, stable, falling

    # Enhanced score
    yield_curve_score: float = 0.0               # -1 to +1

    def to_dict(self) -> dict:
        return {
            "yields": {
                "3m": self.t3m, "2y": self.t2y, "5y": self.t5y,
                "10y": self.t10y, "30y": self.t30y
            },
            "spreads": {
                "2s10s": self.spread_2s10s,
                "3m10y": self.spread_3m10y,
                "2s30s": self.spread_2s30s
            },
            "dynamics": {
                "2s10s_change_30d": self.spread_2s10s_change_30d,
                "2s10s_change_90d": self.spread_2s10s_change_90d,
            },
            "inversion": {
                "is_2s10s_inverted": self.is_2s10s_inverted,
                "is_3m10y_inverted": self.is_3m10y_inverted,
                "depth_2s10s": self.inversion_depth_2s10s,
                "days_inverted_2s10s": self.days_inverted_2s10s,
            },
            "curve_shape": self.curve_shape,
            "recession_signal_score": self.recession_signal_score,
            "real_yields": {
                "5y": self.real_yield_5y,
                "10y": self.real_yield_10y,
                "10y_change_30d": self.real_yield_10y_change_30d,
            },
            "term_premium": {
                "10y": self.term_premium_10y,
                "trend": self.term_premium_trend,
            },
            "yield_curve_score": self.yield_curve_score,
        }


def analyze_yield_curve(
    yields_2y: list[dict],
    yields_5y: list[dict],
    yields_10y: list[dict],
    yields_30y: list[dict],
    yields_3m: list[dict],
    fed_funds: list[dict],
    # NEW parameters
    real_yields_5y: list[dict] = None,
    real_yields_10y: list[dict] = None,
    term_premium_10y: list[dict] = None,
) -> YieldCurveAnalysis:
    """
    Perform complete yield curve analysis.

    Args:
        Each argument is a list of {"date": str, "value": float} dicts
        sorted by date ascending (oldest first).
    """
    # Get latest values
    t3m = _latest(yields_3m)
    t2y = _latest(yields_2y)
    t5y = _latest(yields_5y)
    t10y = _latest(yields_10y)
    t30y = _latest(yields_30y)

    # Compute spreads
    spread_2s10s = (t10y - t2y) if (t10y is not None and t2y is not None) else None
    spread_3m10y = (t10y - t3m) if (t10y is not None and t3m is not None) else None
    spread_2s30s = (t30y - t2y) if (t30y is not None and t2y is not None) else None

    # Compute historical spread changes
    spread_2s10s_30d = _spread_change(yields_10y, yields_2y, 30)
    spread_2s10s_90d = _spread_change(yields_10y, yields_2y, 90)

    # Inversion tracking
    is_2s10s_inv = spread_2s10s is not None and spread_2s10s < 0
    is_3m10y_inv = spread_3m10y is not None and spread_3m10y < 0
    inv_depth = spread_2s10s if spread_2s10s is not None else 0.0
    days_inv = _count_inversion_days(yields_10y, yields_2y)

    # Classify curve shape
    shape = _classify_shape(t3m, t2y, t5y, t10y, t30y, spread_2s10s_30d)

    # Recession signal score
    recession_score = _compute_recession_score(
        spread_2s10s, spread_3m10y, days_inv, spread_2s10s_30d
    )

    # Real yields
    real_5y = _latest(real_yields_5y) if real_yields_5y else None
    real_10y = _latest(real_yields_10y) if real_yields_10y else None
    real_10y_30d = None
    if real_yields_10y and len(real_yields_10y) > 20:
        real_10y_30d = real_yields_10y[-1]["value"] - real_yields_10y[-20]["value"]

    # Term premium
    tp_10y = _latest(term_premium_10y) if term_premium_10y else None
    tp_trend = None
    if term_premium_10y and len(term_premium_10y) > 20:
        tp_change = term_premium_10y[-1]["value"] - term_premium_10y[-20]["value"]
        if tp_change > 0.1:
            tp_trend = "rising"
        elif tp_change < -0.1:
            tp_trend = "falling"
        else:
            tp_trend = "stable"

    # Enhanced yield curve score
    yc_score = _compute_yield_curve_score(
        spread_2s10s, spread_3m10y, recession_score,
        real_10y, real_10y_30d, tp_10y
    )

    return YieldCurveAnalysis(
        t3m=t3m, t2y=t2y, t5y=t5y, t10y=t10y, t30y=t30y,
        spread_2s10s=spread_2s10s,
        spread_3m10y=spread_3m10y,
        spread_2s30s=spread_2s30s,
        spread_2s10s_change_30d=spread_2s10s_30d,
        spread_2s10s_change_90d=spread_2s10s_90d,
        is_2s10s_inverted=is_2s10s_inv,
        is_3m10y_inverted=is_3m10y_inv,
        inversion_depth_2s10s=inv_depth,
        days_inverted_2s10s=days_inv,
        curve_shape=shape,
        recession_signal_score=recession_score,
        real_yield_5y=real_5y,
        real_yield_10y=real_10y,
        real_yield_10y_change_30d=real_10y_30d,
        term_premium_10y=tp_10y,
        term_premium_trend=tp_trend,
        yield_curve_score=yc_score,
    )


def _latest(series: list[dict]) -> Optional[float]:
    """Get most recent value from a series."""
    if not series:
        return None
    return series[-1]["value"]


def _value_n_days_ago(series: list[dict], n: int) -> Optional[float]:
    """Get value approximately N days ago."""
    if len(series) < 2:
        return None
    target_idx = max(0, len(series) - n)
    return series[target_idx]["value"]


def _spread_change(series_long: list[dict], series_short: list[dict], days: int) -> Optional[float]:
    """Compute how a spread changed over N days."""
    if not series_long or not series_short:
        return None

    current_spread = series_long[-1]["value"] - series_short[-1]["value"]

    idx = max(0, min(len(series_long), len(series_short)) - days)
    if idx >= len(series_long) or idx >= len(series_short):
        return None

    past_spread = series_long[idx]["value"] - series_short[idx]["value"]
    return current_spread - past_spread


def _count_inversion_days(series_10y: list[dict], series_2y: list[dict]) -> int:
    """Count consecutive days the 2s10s has been inverted (from most recent)."""
    if not series_10y or not series_2y:
        return 0

    # Align series by using min length from the end
    min_len = min(len(series_10y), len(series_2y))
    s10 = series_10y[-min_len:]
    s2 = series_2y[-min_len:]

    count = 0
    for i in range(len(s10) - 1, -1, -1):
        if s10[i]["value"] < s2[i]["value"]:
            count += 1
        else:
            break
    return count


def _classify_shape(
    t3m: Optional[float], t2y: Optional[float], t5y: Optional[float],
    t10y: Optional[float], t30y: Optional[float],
    spread_change_30d: Optional[float]
) -> str:
    """
    Classify the yield curve shape.

    Definitions:
    - normal: upward sloping, spread positive and stable
    - flat: minimal spread across maturities (±15bp)
    - inverted: downward sloping (short rates above long)
    - bear_steepening: spread positive AND widening — long rates rising faster
      Signals: growth/inflation fears, term premium rising
    - bull_steepening: spread was negative, becoming less negative — short rates
      falling faster. Signals: rate cuts expected, un-inversion in progress
    - bear_flattening: spread positive AND narrowing because short rates rising
      Signals: tightening expectations
    - bull_flattening: spread positive AND narrowing because long rates falling
      Signals: flight to safety, growth concerns
    """
    if t2y is None or t10y is None:
        return "unknown"

    spread = t10y - t2y

    if abs(spread) < 0.15:
        return "flat"

    if spread < -0.15:
        # Curve is inverted
        if spread_change_30d is not None and spread_change_30d > 0.10:
            # Becoming less inverted — un-inversion (bull steepening)
            return "bull_steepening"
        return "inverted"

    # Spread is positive (normal territory)
    if spread_change_30d is not None:
        if spread_change_30d > 0.15:
            # Spread widening with positive spread
            return "bear_steepening"
        elif spread_change_30d < -0.15:
            # Spread narrowing with positive spread
            return "bull_flattening"

    return "normal"


def _compute_recession_score(
    spread_2s10s: Optional[float],
    spread_3m10y: Optional[float],
    days_inverted: int,
    spread_change_30d: Optional[float]
) -> float:
    """
    Compute a 0-1 recession probability signal from yield curve data.

    Based on research:
    - Inversion alone is a warning (0.3-0.5)
    - Sustained inversion (>60 days) elevates risk (0.5-0.7)
    - Un-inversion after sustained inversion is historically the
      highest risk period (0.7-0.9) — recession typically follows
      the un-inversion, not the initial inversion.
    """
    score = 0.0

    if spread_2s10s is not None:
        if spread_2s10s < 0:
            # Currently inverted
            score += 0.3
            # Deeper inversion = higher score
            score += min(0.2, abs(spread_2s10s) * 0.5)
        elif days_inverted > 0 and spread_change_30d is not None and spread_change_30d > 0:
            # Was inverted, now un-inverting — historically most dangerous
            score += 0.7

    if spread_3m10y is not None and spread_3m10y < 0:
        score += 0.15  # Fed's preferred indicator also inverted

    if days_inverted > 180:
        score += 0.15
    elif days_inverted > 60:
        score += 0.1

    return min(1.0, score)


def _compute_yield_curve_score(
    spread_2s10s, spread_3m10y, recession_score,
    real_10y, real_10y_change_30d, term_premium
) -> float:
    """
    Comprehensive yield curve score: -1 (bearish) to +1 (bullish).

    Incorporates: curve shape, recession signal, real yields, term premium.
    Rising real yields compress equity multiples (bearish).
    Rising term premium = market demanding more duration compensation.
    """
    score = 0.0

    # Curve shape (existing logic)
    if spread_2s10s is not None:
        if spread_2s10s > 0.5:
            score += 0.3
        elif spread_2s10s > 0:
            score += 0.1
        elif spread_2s10s > -0.5:
            score -= 0.2
        else:
            score -= 0.4

    # Recession signal
    score -= recession_score * 0.3

    # Real yields (rising real yields = bearish for equities)
    if real_10y_change_30d is not None:
        if real_10y_change_30d > 0.3:
            score -= 0.2  # Sharp rise in real yields
        elif real_10y_change_30d < -0.3:
            score += 0.15  # Falling real yields = supportive

    # Term premium
    if term_premium is not None:
        if term_premium > 0.5:
            score -= 0.1  # High term premium = market worried about duration
        elif term_premium < -0.5:
            score += 0.05  # Negative = flight to safety (mixed signal)

    return max(-1.0, min(1.0, score))
