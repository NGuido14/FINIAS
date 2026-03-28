"""
Cross-Asset Signal Analysis Module

Monitors relationships between asset classes that institutional desks watch:
  - US Dollar (DXY) — strong dollar headwind for risk assets
  - Credit spreads (HY OAS) — credit market's view on default risk
  - Copper/Gold ratio — growth expectations proxy
  - Breakeven inflation — market's inflation expectations
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CrossAssetAnalysis:
    """Cross-asset signal assessment."""
    # Dollar
    dxy_level: Optional[float]
    dxy_trend: Optional[str]          # strengthening, weakening, stable
    dxy_change_30d: Optional[float]

    # Credit
    hy_spread: Optional[float]        # High yield OAS
    hy_spread_trend: Optional[str]    # tightening, widening, stable
    hy_spread_change_30d: Optional[float]
    credit_stress: bool               # True if HY spread > 500bps

    # Inflation
    breakeven_5y: Optional[float]
    breakeven_10y: Optional[float]
    inflation_expectations: str       # anchored, rising, falling

    # Composite
    cross_asset_score: float          # -1 to 1 — positive = risk-on, negative = risk-off

    def to_dict(self) -> dict:
        return {
            "dollar": {
                "dxy": self.dxy_level,
                "trend": self.dxy_trend,
                "change_30d": self.dxy_change_30d,
            },
            "credit": {
                "hy_spread": self.hy_spread,
                "trend": self.hy_spread_trend,
                "change_30d": self.hy_spread_change_30d,
                "stress": self.credit_stress,
            },
            "inflation": {
                "breakeven_5y": self.breakeven_5y,
                "breakeven_10y": self.breakeven_10y,
                "expectations": self.inflation_expectations,
            },
            "cross_asset_score": self.cross_asset_score,
        }


def analyze_cross_assets(
    dxy_series: list[dict],
    hy_spread_series: list[dict],
    breakeven_5y: list[dict],
    breakeven_10y: list[dict],
) -> CrossAssetAnalysis:
    """Analyze cross-asset signals."""
    # Dollar
    dxy = _latest(dxy_series)
    dxy_trend = _classify_trend(dxy_series, 30)
    dxy_30d = _change_over_days(dxy_series, 30)

    # Credit
    hy = _latest(hy_spread_series)
    hy_trend = _classify_trend(hy_spread_series, 30)
    hy_30d = _change_over_days(hy_spread_series, 30)
    credit_stress = hy is not None and hy > 5.0  # 500bps

    # Inflation expectations
    be_5y = _latest(breakeven_5y)
    be_10y = _latest(breakeven_10y)
    infl_exp = _classify_inflation(breakeven_5y, breakeven_10y)

    # Composite score
    score = _compute_cross_asset_score(dxy_trend, hy, hy_trend, credit_stress)

    return CrossAssetAnalysis(
        dxy_level=dxy, dxy_trend=dxy_trend, dxy_change_30d=dxy_30d,
        hy_spread=hy, hy_spread_trend=hy_trend,
        hy_spread_change_30d=hy_30d, credit_stress=credit_stress,
        breakeven_5y=be_5y, breakeven_10y=be_10y,
        inflation_expectations=infl_exp,
        cross_asset_score=score,
    )


def _latest(series: list[dict]) -> Optional[float]:
    if not series:
        return None
    return series[-1]["value"]


def _change_over_days(series: list[dict], days: int) -> Optional[float]:
    if len(series) <= days:
        return None
    return series[-1]["value"] - series[-(days + 1)]["value"]


def _classify_trend(series: list[dict], window: int) -> Optional[str]:
    """Classify trend direction over a window."""
    if len(series) < window:
        return None

    values = [s["value"] for s in series[-window:]]
    change_pct = (values[-1] - values[0]) / abs(values[0]) * 100 if values[0] != 0 else 0

    if change_pct > 2:
        return "strengthening" if "dxy" in str(series) else "widening"
    elif change_pct < -2:
        return "weakening" if "dxy" in str(series) else "tightening"
    return "stable"


def _classify_inflation(be_5y: list[dict], be_10y: list[dict]) -> str:
    """Classify inflation expectations trend."""
    if not be_5y or len(be_5y) < 20:
        return "unknown"

    recent = [s["value"] for s in be_5y[-5:]]
    earlier = [s["value"] for s in be_5y[-25:-20]]

    if not earlier:
        return "unknown"

    avg_recent = np.mean(recent)
    avg_earlier = np.mean(earlier)

    if avg_recent - avg_earlier > 0.15:
        return "rising"
    elif avg_earlier - avg_recent > 0.15:
        return "falling"
    return "anchored"


def _compute_cross_asset_score(
    dxy_trend: Optional[str],
    hy_spread: Optional[float],
    hy_trend: Optional[str],
    credit_stress: bool
) -> float:
    """
    Composite cross-asset score: -1 (risk-off) to +1 (risk-on).
    """
    score = 0.0

    # Weak dollar = risk-on
    if dxy_trend == "weakening":
        score += 0.3
    elif dxy_trend == "strengthening":
        score -= 0.3

    # Tight credit = risk-on
    if credit_stress:
        score -= 0.5
    elif hy_spread is not None:
        if hy_spread < 3.0:
            score += 0.3
        elif hy_spread < 4.0:
            score += 0.1

    if hy_trend == "tightening":
        score += 0.15
    elif hy_trend == "widening":
        score -= 0.15

    return max(-1.0, min(1.0, score))
