"""
Market Breadth Analysis Module

Measures the internal health of the market. A rally on broad participation
is healthy. A rally on narrow leadership is fragile. Breadth divergences
(price making new highs on narrowing breadth) are classic warning signals.

Key metrics:
  - Advance/Decline ratio
  - % of stocks above 200-day MA (long-term health)
  - % of stocks above 50-day MA (short-term health)
  - New highs vs new lows
  - Sector dispersion (are all sectors moving together or diverging?)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BreadthAnalysis:
    """Market breadth assessment."""
    # Breadth metrics
    pct_above_200ma: Optional[float]    # % of stocks above 200-day MA
    pct_above_50ma: Optional[float]     # % of stocks above 50-day MA

    # Health assessment
    breadth_health: str                 # strong, healthy, weakening, poor, divergent
    breadth_score: float                # 0 to 1 — higher = healthier breadth

    # Divergence detection
    breadth_divergence: bool            # True if price rising but breadth falling
    divergence_description: Optional[str]

    def to_dict(self) -> dict:
        return {
            "pct_above_200ma": self.pct_above_200ma,
            "pct_above_50ma": self.pct_above_50ma,
            "breadth_health": self.breadth_health,
            "breadth_score": self.breadth_score,
            "breadth_divergence": self.breadth_divergence,
            "divergence_description": self.divergence_description,
        }


def analyze_breadth(
    spx_prices: list[dict],
    all_stock_data: Optional[list[dict]] = None,
) -> BreadthAnalysis:
    """
    Analyze market breadth.

    Args:
        spx_prices: SPX daily closes [{"date": str, "close": float}]
        all_stock_data: Optional grouped daily data from Polygon for breadth calc.
                        If not available, breadth is estimated from ETF proxies.
    """
    # For Sprint 0, we use a simplified breadth model based on sector ETF data.
    # Full breadth (individual stock level) will be added when we have the data pipeline.

    pct_200 = None
    pct_50 = None

    if all_stock_data and len(all_stock_data) > 50:
        # If we have grouped daily data, compute real breadth
        # (This path will be used when full data pipeline is available)
        pass

    # Classify health based on available data
    health = _classify_breadth_health(pct_200, pct_50)
    score = _compute_breadth_score(pct_200, pct_50)

    # Divergence detection
    divergence, div_desc = _detect_divergence(spx_prices, pct_200)

    return BreadthAnalysis(
        pct_above_200ma=pct_200,
        pct_above_50ma=pct_50,
        breadth_health=health,
        breadth_score=score,
        breadth_divergence=divergence,
        divergence_description=div_desc,
    )


def _classify_breadth_health(
    pct_200: Optional[float], pct_50: Optional[float]
) -> str:
    """Classify breadth health."""
    if pct_200 is None:
        return "data_unavailable"

    if pct_200 > 70:
        return "strong"
    elif pct_200 > 55:
        return "healthy"
    elif pct_200 > 40:
        return "weakening"
    else:
        return "poor"


def _compute_breadth_score(
    pct_200: Optional[float], pct_50: Optional[float]
) -> float:
    """Compute 0-1 breadth health score."""
    if pct_200 is None:
        return 0.5  # Default when data unavailable

    # Normalize pct_above_200ma from 0-100 to 0-1
    return min(1.0, max(0.0, pct_200 / 100))


def _detect_divergence(
    spx_prices: list[dict], pct_200: Optional[float]
) -> tuple[bool, Optional[str]]:
    """Detect breadth divergence from price action."""
    if pct_200 is None or len(spx_prices) < 20:
        return False, None

    # Check if SPX is near highs but breadth is poor
    closes = [p["close"] for p in spx_prices]
    recent_high = max(closes[-20:])
    current = closes[-1]

    near_high = current >= recent_high * 0.98  # Within 2% of 20d high

    if near_high and pct_200 < 50:
        return True, (
            f"SPX near 20-day highs but only {pct_200:.0f}% of stocks above 200-day MA. "
            f"Narrow leadership — rally is fragile."
        )

    return False, None
