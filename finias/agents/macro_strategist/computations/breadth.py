"""
Market Breadth Analysis Module

Measures the internal health of the market using ETF proxies.
We don't need individual stock data — sector ETFs and the SPY/RSP
(cap-weighted vs equal-weight) ratio reveal breadth dynamics.

Key metrics:
  - SPY/RSP ratio: When cap-weighted outperforms equal-weight, leadership
    is narrow (mega-caps leading). When RSP outperforms, breadth is broad.
  - Sector participation: How many of 11 sectors are above their 50-day
    and 200-day moving averages? More sectors above = healthier breadth.
  - Sector relative strength: Which sectors lead? Rotation from cyclicals
    to defensives signals late-cycle behavior.
  - Sector dispersion: Low dispersion = macro-driven (all moving together).
    High dispersion = stock-picker's market.

All computation is pure Python. No API calls. No Claude.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class BreadthAnalysis:
    """Market breadth assessment using ETF proxies."""

    # SPY vs RSP (cap-weighted vs equal-weight)
    spy_rsp_ratio: Optional[float] = None          # Current ratio
    spy_rsp_ratio_change_20d: Optional[float] = None  # 20-day change in ratio
    spy_rsp_ratio_change_60d: Optional[float] = None  # 60-day change
    narrow_leadership: bool = False                 # True if SPY significantly outperforming RSP

    # Sector participation
    sectors_above_200ma: int = 0                    # Out of 11
    sectors_above_50ma: int = 0                     # Out of 11
    pct_sectors_above_200ma: float = 0.0            # 0-100
    pct_sectors_above_50ma: float = 0.0             # 0-100

    # Sector relative strength
    sector_rankings: dict = field(default_factory=dict)  # sector → {rank, rs_20d, rs_60d}
    leading_sectors: list[str] = field(default_factory=list)    # Top 3 by 20d RS
    lagging_sectors: list[str] = field(default_factory=list)    # Bottom 3 by 20d RS
    cyclical_vs_defensive: float = 0.0              # Positive = cyclicals leading
    rotation_signal: str = "neutral"                # risk_on_rotation, risk_off_rotation, neutral

    # Sector absolute returns (not relative to SPY — for interpretation data notes)
    sector_absolute_returns: dict = field(default_factory=dict)  # symbol → {5d, 20d, 60d}

    # Sector dispersion
    sector_dispersion_20d: Optional[float] = None   # Std dev of 20d sector returns
    dispersion_regime: str = "unknown"              # low, normal, high

    # Health assessment
    breadth_health: str = "unknown"                 # strong, healthy, weakening, poor
    breadth_score: float = 0.5                      # 0 to 1 — higher = healthier

    # Divergence detection
    breadth_divergence: bool = False
    divergence_description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "spy_rsp": {
                "price_ratio": self.spy_rsp_ratio,
                "_price_ratio_note": "SPY price / RSP price. The absolute level (~3.3) is meaningless — only the CHANGE matters. Do not cite the ratio level.",
                "ratio_change_20d": self.spy_rsp_ratio_change_20d,
                "ratio_change_60d": self.spy_rsp_ratio_change_60d,
                "_change_note": "Change in the price ratio. Positive = SPY outperforming RSP (narrow leadership). Negative = RSP outperforming (broad breadth). Typical range: -0.1 to +0.1.",
                "narrow_leadership": self.narrow_leadership,
            },
            "sector_participation": {
                "above_200ma": self.sectors_above_200ma,
                "above_50ma": self.sectors_above_50ma,
                "pct_above_200ma": self.pct_sectors_above_200ma,
                "pct_above_50ma": self.pct_sectors_above_50ma,
            },
            "sector_rotation": {
                "leading": self.leading_sectors,
                "lagging": self.lagging_sectors,
                "cyclical_vs_defensive": self.cyclical_vs_defensive,
                "rotation_signal": self.rotation_signal,
            },
            "sector_returns": self.sector_absolute_returns,
            "dispersion": {
                "dispersion_20d": self.sector_dispersion_20d,
                "regime": self.dispersion_regime,
            },
            "breadth_health": self.breadth_health,
            "breadth_score": self.breadth_score,
            "breadth_divergence": self.breadth_divergence,
            "divergence_description": self.divergence_description,
        }


# Sector classification for rotation analysis
CYCLICAL_SECTORS = {"XLF", "XLI", "XLY", "XLK", "XLB", "XLRE"}
DEFENSIVE_SECTORS = {"XLP", "XLU", "XLV"}
# XLE and XLC don't fit neatly into either — treated as neutral


def analyze_breadth(
    spx_prices: list[dict],
    sector_prices: dict[str, list[dict]] = None,
    rsp_prices: list[dict] = None,
) -> BreadthAnalysis:
    """
    Analyze market breadth using ETF proxies.

    Args:
        spx_prices: SPY daily closes [{"date": str, "close": float}]
        sector_prices: Dict mapping sector ETF symbol to price list
                       e.g., {"XLF": [{"date": ..., "close": ...}, ...]}
        rsp_prices: RSP (equal-weight S&P 500) daily closes
    """
    result = BreadthAnalysis()

    if not spx_prices or len(spx_prices) < 201:
        return result  # Need at least 200 days for MA calculations

    # === SPY/RSP Ratio Analysis ===
    if rsp_prices and len(rsp_prices) >= 60:
        _analyze_spy_rsp(result, spx_prices, rsp_prices)

    # === Sector Analysis ===
    if sector_prices and len(sector_prices) >= 5:
        _analyze_sectors(result, spx_prices, sector_prices)

    # === Divergence Detection ===
    _detect_divergence(result, spx_prices)

    # === Composite Score ===
    result.breadth_score = _compute_breadth_score(result)
    result.breadth_health = _classify_health(result.breadth_score)

    return result


def _analyze_spy_rsp(
    result: BreadthAnalysis,
    spy: list[dict],
    rsp: list[dict],
):
    """Analyze SPY/RSP ratio for leadership breadth."""
    # Align by using min length
    min_len = min(len(spy), len(rsp))
    spy_closes = np.array([p["close"] for p in spy[-min_len:]])
    rsp_closes = np.array([p["close"] for p in rsp[-min_len:]])

    ratio = spy_closes / rsp_closes

    result.spy_rsp_ratio = float(ratio[-1])

    if len(ratio) >= 20:
        result.spy_rsp_ratio_change_20d = float(ratio[-1] - ratio[-20])
    if len(ratio) >= 60:
        result.spy_rsp_ratio_change_60d = float(ratio[-1] - ratio[-60])

    # Narrow leadership: SPY outperforming RSP over 20 and 60 days
    if result.spy_rsp_ratio_change_20d is not None and result.spy_rsp_ratio_change_60d is not None:
        result.narrow_leadership = (
            result.spy_rsp_ratio_change_20d > 0.005 and
            result.spy_rsp_ratio_change_60d > 0.01
        )


def _analyze_sectors(
    result: BreadthAnalysis,
    spy: list[dict],
    sector_prices: dict[str, list[dict]],
):
    """Analyze sector participation, relative strength, and rotation."""
    spy_closes = np.array([p["close"] for p in spy])

    above_200 = 0
    above_50 = 0
    sector_returns_20d = {}
    sector_returns_60d = {}
    sector_count = 0

    for symbol, prices in sector_prices.items():
        if len(prices) < 201:
            continue

        closes = np.array([p["close"] for p in prices])
        sector_count += 1

        # Moving average checks
        ma_200 = np.mean(closes[-200:])
        ma_50 = np.mean(closes[-50:])
        current = closes[-1]

        if current > ma_200:
            above_200 += 1
        if current > ma_50:
            above_50 += 1

        # Relative strength vs SPY
        # Use returns to avoid level bias
        spy_window = spy_closes[-min(len(closes), len(spy_closes)):]
        sector_window = closes[-min(len(closes), len(spy_closes)):]

        if len(sector_window) >= 20 and len(spy_window) >= 20:
            sector_ret_20 = (sector_window[-1] / sector_window[-20] - 1) * 100
            spy_ret_20 = (spy_window[-1] / spy_window[-20] - 1) * 100
            sector_returns_20d[symbol] = sector_ret_20 - spy_ret_20

        if len(sector_window) >= 60 and len(spy_window) >= 60:
            sector_ret_60 = (sector_window[-1] / sector_window[-60] - 1) * 100
            spy_ret_60 = (spy_window[-1] / spy_window[-60] - 1) * 100
            sector_returns_60d[symbol] = sector_ret_60 - spy_ret_60

        # Absolute returns for interpretation data notes
        abs_returns = {}
        if len(closes) >= 6:
            abs_returns["5d"] = round(float((closes[-1] / closes[-6] - 1) * 100), 2)
        if len(closes) >= 21:
            abs_returns["20d"] = round(float((closes[-1] / closes[-21] - 1) * 100), 2)
        if len(closes) >= 61:
            abs_returns["60d"] = round(float((closes[-1] / closes[-61] - 1) * 100), 2)
        if abs_returns:
            result.sector_absolute_returns[symbol] = abs_returns

    if sector_count > 0:
        result.sectors_above_200ma = above_200
        result.sectors_above_50ma = above_50
        result.pct_sectors_above_200ma = (above_200 / sector_count) * 100
        result.pct_sectors_above_50ma = (above_50 / sector_count) * 100

    # Rank sectors by 20d relative strength
    if sector_returns_20d:
        sorted_sectors = sorted(sector_returns_20d.items(), key=lambda x: x[1], reverse=True)
        result.leading_sectors = [s[0] for s in sorted_sectors[:3]]
        result.lagging_sectors = [s[0] for s in sorted_sectors[-3:]]

        result.sector_rankings = {
            sym: {
                "rank": i + 1,
                "rs_20d": round(ret, 2),
                "rs_60d": round(sector_returns_60d.get(sym, 0), 2),
            }
            for i, (sym, ret) in enumerate(sorted_sectors)
        }

    # Cyclical vs Defensive relative strength
    if sector_returns_20d:
        cyc_rets = [v for k, v in sector_returns_20d.items() if k in CYCLICAL_SECTORS]
        def_rets = [v for k, v in sector_returns_20d.items() if k in DEFENSIVE_SECTORS]

        if cyc_rets and def_rets:
            result.cyclical_vs_defensive = float(np.mean(cyc_rets) - np.mean(def_rets))

            if result.cyclical_vs_defensive > 1.5:
                result.rotation_signal = "risk_on_rotation"
            elif result.cyclical_vs_defensive < -1.5:
                result.rotation_signal = "risk_off_rotation"
            else:
                result.rotation_signal = "neutral"

    # Sector dispersion (standard deviation of 20d returns across sectors)
    if sector_returns_20d and len(sector_returns_20d) >= 5:
        rets = list(sector_returns_20d.values())
        result.sector_dispersion_20d = float(np.std(rets))

        if result.sector_dispersion_20d < 2.0:
            result.dispersion_regime = "low"  # Macro-driven
        elif result.sector_dispersion_20d < 5.0:
            result.dispersion_regime = "normal"
        else:
            result.dispersion_regime = "high"  # Stock-picker's market


def _detect_divergence(result: BreadthAnalysis, spx_prices: list[dict]):
    """Detect breadth divergence from price action."""
    closes = [p["close"] for p in spx_prices]

    if len(closes) < 20:
        return

    recent_high = max(closes[-20:])
    current = closes[-1]
    near_high = current >= recent_high * 0.98

    # Multiple divergence signals
    divergences = []

    if near_high and result.pct_sectors_above_200ma < 60:
        divergences.append(
            f"SPX near 20-day highs but only {result.pct_sectors_above_200ma:.0f}% "
            f"of sectors above 200-day MA"
        )

    if near_high and result.narrow_leadership:
        divergences.append(
            "Cap-weighted (SPY) outperforming equal-weight (RSP) — "
            "narrow mega-cap leadership"
        )

    if near_high and result.rotation_signal == "risk_off_rotation":
        divergences.append(
            "Defensive sectors leading despite market near highs — "
            "smart money rotating out of risk"
        )

    if divergences:
        result.breadth_divergence = True
        result.divergence_description = ". ".join(divergences) + "."


def _compute_breadth_score(result: BreadthAnalysis) -> float:
    """Compute 0-1 breadth health score from multiple signals."""
    scores = []
    weights = []

    # Sector participation (weight: 0.35)
    if result.pct_sectors_above_200ma > 0:
        scores.append(result.pct_sectors_above_200ma / 100)
        weights.append(0.35)

    # SPY/RSP ratio trend (weight: 0.25)
    # Narrow leadership = lower score
    if result.spy_rsp_ratio_change_20d is not None:
        # Negative change (RSP outperforming) = broad breadth = higher score
        rsp_score = 0.5 - (result.spy_rsp_ratio_change_20d * 10)  # Scale appropriately
        rsp_score = max(0.0, min(1.0, rsp_score))
        scores.append(rsp_score)
        weights.append(0.25)

    # Sector rotation (weight: 0.20)
    if result.rotation_signal == "risk_on_rotation":
        scores.append(0.75)
    elif result.rotation_signal == "risk_off_rotation":
        scores.append(0.25)
    else:
        scores.append(0.5)
    weights.append(0.20)

    # Divergence penalty (weight: 0.20)
    if result.breadth_divergence:
        scores.append(0.15)
    else:
        scores.append(0.65)
    weights.append(0.20)

    if not scores:
        return 0.5

    total_weight = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_weight


def _classify_health(score: float) -> str:
    if score >= 0.7:
        return "strong"
    elif score >= 0.55:
        return "healthy"
    elif score >= 0.4:
        return "weakening"
    else:
        return "poor"
