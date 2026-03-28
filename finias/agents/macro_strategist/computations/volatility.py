"""
Volatility Regime Analysis Module

Analyzes VIX and related volatility metrics to classify the current
volatility regime. This tells us about market fear, complacency,
and the pricing of tail risk.

Key metrics:
  - VIX absolute level (fear gauge)
  - VIX term structure: contango (normal) vs backwardation (stressed)
  - Realized vs Implied vol spread (are options overpriced or underpriced?)
  - VIX percentile rank (context: is 20 high or low vs history?)
  - Rate of VIX change (spike = panic, grind = worry)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class VolatilityAnalysis:
    """Complete volatility regime assessment."""
    # Current levels
    vix_current: Optional[float]
    vix_percentile_1y: Optional[float]   # Where VIX sits vs last year (0-100)

    # VIX dynamics
    vix_change_1d: Optional[float]       # 1-day change (% points)
    vix_change_5d: Optional[float]       # 5-day change
    vix_change_20d: Optional[float]      # 20-day change
    vix_sma_20: Optional[float]          # 20-day simple moving average
    vix_is_elevated: bool                # Above 20
    vix_is_spike: bool                   # Rose >30% in 5 days

    # Realized vol
    realized_vol_20d: Optional[float]    # 20-day realized vol of SPX (annualized)
    realized_vol_60d: Optional[float]    # 60-day realized vol
    iv_rv_spread: Optional[float]        # VIX - realized vol (positive = fear premium)

    # Regime
    vol_regime: str                      # low, normal, elevated, extreme

    # Risk metrics
    vol_risk_score: float                # 0 to 1 — higher = more volatile environment

    # Correlation regime
    sector_correlation: Optional[float] = None      # Average pairwise sector correlation
    correlation_regime: str = "unknown"              # low, normal, elevated, extreme

    # Variance risk premium
    variance_risk_premium: Optional[float] = None   # IV² - RV²
    vrp_regime: str = "unknown"                      # normal, compressed, negative

    # Enhanced score
    vol_score: float = 0.0                           # -1 (bearish high vol) to +1 (bullish low vol)

    def to_dict(self) -> dict:
        return {
            "vix": {
                "current": self.vix_current,
                "percentile_1y": self.vix_percentile_1y,
                "change_1d": self.vix_change_1d,
                "change_5d": self.vix_change_5d,
                "change_20d": self.vix_change_20d,
                "sma_20": self.vix_sma_20,
                "is_elevated": self.vix_is_elevated,
                "is_spike": self.vix_is_spike,
            },
            "realized_vol": {
                "20d": self.realized_vol_20d,
                "60d": self.realized_vol_60d,
                "iv_rv_spread": self.iv_rv_spread,
            },
            "vol_regime": self.vol_regime,
            "vol_risk_score": self.vol_risk_score,
            "correlation": {
                "sector_correlation": self.sector_correlation,
                "correlation_regime": self.correlation_regime,
            },
            "vrp": {
                "variance_risk_premium": self.variance_risk_premium,
                "vrp_regime": self.vrp_regime,
            },
            "vol_score": self.vol_score,
        }


def analyze_volatility(
    vix_series: list[dict],
    spx_prices: list[dict],
) -> VolatilityAnalysis:
    """
    Perform complete volatility analysis.

    Args:
        vix_series: FRED VIX data [{"date": str, "value": float}], ascending
        spx_prices: SPX daily closes [{"date": str, "close": float}], ascending
    """
    # VIX current and dynamics
    vix_current = vix_series[-1]["value"] if vix_series else None
    vix_values = np.array([v["value"] for v in vix_series]) if vix_series else np.array([])

    vix_percentile = _percentile_rank(vix_values, 252) if len(vix_values) >= 252 else None
    vix_1d = _change(vix_values, 1)
    vix_5d = _change(vix_values, 5)
    vix_20d = _change(vix_values, 20)
    vix_sma20 = float(np.mean(vix_values[-20:])) if len(vix_values) >= 20 else None

    is_elevated = vix_current is not None and vix_current > 20
    is_spike = vix_5d is not None and vix_5d > 6  # >6 pt rise in 5 days

    # Realized volatility from SPX returns
    rv_20, rv_60 = _compute_realized_vol(spx_prices)

    # IV-RV spread
    iv_rv = (vix_current - rv_20) if (vix_current is not None and rv_20 is not None) else None

    # Classify regime
    regime = _classify_vol_regime(vix_current, vix_percentile, is_spike)

    # Risk score
    risk_score = _compute_vol_risk_score(vix_current, vix_percentile, is_spike, iv_rv)

    # Variance Risk Premium: VIX² - RV²
    # Positive = normal (selling vol is profitable, implied > realized)
    # Negative = market underpricing risk (realized exceeding implied)
    # Near zero = compressed premium, crowded vol-selling trade
    vrp = None
    vrp_regime = "unknown"
    if vix_current is not None and rv_20 is not None:
        vrp = (vix_current ** 2 - rv_20 ** 2) / 100  # Scale to readable units
        if vrp > 3.0:
            vrp_regime = "normal"
        elif vrp > 0.5:
            vrp_regime = "compressed"
        elif vrp > -1.0:
            vrp_regime = "flat"
        else:
            vrp_regime = "negative"  # Realized exceeding implied — danger

    return VolatilityAnalysis(
        vix_current=vix_current,
        vix_percentile_1y=vix_percentile,
        vix_change_1d=vix_1d,
        vix_change_5d=vix_5d,
        vix_change_20d=vix_20d,
        vix_sma_20=vix_sma20,
        vix_is_elevated=is_elevated,
        vix_is_spike=is_spike,
        realized_vol_20d=rv_20,
        realized_vol_60d=rv_60,
        iv_rv_spread=iv_rv,
        vol_regime=regime,
        vol_risk_score=risk_score,
        variance_risk_premium=vrp,
        vrp_regime=vrp_regime,
    )


def _percentile_rank(values: np.ndarray, lookback: int) -> Optional[float]:
    """Compute percentile rank of latest value vs trailing window."""
    if len(values) < lookback:
        return None
    window = values[-lookback:]
    current = values[-1]
    return float(np.sum(window <= current) / len(window) * 100)


def _change(values: np.ndarray, periods: int) -> Optional[float]:
    """Compute absolute change over N periods."""
    if len(values) <= periods:
        return None
    return float(values[-1] - values[-(periods + 1)])


def _compute_realized_vol(prices: list[dict]) -> tuple[Optional[float], Optional[float]]:
    """
    Compute annualized realized volatility from daily prices.

    RV = std(log returns) * sqrt(252)
    """
    if len(prices) < 61:
        return None, None

    closes = np.array([p["close"] for p in prices])
    log_returns = np.diff(np.log(closes))

    rv_20 = float(np.std(log_returns[-20:]) * np.sqrt(252) * 100) if len(log_returns) >= 20 else None
    rv_60 = float(np.std(log_returns[-60:]) * np.sqrt(252) * 100) if len(log_returns) >= 60 else None

    return rv_20, rv_60


def _classify_vol_regime(
    vix: Optional[float],
    percentile: Optional[float],
    is_spike: bool
) -> str:
    """Classify current volatility regime."""
    if vix is None:
        return "unknown"

    if vix > 35 or is_spike:
        return "extreme"
    elif vix > 25:
        return "elevated"
    elif vix > 15:
        return "normal"
    else:
        return "low"


def _compute_vol_risk_score(
    vix: Optional[float],
    percentile: Optional[float],
    is_spike: bool,
    iv_rv_spread: Optional[float]
) -> float:
    """
    Compute 0-1 volatility risk score.

    Higher = more risk/stress in the volatility environment.
    """
    if vix is None:
        return 0.5  # Unknown = moderate default

    score = 0.0

    # VIX level contribution
    if vix > 35:
        score += 0.4
    elif vix > 25:
        score += 0.25
    elif vix > 20:
        score += 0.15
    elif vix > 15:
        score += 0.05

    # Percentile contribution
    if percentile is not None:
        score += (percentile / 100) * 0.3

    # Spike contribution
    if is_spike:
        score += 0.2

    # Negative IV-RV spread (realized > implied = market underpricing risk)
    if iv_rv_spread is not None and iv_rv_spread < -3:
        score += 0.1

    return min(1.0, score)


def compute_sector_correlation(
    sector_prices: dict[str, list[dict]],
    window: int = 60,
) -> Optional[float]:
    """
    Compute average pairwise correlation across sector ETFs.

    Low correlation = healthy (stocks moving on fundamentals)
    High correlation = risk-off (everything moving together)

    Args:
        sector_prices: dict mapping sector ETF symbol to price list
        window: lookback window in trading days
    """
    returns = {}
    for symbol, prices in sector_prices.items():
        if len(prices) < window + 1:
            continue
        closes = np.array([p["close"] for p in prices[-(window + 1):]])
        ret = np.diff(np.log(closes))
        returns[symbol] = ret

    if len(returns) < 3:
        return None

    symbols = list(returns.keys())
    correlations = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            corr = np.corrcoef(returns[symbols[i]], returns[symbols[j]])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    if not correlations:
        return None

    return float(np.mean(correlations))


def classify_correlation_regime(avg_corr: Optional[float]) -> str:
    """Classify correlation regime based on average pairwise correlation."""
    if avg_corr is None:
        return "unknown"
    if avg_corr > 0.7:
        return "extreme"
    elif avg_corr > 0.5:
        return "elevated"
    elif avg_corr > 0.3:
        return "normal"
    return "low"
