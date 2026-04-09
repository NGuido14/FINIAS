"""
Volume Confirmation Module (Dimension 3: Volume)

Answers: "Is volume confirming or contradicting the price move?"

This is the most important filter our backtest identified as missing.
A downtrend on declining volume = exhaustion (sellers drying up, bounce likely).
A downtrend on expanding volume = distribution (institutions selling, more pain).

Key metrics:
  - OBV (On Balance Volume): cumulative volume confirming price direction
  - Relative Volume: today's volume vs 20-day average (identifies unusual activity)
  - Volume Trend: is volume expanding or contracting during the current move
  - MFI (Money Flow Index): volume-weighted RSI
  - Volume Confirmation Score: does volume agree with the price trend?

All computation is pure Python + pandas-ta. No API calls.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import pandas_ta as ta


@dataclass
class VolumeAnalysis:
    """Complete volume analysis for a single symbol."""

    symbol: str = ""
    trend_regime_used: str = "unknown"

    # Composite
    volume_confirmation_score: float = 0.0  # -1 (contradicts) to +1 (confirms)

    # OBV
    obv_trend: str = "neutral"  # rising, falling, neutral (20-bar slope)
    obv_divergence: str = "none"  # bullish (price down, OBV up), bearish (price up, OBV down), none
    obv_slope_20d: Optional[float] = None  # Normalized OBV slope

    # Relative Volume
    relative_volume: Optional[float] = None  # Current vol / 20d avg (1.0 = normal)
    relative_volume_5d_avg: Optional[float] = None  # 5-day average relative volume
    volume_zone: str = "normal"  # high (>1.5x), low (<0.5x), normal

    # Volume Trend During Current Regime
    volume_trend_in_regime: str = "neutral"  # expanding, contracting, neutral
    volume_trend_slope: Optional[float] = None

    # MFI (Money Flow Index)
    mfi_14: Optional[float] = None
    mfi_zone: str = "neutral"  # overbought (>80), oversold (<20), neutral

    # Accumulation/Distribution
    ad_trend: str = "neutral"  # accumulation (rising), distribution (falling), neutral

    def to_dict(self) -> dict:
        return {
            "volume_confirmation_score": round(self.volume_confirmation_score, 4),
            "trend_regime_used": self.trend_regime_used,
            "obv": {
                "trend": self.obv_trend,
                "divergence": self.obv_divergence,
                "slope_20d": round(self.obv_slope_20d, 6) if self.obv_slope_20d is not None else None,
            },
            "relative_volume": {
                "current": round(self.relative_volume, 2) if self.relative_volume is not None else None,
                "avg_5d": round(self.relative_volume_5d_avg, 2) if self.relative_volume_5d_avg is not None else None,
                "zone": self.volume_zone,
            },
            "regime_volume_trend": {
                "direction": self.volume_trend_in_regime,
                "slope": round(self.volume_trend_slope, 6) if self.volume_trend_slope is not None else None,
            },
            "mfi": {
                "value": round(self.mfi_14, 2) if self.mfi_14 is not None else None,
                "zone": self.mfi_zone,
            },
            "ad_trend": self.ad_trend,
        }


def analyze_volume(
    df: pd.DataFrame,
    symbol: str = "",
    trend_regime: str = "unknown",
) -> VolumeAnalysis:
    """
    Compute volume-based confirmation/divergence signals.

    Args:
        df: OHLCV DataFrame, sorted chronologically.
        symbol: Ticker symbol.
        trend_regime: From trend.py — determines what "confirmation" means.

    Returns:
        VolumeAnalysis with all volume signals.
    """
    result = VolumeAnalysis(symbol=symbol, trend_regime_used=trend_regime)

    if df is None or len(df) < 30:
        return result

    close = df["close"].values
    volume = df["volume"].values

    # === OBV ===
    _compute_obv(df, result, close)

    # === Relative Volume ===
    _compute_relative_volume(df, result, volume)

    # === Volume Trend During Regime ===
    _compute_volume_trend(df, result, volume, trend_regime)

    # === MFI ===
    _compute_mfi(df, result)

    # === Accumulation/Distribution ===
    _compute_ad(df, result)

    # === Confirmation Score ===
    _compute_confirmation_score(result, trend_regime)

    return result


def _compute_obv(df: pd.DataFrame, result: VolumeAnalysis, close: np.ndarray):
    """Compute OBV and its trend/divergence."""
    obv_series = ta.obv(df["close"], df["volume"])
    if obv_series is None or obv_series.empty:
        return

    obv = obv_series.values

    # OBV 20-day slope (normalized by mean volume)
    if len(obv) >= 20:
        obv_20 = obv[-20:]
        x = np.arange(20)
        slope = np.polyfit(x, obv_20, 1)[0]
        mean_vol = np.mean(df["volume"].values[-20:])
        result.obv_slope_20d = slope / mean_vol if mean_vol > 0 else 0

        if result.obv_slope_20d > 0.01:
            result.obv_trend = "rising"
        elif result.obv_slope_20d < -0.01:
            result.obv_trend = "falling"
        else:
            result.obv_trend = "neutral"

    # OBV vs Price divergence (compare 20-bar slopes)
    if len(close) >= 20:
        price_20 = close[-20:]
        price_slope = np.polyfit(np.arange(20), price_20, 1)[0]
        price_slope_norm = price_slope / np.mean(price_20) if np.mean(price_20) > 0 else 0

        if result.obv_slope_20d is not None:
            # Bullish divergence: price falling, OBV rising (accumulation on dips)
            if price_slope_norm < -0.001 and result.obv_slope_20d > 0.005:
                result.obv_divergence = "bullish"
            # Bearish divergence: price rising, OBV falling (distribution on rallies)
            elif price_slope_norm > 0.001 and result.obv_slope_20d < -0.005:
                result.obv_divergence = "bearish"


def _compute_relative_volume(df: pd.DataFrame, result: VolumeAnalysis, volume: np.ndarray):
    """Compute relative volume vs 20-day average."""
    if len(volume) < 21:
        return

    avg_20d = np.mean(volume[-21:-1])  # 20-day average excluding today
    if avg_20d > 0:
        result.relative_volume = float(volume[-1]) / avg_20d

        # 5-day average relative volume
        if len(volume) >= 25:
            recent_5 = volume[-5:]
            result.relative_volume_5d_avg = float(np.mean(recent_5)) / avg_20d

    if result.relative_volume is not None:
        if result.relative_volume > 1.5:
            result.volume_zone = "high"
        elif result.relative_volume < 0.5:
            result.volume_zone = "low"


def _compute_volume_trend(
    df: pd.DataFrame, result: VolumeAnalysis, volume: np.ndarray, trend_regime: str,
):
    """Compute whether volume is expanding or contracting during the current move."""
    # Use last 20 bars to measure volume trend
    if len(volume) < 20:
        return

    vol_20 = volume[-20:].astype(float)
    x = np.arange(20)
    slope = np.polyfit(x, vol_20, 1)[0]
    mean_vol = np.mean(vol_20)
    normalized_slope = slope / mean_vol if mean_vol > 0 else 0

    result.volume_trend_slope = normalized_slope

    if normalized_slope > 0.02:
        result.volume_trend_in_regime = "expanding"
    elif normalized_slope < -0.02:
        result.volume_trend_in_regime = "contracting"
    else:
        result.volume_trend_in_regime = "neutral"


def _compute_mfi(df: pd.DataFrame, result: VolumeAnalysis):
    """Compute Money Flow Index (volume-weighted RSI)."""
    if len(df) < 20:
        return

    mfi = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
    if mfi is None or mfi.empty:
        return

    result.mfi_14 = float(mfi.iloc[-1])

    if result.mfi_14 > 80:
        result.mfi_zone = "overbought"
    elif result.mfi_14 < 20:
        result.mfi_zone = "oversold"


def _compute_ad(df: pd.DataFrame, result: VolumeAnalysis):
    """Compute Accumulation/Distribution line trend."""
    ad = ta.ad(df["high"], df["low"], df["close"], df["volume"])
    if ad is None or ad.empty or len(ad) < 20:
        return

    # 20-bar slope of A/D line
    ad_20 = ad.values[-20:]
    x = np.arange(20)
    slope = np.polyfit(x, ad_20, 1)[0]
    mean_ad = np.mean(np.abs(ad_20))
    normalized = slope / mean_ad if mean_ad > 0 else 0

    if normalized > 0.01:
        result.ad_trend = "accumulation"
    elif normalized < -0.01:
        result.ad_trend = "distribution"


def _compute_confirmation_score(result: VolumeAnalysis, trend_regime: str):
    """
    Compute whether volume confirms or contradicts the current trend.

    In an uptrend, confirmation = rising OBV, expanding volume, accumulation
    In a downtrend, confirmation would mean the downtrend is strong (volume expanding).
    BUT: for mean-reversion, we want the OPPOSITE — declining volume in downtrend = exhaustion.
    So the score is contextual:

    Positive score = volume setup supports a BULLISH outcome
      - In downtrend: declining volume + OBV bullish divergence + accumulation = exhaustion
      - In uptrend: rising OBV + expanding volume + accumulation = trend confirmation
    Negative score = volume setup supports a BEARISH outcome
      - In uptrend: declining OBV + distribution + volume falling = distribution
      - In downtrend: expanding volume + no divergence + distribution = continuation
    """
    score = 0.0

    # OBV divergence (strongest signal — institutional behavior)
    if result.obv_divergence == "bullish":
        score += 0.4  # Accumulation happening despite price decline
    elif result.obv_divergence == "bearish":
        score -= 0.4  # Distribution happening despite price rise

    # OBV trend
    if result.obv_trend == "rising":
        score += 0.15
    elif result.obv_trend == "falling":
        score -= 0.15

    # A/D trend
    if result.ad_trend == "accumulation":
        score += 0.15
    elif result.ad_trend == "distribution":
        score -= 0.15

    # Volume trend in context
    if trend_regime in ("downtrend", "strong_downtrend"):
        # In downtrends, contracting volume = exhaustion (bullish)
        if result.volume_trend_in_regime == "contracting":
            score += 0.2
        elif result.volume_trend_in_regime == "expanding":
            score -= 0.2
    elif trend_regime in ("uptrend", "strong_uptrend"):
        # In uptrends, expanding volume = healthy trend (bullish)
        if result.volume_trend_in_regime == "expanding":
            score += 0.2
        elif result.volume_trend_in_regime == "contracting":
            score -= 0.2

    # MFI extreme zones
    if result.mfi_zone == "oversold":
        score += 0.1
    elif result.mfi_zone == "overbought":
        score -= 0.1

    result.volume_confirmation_score = max(-1.0, min(1.0, score))
