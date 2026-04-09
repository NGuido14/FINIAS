"""
Regime-Adaptive Momentum Module (Dimension 2: Momentum)

Answers: "Is momentum confirming the trend, or diverging? Are we at an extreme?"

Key intelligence features:
  - RSI with ADAPTIVE thresholds based on trend regime
  - MACD signal line, histogram direction, histogram momentum
  - Stochastic %K/%D for short-term timing
  - Divergence detection: regular bullish/bearish + hidden bullish/bearish
  - Momentum thrust detection

CRITICAL DESIGN: This module DEPENDS on trend regime from trend.py.
In a strong uptrend, RSI 70 is NOT overbought — the threshold shifts to 80.
In a downtrend, RSI 30 is NOT oversold — the threshold shifts to 20.
Fixed thresholds are retail noise. Adaptive thresholds are institutional quality.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import finias.agents.technical_analyst.computations.indicators as ta


# Regime-adaptive RSI thresholds
RSI_THRESHOLDS = {
    "strong_uptrend":  {"overbought": 80, "oversold": 40},
    "uptrend":         {"overbought": 75, "oversold": 35},
    "consolidation":   {"overbought": 70, "oversold": 30},
    "downtrend":       {"overbought": 65, "oversold": 25},
    "strong_downtrend": {"overbought": 60, "oversold": 20},
    "unknown":         {"overbought": 70, "oversold": 30},
}


@dataclass
class MomentumAnalysis:
    """Complete momentum analysis for a single symbol."""

    symbol: str = ""
    trend_regime_used: str = "unknown"  # Which regime drove threshold adaptation

    # Composite
    momentum_score: float = 0.0  # -1 to +1

    # RSI
    rsi_14: Optional[float] = None
    rsi_zone: str = "neutral"  # overbought, oversold, neutral (regime-adaptive)
    rsi_overbought_threshold: int = 70
    rsi_oversold_threshold: int = 30

    # MACD
    macd_value: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    macd_direction: str = "neutral"  # bullish (hist > 0), bearish (hist < 0)
    macd_momentum: str = "neutral"  # accelerating, decelerating, neutral
    macd_cross: str = "none"  # bullish_cross, bearish_cross, none

    # Stochastic
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    stoch_zone: str = "neutral"  # overbought (>80), oversold (<20), neutral

    # Rate of Change
    roc_5d: Optional[float] = None
    roc_20d: Optional[float] = None
    roc_60d: Optional[float] = None

    # Divergence
    divergence_type: str = "none"  # bullish, bearish, hidden_bullish, hidden_bearish, none
    divergence_description: str = ""

    # Momentum Thrust
    thrust_detected: bool = False
    thrust_description: str = ""

    def to_dict(self) -> dict:
        return {
            "momentum_score": round(self.momentum_score, 4),
            "trend_regime_used": self.trend_regime_used,
            "rsi": {
                "value": round(self.rsi_14, 2) if self.rsi_14 is not None else None,
                "zone": self.rsi_zone,
                "overbought_threshold": self.rsi_overbought_threshold,
                "oversold_threshold": self.rsi_oversold_threshold,
            },
            "macd": {
                "value": round(self.macd_value, 4) if self.macd_value is not None else None,
                "signal": round(self.macd_signal, 4) if self.macd_signal is not None else None,
                "histogram": round(self.macd_histogram, 4) if self.macd_histogram is not None else None,
                "direction": self.macd_direction,
                "momentum": self.macd_momentum,
                "cross": self.macd_cross,
            },
            "stochastic": {
                "k": round(self.stoch_k, 2) if self.stoch_k is not None else None,
                "d": round(self.stoch_d, 2) if self.stoch_d is not None else None,
                "zone": self.stoch_zone,
            },
            "roc": {
                "5d": round(self.roc_5d, 4) if self.roc_5d is not None else None,
                "20d": round(self.roc_20d, 4) if self.roc_20d is not None else None,
                "60d": round(self.roc_60d, 4) if self.roc_60d is not None else None,
            },
            "divergence": {
                "type": self.divergence_type,
                "description": self.divergence_description,
            },
            "thrust": {
                "detected": self.thrust_detected,
                "description": self.thrust_description,
            },
        }


def analyze_momentum(
    df: pd.DataFrame,
    symbol: str = "",
    trend_regime: str = "unknown",
) -> MomentumAnalysis:
    """
    Compute regime-adaptive momentum analysis.

    Args:
        df: OHLCV DataFrame, sorted chronologically.
        symbol: Ticker symbol.
        trend_regime: From trend.py — drives adaptive thresholds.

    Returns:
        MomentumAnalysis with all momentum signals.
    """
    result = MomentumAnalysis(symbol=symbol, trend_regime_used=trend_regime)

    if df is None or len(df) < 30:
        return result

    # Set adaptive thresholds
    thresholds = RSI_THRESHOLDS.get(trend_regime, RSI_THRESHOLDS["unknown"])
    result.rsi_overbought_threshold = thresholds["overbought"]
    result.rsi_oversold_threshold = thresholds["oversold"]

    # === RSI ===
    _compute_rsi(df, result)

    # === MACD ===
    _compute_macd(df, result)

    # === Stochastic ===
    _compute_stochastic(df, result)

    # === Rate of Change ===
    _compute_roc(df, result)

    # === Divergence Detection ===
    if len(df) >= 60:
        _detect_divergence(df, result, trend_regime)

    # === Momentum Thrust ===
    if result.rsi_14 is not None and len(df) >= 20:
        _detect_thrust(df, result)

    # === Composite Score ===
    _compute_momentum_score(result, trend_regime)

    return result


def _compute_rsi(df: pd.DataFrame, result: MomentumAnalysis):
    """Compute RSI with regime-adaptive zone classification."""
    rsi = ta.rsi(df["close"], length=14)
    if rsi is None or rsi.empty:
        return

    result.rsi_14 = float(rsi.iloc[-1])

    if result.rsi_14 >= result.rsi_overbought_threshold:
        result.rsi_zone = "overbought"
    elif result.rsi_14 <= result.rsi_oversold_threshold:
        result.rsi_zone = "oversold"
    else:
        result.rsi_zone = "neutral"


def _compute_macd(df: pd.DataFrame, result: MomentumAnalysis):
    """Compute MACD with direction and momentum analysis."""
    macd_df = ta.macd(df["close"])
    if macd_df is None or macd_df.empty:
        return

    # pandas-ta MACD columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    macd_col = [c for c in macd_df.columns if c.startswith("MACD_")]
    hist_col = [c for c in macd_df.columns if c.startswith("MACDh_")]
    signal_col = [c for c in macd_df.columns if c.startswith("MACDs_")]

    if macd_col:
        result.macd_value = float(macd_df[macd_col[0]].iloc[-1])
    if signal_col:
        result.macd_signal = float(macd_df[signal_col[0]].iloc[-1])
    if hist_col:
        result.macd_histogram = float(macd_df[hist_col[0]].iloc[-1])

        hist = macd_df[hist_col[0]]

        # Direction
        if result.macd_histogram > 0:
            result.macd_direction = "bullish"
        elif result.macd_histogram < 0:
            result.macd_direction = "bearish"

        # Momentum (is histogram accelerating or decelerating?)
        if len(hist.dropna()) >= 3:
            prev = float(hist.iloc[-2])
            curr = float(hist.iloc[-1])
            if abs(curr) > abs(prev) and np.sign(curr) == np.sign(prev):
                result.macd_momentum = "accelerating"
            elif abs(curr) < abs(prev) and np.sign(curr) == np.sign(prev):
                result.macd_momentum = "decelerating"

    # Cross detection
    if macd_col and signal_col and len(macd_df) >= 2:
        macd_now = float(macd_df[macd_col[0]].iloc[-1])
        macd_prev = float(macd_df[macd_col[0]].iloc[-2])
        sig_now = float(macd_df[signal_col[0]].iloc[-1])
        sig_prev = float(macd_df[signal_col[0]].iloc[-2])

        if macd_now > sig_now and macd_prev <= sig_prev:
            result.macd_cross = "bullish_cross"
        elif macd_now < sig_now and macd_prev >= sig_prev:
            result.macd_cross = "bearish_cross"


def _compute_stochastic(df: pd.DataFrame, result: MomentumAnalysis):
    """Compute Stochastic oscillator."""
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    if stoch is None or stoch.empty:
        return

    k_col = [c for c in stoch.columns if c.startswith("STOCHk")]
    d_col = [c for c in stoch.columns if c.startswith("STOCHd")]

    if k_col:
        result.stoch_k = float(stoch[k_col[0]].iloc[-1])
    if d_col:
        result.stoch_d = float(stoch[d_col[0]].iloc[-1])

    if result.stoch_k is not None:
        if result.stoch_k > 80:
            result.stoch_zone = "overbought"
        elif result.stoch_k < 20:
            result.stoch_zone = "oversold"


def _compute_roc(df: pd.DataFrame, result: MomentumAnalysis):
    """Compute Rate of Change at multiple timeframes."""
    close = df["close"]
    if len(close) >= 6:
        result.roc_5d = float(close.iloc[-1] / close.iloc[-6] - 1)
    if len(close) >= 21:
        result.roc_20d = float(close.iloc[-1] / close.iloc[-21] - 1)
    if len(close) >= 61:
        result.roc_60d = float(close.iloc[-1] / close.iloc[-61] - 1)


def _detect_divergence(df: pd.DataFrame, result: MomentumAnalysis, trend_regime: str):
    """
    Detect RSI divergences with price.

    Regular bullish: price makes lower low, RSI makes higher low → reversal signal
    Regular bearish: price makes higher high, RSI makes lower high → reversal signal
    Hidden bullish: price makes higher low, RSI makes lower low → continuation in uptrend
    Hidden bearish: price makes lower high, RSI makes higher high → continuation in downtrend
    """
    rsi_series = ta.rsi(df["close"], length=14)
    if rsi_series is None or len(rsi_series.dropna()) < 30:
        return

    close = df["close"].values[-60:]
    rsi = rsi_series.values[-60:]

    # Remove NaN
    valid = ~np.isnan(rsi)
    if np.sum(valid) < 30:
        return

    close = close[valid]
    rsi = rsi[valid]

    # Split into two halves and compare swing lows/highs
    mid = len(close) // 2
    first_half_close = close[:mid]
    second_half_close = close[mid:]
    first_half_rsi = rsi[:mid]
    second_half_rsi = rsi[mid:]

    # Price lows and highs
    price_low_1 = float(np.min(first_half_close))
    price_low_2 = float(np.min(second_half_close))
    price_high_1 = float(np.max(first_half_close))
    price_high_2 = float(np.max(second_half_close))

    # RSI at those price extremes
    rsi_at_low_1 = float(first_half_rsi[np.argmin(first_half_close)])
    rsi_at_low_2 = float(second_half_rsi[np.argmin(second_half_close)])
    rsi_at_high_1 = float(first_half_rsi[np.argmax(first_half_close)])
    rsi_at_high_2 = float(second_half_rsi[np.argmax(second_half_close)])

    # Regular bullish: lower price low, higher RSI low
    if price_low_2 < price_low_1 and rsi_at_low_2 > rsi_at_low_1 + 3:
        result.divergence_type = "bullish"
        result.divergence_description = (
            f"Price made lower low ({price_low_2:.2f} < {price_low_1:.2f}) "
            f"but RSI made higher low ({rsi_at_low_2:.1f} > {rsi_at_low_1:.1f}) — "
            f"bullish divergence, potential reversal"
        )
        return

    # Regular bearish: higher price high, lower RSI high
    if price_high_2 > price_high_1 and rsi_at_high_2 < rsi_at_high_1 - 3:
        result.divergence_type = "bearish"
        result.divergence_description = (
            f"Price made higher high ({price_high_2:.2f} > {price_high_1:.2f}) "
            f"but RSI made lower high ({rsi_at_high_2:.1f} < {rsi_at_high_1:.1f}) — "
            f"bearish divergence, potential reversal"
        )
        return

    # Hidden bullish (only in uptrends): higher price low, lower RSI low
    if trend_regime in ("uptrend", "strong_uptrend"):
        if price_low_2 > price_low_1 and rsi_at_low_2 < rsi_at_low_1 - 3:
            result.divergence_type = "hidden_bullish"
            result.divergence_description = (
                f"Price made higher low ({price_low_2:.2f} > {price_low_1:.2f}) "
                f"but RSI made lower low ({rsi_at_low_2:.1f} < {rsi_at_low_1:.1f}) — "
                f"hidden bullish divergence, trend continuation"
            )
            return

    # Hidden bearish (only in downtrends): lower price high, higher RSI high
    if trend_regime in ("downtrend", "strong_downtrend"):
        if price_high_2 < price_high_1 and rsi_at_high_2 > rsi_at_high_1 + 3:
            result.divergence_type = "hidden_bearish"
            result.divergence_description = (
                f"Price made lower high ({price_high_2:.2f} < {price_high_1:.2f}) "
                f"but RSI made higher high ({rsi_at_high_2:.1f} > {rsi_at_high_1:.1f}) — "
                f"hidden bearish divergence, trend continuation"
            )


def _detect_thrust(df: pd.DataFrame, result: MomentumAnalysis):
    """
    Detect momentum thrust: RSI moves from below 30 to above 70 within N bars.

    A thrust is one of the most powerful momentum signals. It indicates
    a dramatic shift from oversold to overbought territory, typically
    occurring at the start of a new trend leg.
    """
    rsi = ta.rsi(df["close"], length=14)
    if rsi is None:
        return

    rsi_values = rsi.dropna().values
    if len(rsi_values) < 15:
        return

    # Check last 15 bars for a thrust
    window = rsi_values[-15:]
    min_rsi = float(np.min(window))
    max_rsi = float(np.max(window))

    # Bullish thrust: went from <30 to >70 within 15 bars
    if min_rsi < 30 and max_rsi > 70:
        min_idx = np.argmin(window)
        max_idx = np.argmax(window)
        if max_idx > min_idx:  # Low came before high
            result.thrust_detected = True
            result.thrust_description = (
                f"Bullish momentum thrust: RSI moved from {min_rsi:.1f} to {max_rsi:.1f} "
                f"in {max_idx - min_idx} bars — extremely bullish"
            )

    # Bearish thrust: went from >70 to <30 within 15 bars
    elif max_rsi > 70 and min_rsi < 30:
        min_idx = np.argmin(window)
        max_idx = np.argmax(window)
        if min_idx > max_idx:  # High came before low
            result.thrust_detected = True
            result.thrust_description = (
                f"Bearish momentum thrust: RSI moved from {max_rsi:.1f} to {min_rsi:.1f} "
                f"in {min_idx - max_idx} bars — extremely bearish"
            )


def _compute_momentum_score(result: MomentumAnalysis, trend_regime: str):
    """
    Compute composite momentum score from all indicators.

    In trending regimes, weight trend-following indicators higher.
    In consolidation, weight mean-reversion indicators higher.
    """
    score = 0.0
    weights_total = 0.0

    # RSI contribution
    if result.rsi_14 is not None:
        midpoint = (result.rsi_overbought_threshold + result.rsi_oversold_threshold) / 2
        rsi_range = (result.rsi_overbought_threshold - result.rsi_oversold_threshold) / 2
        rsi_normalized = (result.rsi_14 - midpoint) / rsi_range if rsi_range > 0 else 0
        rsi_normalized = max(-1.0, min(1.0, rsi_normalized))

        if trend_regime in ("consolidation", "unknown"):
            # In consolidation, extreme RSI is a mean-reversion signal (inverse)
            score += -rsi_normalized * 0.25
        else:
            # In trends, RSI direction confirms trend
            score += rsi_normalized * 0.20
        weights_total += 0.25

    # MACD contribution
    if result.macd_histogram is not None:
        macd_signal_val = 0.0
        if result.macd_direction == "bullish":
            macd_signal_val = 0.5
            if result.macd_momentum == "accelerating":
                macd_signal_val = 1.0
            elif result.macd_momentum == "decelerating":
                macd_signal_val = 0.25
        elif result.macd_direction == "bearish":
            macd_signal_val = -0.5
            if result.macd_momentum == "accelerating":
                macd_signal_val = -1.0
            elif result.macd_momentum == "decelerating":
                macd_signal_val = -0.25

        if result.macd_cross == "bullish_cross":
            macd_signal_val = max(macd_signal_val, 0.75)
        elif result.macd_cross == "bearish_cross":
            macd_signal_val = min(macd_signal_val, -0.75)

        score += macd_signal_val * 0.35
        weights_total += 0.35

    # Stochastic contribution
    if result.stoch_k is not None:
        stoch_normalized = (result.stoch_k - 50) / 50  # -1 to +1
        stoch_normalized = max(-1.0, min(1.0, stoch_normalized))
        score += stoch_normalized * 0.15
        weights_total += 0.15

    # Divergence contribution (high weight — divergences are powerful)
    div_map = {
        "bullish": 0.8, "hidden_bullish": 0.6,
        "bearish": -0.8, "hidden_bearish": -0.6,
        "none": 0.0,
    }
    div_val = div_map.get(result.divergence_type, 0.0)
    if div_val != 0.0:
        score += div_val * 0.25
        weights_total += 0.25

    # Thrust override (very powerful signal)
    if result.thrust_detected:
        if "Bullish" in result.thrust_description:
            score = max(score, 0.8)
        elif "Bearish" in result.thrust_description:
            score = min(score, -0.8)

    result.momentum_score = max(-1.0, min(1.0, score))
