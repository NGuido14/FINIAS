"""
Volatility & Squeeze Detection Module (Dimension 5: Volatility)

Answers: "Is this stock coiling for a big move? Is volatility expanding or contracting?"

Key metrics:
  - Bollinger Squeeze: When Bollinger Bands contract inside Keltner Channels,
    a big move is imminent. Direction unknown but magnitude predictable.
  - ATR Analysis: Average True Range trend — expanding = trending, contracting = coiling
  - Historical Volatility Percentile: Where is current vol vs past year?
  - Volatility Regime: low_vol, normal, high_vol, extreme

All computation is pure Python + pandas-ta. No API calls.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import pandas_ta as ta


@dataclass
class VolatilityAnalysis:
    """Complete volatility analysis for a single symbol."""

    symbol: str = ""

    # ATR
    atr_14: Optional[float] = None
    atr_pct: Optional[float] = None  # ATR as % of price
    atr_trend: str = "neutral"  # expanding, contracting, neutral (20-bar trend)
    atr_slope: Optional[float] = None  # Normalized slope of ATR

    # Bollinger Squeeze
    bb_bandwidth: Optional[float] = None
    bb_bandwidth_percentile: Optional[float] = None  # 0-100, where current BW ranks vs 1yr
    squeeze_on: bool = False  # BB inside Keltner = squeeze active
    squeeze_bars: int = 0  # How many bars the squeeze has been active
    squeeze_released: bool = False  # Was a squeeze active recently and just broke?

    # Historical Volatility
    hvol_20d: Optional[float] = None  # 20-day annualized historical vol
    hvol_percentile: Optional[float] = None  # Where 20d hvol ranks vs past 252 days

    # Volatility Regime
    vol_regime: str = "normal"  # low_vol, normal, high_vol, extreme
    vol_score: float = 0.0  # -1 (extreme compression) to +1 (extreme expansion)

    def to_dict(self) -> dict:
        return {
            "atr": {
                "value": round(self.atr_14, 4) if self.atr_14 is not None else None,
                "pct_of_price": round(self.atr_pct, 4) if self.atr_pct is not None else None,
                "trend": self.atr_trend,
                "slope": round(self.atr_slope, 6) if self.atr_slope is not None else None,
            },
            "squeeze": {
                "bb_bandwidth": round(self.bb_bandwidth, 4) if self.bb_bandwidth is not None else None,
                "bb_bandwidth_percentile": round(self.bb_bandwidth_percentile, 1) if self.bb_bandwidth_percentile is not None else None,
                "active": self.squeeze_on,
                "bars": self.squeeze_bars,
                "just_released": self.squeeze_released,
            },
            "historical_vol": {
                "hvol_20d": round(self.hvol_20d, 4) if self.hvol_20d is not None else None,
                "percentile": round(self.hvol_percentile, 1) if self.hvol_percentile is not None else None,
            },
            "vol_regime": self.vol_regime,
            "vol_score": round(self.vol_score, 4),
        }


def analyze_volatility(df: pd.DataFrame, symbol: str = "") -> VolatilityAnalysis:
    """
    Compute volatility and squeeze detection for a single symbol.

    Args:
        df: OHLCV DataFrame, sorted chronologically. Needs 252+ bars for full analysis.
        symbol: Ticker symbol.

    Returns:
        VolatilityAnalysis with all volatility signals.
    """
    result = VolatilityAnalysis(symbol=symbol)

    if df is None or len(df) < 30:
        return result

    price = float(df["close"].iloc[-1])

    # === ATR ===
    _compute_atr(df, result, price)

    # === Bollinger Squeeze ===
    _compute_squeeze(df, result)

    # === Historical Volatility ===
    _compute_hvol(df, result)

    # === Vol Regime Classification ===
    _classify_vol_regime(result)

    return result


def _compute_atr(df: pd.DataFrame, result: VolatilityAnalysis, price: float):
    """Compute ATR and its trend."""
    atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
    if atr_series is None or atr_series.empty:
        return

    result.atr_14 = float(atr_series.iloc[-1])
    result.atr_pct = result.atr_14 / price if price > 0 else None

    # ATR trend over 20 bars
    atr_vals = atr_series.dropna().values
    if len(atr_vals) >= 20:
        recent = atr_vals[-20:]
        x = np.arange(20)
        slope = np.polyfit(x, recent, 1)[0]
        mean_atr = np.mean(recent)
        result.atr_slope = slope / mean_atr if mean_atr > 0 else 0

        if result.atr_slope > 0.02:
            result.atr_trend = "expanding"
        elif result.atr_slope < -0.02:
            result.atr_trend = "contracting"


def _compute_squeeze(df: pd.DataFrame, result: VolatilityAnalysis):
    """
    Detect Bollinger Squeeze: BB contracts inside Keltner Channels.

    When Bollinger Bands (volatility measure) are INSIDE Keltner Channels
    (ATR-based channels), the stock is in a low-volatility squeeze.
    When the squeeze releases, a large directional move typically follows.
    """
    if len(df) < 30:
        return

    # Bollinger Bands
    bbands = ta.bbands(df["close"], length=20, std=2.0)
    if bbands is None or bbands.empty:
        return

    bb_upper_col = [c for c in bbands.columns if c.startswith("BBU_")][0]
    bb_lower_col = [c for c in bbands.columns if c.startswith("BBL_")][0]
    bb_bw_col = [c for c in bbands.columns if c.startswith("BBB_")][0]

    result.bb_bandwidth = float(bbands[bb_bw_col].iloc[-1])

    # BB bandwidth percentile (where current BW ranks vs last 252 bars)
    bw_values = bbands[bb_bw_col].dropna().values
    if len(bw_values) >= 50:
        lookback = min(252, len(bw_values))
        historical_bw = bw_values[-lookback:]
        current_bw = bw_values[-1]
        below_count = np.sum(historical_bw < current_bw)
        result.bb_bandwidth_percentile = (below_count / len(historical_bw)) * 100

    # Keltner Channels (for squeeze detection)
    kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=1.5)
    if kc is None or kc.empty:
        return

    kc_upper_col = [c for c in kc.columns if c.startswith("KCU")][0]
    kc_lower_col = [c for c in kc.columns if c.startswith("KCL")][0]

    # Squeeze: BB inside KC
    bb_upper = bbands[bb_upper_col]
    bb_lower = bbands[bb_lower_col]
    kc_upper = kc[kc_upper_col]
    kc_lower = kc[kc_lower_col]

    squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    squeeze_clean = squeeze.dropna()

    if len(squeeze_clean) > 0:
        result.squeeze_on = bool(squeeze_clean.iloc[-1])

        # Count consecutive squeeze bars
        if result.squeeze_on:
            count = 0
            for i in range(len(squeeze_clean) - 1, -1, -1):
                if squeeze_clean.iloc[i]:
                    count += 1
                else:
                    break
            result.squeeze_bars = count

        # Squeeze just released: was on 2 bars ago, off now
        if len(squeeze_clean) >= 3:
            was_on = bool(squeeze_clean.iloc[-3]) or bool(squeeze_clean.iloc[-2])
            is_off = not bool(squeeze_clean.iloc[-1])
            result.squeeze_released = was_on and is_off


def _compute_hvol(df: pd.DataFrame, result: VolatilityAnalysis):
    """Compute historical (realized) volatility and its percentile."""
    close = df["close"]
    if len(close) < 30:
        return

    # 20-day annualized historical volatility
    log_returns = np.log(close / close.shift(1)).dropna()
    if len(log_returns) < 20:
        return

    hvol_20 = float(log_returns.iloc[-20:].std() * np.sqrt(252))
    result.hvol_20d = hvol_20

    # Percentile vs past year
    if len(log_returns) >= 252:
        rolling_vol = log_returns.rolling(20).std() * np.sqrt(252)
        vol_values = rolling_vol.dropna().values
        lookback = min(252, len(vol_values))
        historical = vol_values[-lookback:]
        below_count = np.sum(historical < hvol_20)
        result.hvol_percentile = (below_count / len(historical)) * 100


def _classify_vol_regime(result: VolatilityAnalysis):
    """Classify the volatility regime and compute vol score."""
    # Score based on percentiles and trends
    score = 0.0

    # HVol percentile
    if result.hvol_percentile is not None:
        if result.hvol_percentile < 20:
            score -= 0.4  # Low vol = compressed
            result.vol_regime = "low_vol"
        elif result.hvol_percentile > 80:
            score += 0.4  # High vol = expanded
            result.vol_regime = "high_vol"
            if result.hvol_percentile > 95:
                result.vol_regime = "extreme"
                score += 0.2
        else:
            result.vol_regime = "normal"

    # Squeeze adds to compression signal
    if result.squeeze_on:
        score -= 0.3  # Squeeze = compression
    if result.squeeze_released:
        score += 0.3  # Release = expansion imminent

    # ATR trend
    if result.atr_trend == "expanding":
        score += 0.2
    elif result.atr_trend == "contracting":
        score -= 0.2

    result.vol_score = max(-1.0, min(1.0, score))
