"""
Multi-Timeframe Trend Analysis Module (Dimension 1: Structure)

Answers: "What direction is the trend, on what timeframes, and how strong?"

Key intelligence features:
  - Ichimoku Cloud: trend direction, momentum, forward-looking S/R
  - ADX: trend strength (trending vs ranging)
  - MA Constellation: 8/21/50/200 alignment for trend quality
  - Trend Regime Classification: strong_uptrend → strong_downtrend
  - Trend Maturity: fresh vs late-stage trends
  - Higher-Highs/Higher-Lows: quantified Dow Theory structure

All computation is pure Python + pandas-ta. No API calls.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import finias.agents.technical_analyst.computations.indicators as ta


@dataclass
class TrendAnalysis:
    """Complete trend analysis for a single symbol."""

    symbol: str = ""

    # Trend Regime
    trend_regime: str = "unknown"  # strong_uptrend, uptrend, consolidation, downtrend, strong_downtrend
    trend_score: float = 0.0  # -1 (max bearish) to +1 (max bullish)

    # MA Constellation
    price_vs_sma8: str = "unknown"   # above / below
    price_vs_sma21: str = "unknown"
    price_vs_sma50: str = "unknown"
    price_vs_sma200: str = "unknown"
    ma_alignment: str = "unknown"  # perfect_bull, bull, mixed, bear, perfect_bear
    ma_alignment_score: float = 0.0  # -1 to +1

    # MA Slopes (annualized rate of change of the MA itself)
    sma50_slope: Optional[float] = None
    sma200_slope: Optional[float] = None

    # ADX (trend strength)
    adx: Optional[float] = None  # 0-100, >25 = trending
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    adx_trending: bool = False  # True if ADX > 25

    # Ichimoku Cloud
    ichimoku_signal: str = "neutral"  # strong_bullish, bullish, neutral, bearish, strong_bearish
    price_vs_cloud: str = "unknown"  # above, inside, below
    cloud_color: str = "unknown"  # green (bullish) / red (bearish)
    tenkan_kijun_cross: str = "none"  # bullish_cross, bearish_cross, none
    future_cloud_bullish: Optional[bool] = None  # Senkou Span A > B projected forward

    # Trend Maturity
    trend_age_bars: int = 0  # How many bars the current trend regime has persisted
    trend_maturity: str = "unknown"  # nascent (<20 bars), developing (20-60), mature (60-120), late (>120)

    # Higher-Highs / Higher-Lows (Dow Theory)
    hh_hl_intact: Optional[bool] = None  # True = uptrend structure intact
    ll_lh_intact: Optional[bool] = None  # True = downtrend structure intact
    structure_description: str = ""

    def to_dict(self) -> dict:
        return {
            "trend_regime": self.trend_regime,
            "trend_score": round(self.trend_score, 4),
            "ma": {
                "price_vs_sma8": self.price_vs_sma8,
                "price_vs_sma21": self.price_vs_sma21,
                "price_vs_sma50": self.price_vs_sma50,
                "price_vs_sma200": self.price_vs_sma200,
                "alignment": self.ma_alignment,
                "alignment_score": round(self.ma_alignment_score, 4),
                "sma50_slope": round(self.sma50_slope, 4) if self.sma50_slope is not None else None,
                "sma200_slope": round(self.sma200_slope, 4) if self.sma200_slope is not None else None,
            },
            "adx": {
                "value": round(self.adx, 2) if self.adx is not None else None,
                "plus_di": round(self.plus_di, 2) if self.plus_di is not None else None,
                "minus_di": round(self.minus_di, 2) if self.minus_di is not None else None,
                "trending": self.adx_trending,
            },
            "ichimoku": {
                "signal": self.ichimoku_signal,
                "price_vs_cloud": self.price_vs_cloud,
                "cloud_color": self.cloud_color,
                "tenkan_kijun_cross": self.tenkan_kijun_cross,
                "future_cloud_bullish": self.future_cloud_bullish,
            },
            "maturity": {
                "age_bars": self.trend_age_bars,
                "stage": self.trend_maturity,
            },
            "structure": {
                "hh_hl_intact": self.hh_hl_intact,
                "ll_lh_intact": self.ll_lh_intact,
                "description": self.structure_description,
            },
        }


def analyze_trend(df: pd.DataFrame, symbol: str = "") -> TrendAnalysis:
    """
    Compute multi-timeframe trend analysis for a single symbol.

    Args:
        df: DataFrame with columns: open, high, low, close, volume.
            Must be sorted chronologically (oldest first).
            Needs at least 200 rows for full analysis.
        symbol: Ticker symbol for labeling.

    Returns:
        TrendAnalysis dataclass with all trend signals.
    """
    result = TrendAnalysis(symbol=symbol)

    if df is None or len(df) < 50:
        return result

    close = df["close"].values
    current_price = float(close[-1])

    # === MA Constellation ===
    _compute_ma_constellation(df, result, current_price)

    # === ADX ===
    _compute_adx(df, result)

    # === Ichimoku Cloud ===
    if len(df) >= 52:  # Ichimoku needs at least 52 bars
        _compute_ichimoku(df, result, current_price)

    # === Trend Regime Classification ===
    _classify_trend_regime(result)

    # === MA Slopes ===
    _compute_ma_slopes(df, result)

    # === Higher-Highs / Higher-Lows ===
    if len(df) >= 60:
        _compute_swing_structure(df, result)

    # === Trend Maturity ===
    # This is a simplified version — in production, we'd track regime changes
    # across refreshes using the signal lifecycle in signals.py (Prompt 4)
    _estimate_trend_maturity(df, result)

    return result


def _compute_ma_constellation(df: pd.DataFrame, result: TrendAnalysis, price: float):
    """Compute SMA/EMA constellation and alignment."""
    sma8 = ta.sma(df["close"], length=8)
    sma21 = ta.sma(df["close"], length=21)
    sma50 = ta.sma(df["close"], length=50)
    sma200 = ta.sma(df["close"], length=200) if len(df) >= 200 else None

    # Current values
    s8 = float(sma8.iloc[-1]) if sma8 is not None and not sma8.empty else None
    s21 = float(sma21.iloc[-1]) if sma21 is not None and not sma21.empty else None
    s50 = float(sma50.iloc[-1]) if sma50 is not None and not sma50.empty else None
    s200 = float(sma200.iloc[-1]) if sma200 is not None and not sma200.empty else None

    # Price vs MAs
    if s8 is not None:
        result.price_vs_sma8 = "above" if price > s8 else "below"
    if s21 is not None:
        result.price_vs_sma21 = "above" if price > s21 else "below"
    if s50 is not None:
        result.price_vs_sma50 = "above" if price > s50 else "below"
    if s200 is not None:
        result.price_vs_sma200 = "above" if price > s200 else "below"

    # MA Alignment scoring
    # Perfect bull: price > 8 > 21 > 50 > 200 (all rising)
    # Perfect bear: price < 8 < 21 < 50 < 200 (all falling)
    ma_values = [v for v in [s8, s21, s50, s200] if v is not None]
    if len(ma_values) >= 3:
        # Count how many consecutive MAs are in descending order (bullish)
        bull_score = 0
        bear_score = 0
        all_vals = [price] + ma_values

        for i in range(len(all_vals) - 1):
            if all_vals[i] > all_vals[i + 1]:
                bull_score += 1
            elif all_vals[i] < all_vals[i + 1]:
                bear_score += 1

        total_pairs = len(all_vals) - 1
        if total_pairs > 0:
            result.ma_alignment_score = (bull_score - bear_score) / total_pairs

        if result.ma_alignment_score >= 0.9:
            result.ma_alignment = "perfect_bull"
        elif result.ma_alignment_score >= 0.5:
            result.ma_alignment = "bull"
        elif result.ma_alignment_score <= -0.9:
            result.ma_alignment = "perfect_bear"
        elif result.ma_alignment_score <= -0.5:
            result.ma_alignment = "bear"
        else:
            result.ma_alignment = "mixed"


def _compute_adx(df: pd.DataFrame, result: TrendAnalysis):
    """Compute ADX and directional indicators."""
    if len(df) < 28:  # ADX needs ~28 bars minimum
        return

    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_df is None or adx_df.empty:
        return

    # pandas-ta returns columns: ADX_14, DMP_14, DMN_14
    adx_col = [c for c in adx_df.columns if c.startswith("ADX")]
    dmp_col = [c for c in adx_df.columns if c.startswith("DMP")]
    dmn_col = [c for c in adx_df.columns if c.startswith("DMN")]

    if adx_col:
        result.adx = float(adx_df[adx_col[0]].iloc[-1])
    if dmp_col:
        result.plus_di = float(adx_df[dmp_col[0]].iloc[-1])
    if dmn_col:
        result.minus_di = float(adx_df[dmn_col[0]].iloc[-1])

    if result.adx is not None:
        result.adx_trending = result.adx > 25


def _compute_ichimoku(df: pd.DataFrame, result: TrendAnalysis, price: float):
    """Compute Ichimoku Cloud signals."""
    try:
        ichi = ta.ichimoku(df["high"], df["low"], df["close"])
        if ichi is None or len(ichi) < 2:
            return

        # ichimoku returns a tuple: (ichimoku_df, span_df)
        ichi_df = ichi[0]
        span_df = ichi[1] if len(ichi) > 1 else None

        # Get Senkou Span values from the main dataframe for current cloud
        # pandas-ta ichimoku column naming: ITS_9, IKS_26, ISA_9, ISB_26, ICS_26
        cols = ichi_df.columns.tolist()

        tenkan = None
        kijun = None
        span_a = None
        span_b = None

        for c in cols:
            if c.startswith("ITS_"):
                tenkan = float(ichi_df[c].iloc[-1]) if not pd.isna(ichi_df[c].iloc[-1]) else None
            elif c.startswith("IKS_"):
                kijun = float(ichi_df[c].iloc[-1]) if not pd.isna(ichi_df[c].iloc[-1]) else None
            elif c.startswith("ISA_"):
                span_a = float(ichi_df[c].iloc[-1]) if not pd.isna(ichi_df[c].iloc[-1]) else None
            elif c.startswith("ISB_"):
                span_b = float(ichi_df[c].iloc[-1]) if not pd.isna(ichi_df[c].iloc[-1]) else None

        # Cloud color
        if span_a is not None and span_b is not None:
            result.cloud_color = "green" if span_a > span_b else "red"

            # Price vs cloud
            cloud_top = max(span_a, span_b)
            cloud_bottom = min(span_a, span_b)
            if price > cloud_top:
                result.price_vs_cloud = "above"
            elif price < cloud_bottom:
                result.price_vs_cloud = "below"
            else:
                result.price_vs_cloud = "inside"

        # Tenkan/Kijun cross
        if tenkan is not None and kijun is not None:
            # Check previous bar for cross detection
            prev_tenkan = None
            prev_kijun = None
            for c in cols:
                if c.startswith("ITS_"):
                    prev_tenkan = float(ichi_df[c].iloc[-2]) if len(ichi_df) >= 2 and not pd.isna(ichi_df[c].iloc[-2]) else None
                elif c.startswith("IKS_"):
                    prev_kijun = float(ichi_df[c].iloc[-2]) if len(ichi_df) >= 2 and not pd.isna(ichi_df[c].iloc[-2]) else None

            if prev_tenkan is not None and prev_kijun is not None:
                if tenkan > kijun and prev_tenkan <= prev_kijun:
                    result.tenkan_kijun_cross = "bullish_cross"
                elif tenkan < kijun and prev_tenkan >= prev_kijun:
                    result.tenkan_kijun_cross = "bearish_cross"

        # Future cloud (from span_df if available)
        if span_df is not None and not span_df.empty:
            future_cols_a = [c for c in span_df.columns if c.startswith("ISA_")]
            future_cols_b = [c for c in span_df.columns if c.startswith("ISB_")]
            if future_cols_a and future_cols_b:
                fa = span_df[future_cols_a[0]].iloc[-1] if not pd.isna(span_df[future_cols_a[0]].iloc[-1]) else None
                fb = span_df[future_cols_b[0]].iloc[-1] if not pd.isna(span_df[future_cols_b[0]].iloc[-1]) else None
                if fa is not None and fb is not None:
                    result.future_cloud_bullish = float(fa) > float(fb)

        # Composite Ichimoku signal
        ichi_score = 0
        if result.price_vs_cloud == "above":
            ichi_score += 1
        elif result.price_vs_cloud == "below":
            ichi_score -= 1

        if result.cloud_color == "green":
            ichi_score += 1
        elif result.cloud_color == "red":
            ichi_score -= 1

        if result.tenkan_kijun_cross == "bullish_cross":
            ichi_score += 1
        elif result.tenkan_kijun_cross == "bearish_cross":
            ichi_score -= 1

        if result.future_cloud_bullish is True:
            ichi_score += 0.5
        elif result.future_cloud_bullish is False:
            ichi_score -= 0.5

        if ichi_score >= 2.5:
            result.ichimoku_signal = "strong_bullish"
        elif ichi_score >= 1:
            result.ichimoku_signal = "bullish"
        elif ichi_score <= -2.5:
            result.ichimoku_signal = "strong_bearish"
        elif ichi_score <= -1:
            result.ichimoku_signal = "bearish"
        else:
            result.ichimoku_signal = "neutral"

    except Exception as e:
        # Ichimoku can fail with insufficient data — non-blocking
        import logging
        logging.getLogger("finias.ta.trend").warning(f"Ichimoku computation failed for {result.symbol}: {e}")


def _compute_ma_slopes(df: pd.DataFrame, result: TrendAnalysis):
    """Compute MA slopes as annualized rate of change."""
    if len(df) < 200:
        return

    sma50 = ta.sma(df["close"], length=50)
    sma200 = ta.sma(df["close"], length=200)

    if sma50 is not None and len(sma50.dropna()) >= 20:
        slope_20d = (float(sma50.iloc[-1]) / float(sma50.iloc[-20]) - 1)
        result.sma50_slope = slope_20d * (252 / 20)  # Annualize

    if sma200 is not None and len(sma200.dropna()) >= 20:
        slope_20d = (float(sma200.iloc[-1]) / float(sma200.iloc[-20]) - 1)
        result.sma200_slope = slope_20d * (252 / 20)


def _classify_trend_regime(result: TrendAnalysis):
    """
    Classify the trend regime from computed signals.

    Uses MA alignment, ADX, and Ichimoku for composite classification.
    """
    score = 0.0
    components = 0

    # MA Alignment (weight: 40%)
    score += result.ma_alignment_score * 0.40
    components += 1

    # ADX + DI Direction (weight: 30%)
    if result.adx is not None and result.plus_di is not None and result.minus_di is not None:
        if result.adx_trending:
            # Strong trend — direction from DI
            di_direction = 1.0 if result.plus_di > result.minus_di else -1.0
            # Scale by ADX strength (25-50 range normalized to 0-1)
            adx_strength = min((result.adx - 25) / 25, 1.0)
            score += di_direction * adx_strength * 0.30
        else:
            # Weak trend — contributes 0
            score += 0.0
        components += 1

    # Ichimoku (weight: 30%)
    ichi_map = {
        "strong_bullish": 1.0, "bullish": 0.5, "neutral": 0.0,
        "bearish": -0.5, "strong_bearish": -1.0,
    }
    ichi_val = ichi_map.get(result.ichimoku_signal, 0.0)
    score += ichi_val * 0.30
    components += 1

    result.trend_score = max(-1.0, min(1.0, score))

    # Classify regime
    if result.trend_score >= 0.5:
        result.trend_regime = "strong_uptrend"
    elif result.trend_score >= 0.2:
        result.trend_regime = "uptrend"
    elif result.trend_score <= -0.5:
        result.trend_regime = "strong_downtrend"
    elif result.trend_score <= -0.2:
        result.trend_regime = "downtrend"
    else:
        result.trend_regime = "consolidation"


def _compute_swing_structure(df: pd.DataFrame, result: TrendAnalysis):
    """
    Detect higher-highs/higher-lows (uptrend) or lower-lows/lower-highs (downtrend).

    Uses a simple swing detection: local highs/lows over a lookback window.
    """
    highs = df["high"].values
    lows = df["low"].values
    lookback = 10  # Bars to look left/right for swing detection

    # Find swing highs and lows (last 120 bars)
    analysis_window = min(len(highs), 120)
    h = highs[-analysis_window:]
    l = lows[-analysis_window:]

    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(h) - lookback):
        # Swing high: higher than all neighbors within lookback
        if h[i] == max(h[i - lookback:i + lookback + 1]):
            swing_highs.append((i, float(h[i])))
        # Swing low: lower than all neighbors within lookback
        if l[i] == min(l[i - lookback:i + lookback + 1]):
            swing_lows.append((i, float(l[i])))

    # Check last 3-4 swings for structure
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        # Higher highs?
        recent_highs = [s[1] for s in swing_highs[-3:]]
        hh = all(recent_highs[i] > recent_highs[i - 1] for i in range(1, len(recent_highs)))

        # Higher lows?
        recent_lows = [s[1] for s in swing_lows[-3:]]
        hl = all(recent_lows[i] > recent_lows[i - 1] for i in range(1, len(recent_lows)))

        # Lower lows?
        ll = all(recent_lows[i] < recent_lows[i - 1] for i in range(1, len(recent_lows)))

        # Lower highs?
        lh = all(recent_highs[i] < recent_highs[i - 1] for i in range(1, len(recent_highs)))

        result.hh_hl_intact = hh and hl
        result.ll_lh_intact = ll and lh

        if hh and hl:
            result.structure_description = "Higher highs and higher lows — uptrend structure intact"
        elif ll and lh:
            result.structure_description = "Lower lows and lower highs — downtrend structure intact"
        elif hh and not hl:
            result.structure_description = "Higher highs but not higher lows — uptrend weakening"
        elif ll and not lh:
            result.structure_description = "Lower lows but not lower highs — downtrend weakening"
        else:
            result.structure_description = "Mixed swing structure — no clear trend"


def _estimate_trend_maturity(df: pd.DataFrame, result: TrendAnalysis):
    """
    Estimate how long the current trend regime has persisted.

    Simplified: counts consecutive bars where the 8 SMA has been
    above (uptrend) or below (downtrend) the 21 SMA.
    """
    if len(df) < 30:
        return

    sma8 = ta.sma(df["close"], length=8)
    sma21 = ta.sma(df["close"], length=21)

    if sma8 is None or sma21 is None:
        return

    diff = sma8 - sma21
    diff = diff.dropna()

    if len(diff) < 5:
        return

    # Count consecutive bars from the end with same sign
    current_sign = 1 if float(diff.iloc[-1]) > 0 else -1
    count = 0
    for i in range(len(diff) - 1, -1, -1):
        val = float(diff.iloc[i])
        if (val > 0 and current_sign > 0) or (val < 0 and current_sign < 0):
            count += 1
        else:
            break

    result.trend_age_bars = count

    if count < 20:
        result.trend_maturity = "nascent"
    elif count < 60:
        result.trend_maturity = "developing"
    elif count < 120:
        result.trend_maturity = "mature"
    else:
        result.trend_maturity = "late"
