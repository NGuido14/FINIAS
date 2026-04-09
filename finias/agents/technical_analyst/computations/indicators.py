"""
Pure pandas/numpy technical indicator implementations.

Replaces pandas-ta dependency with direct implementations for reliability
across all Python versions. Each function matches the pandas-ta API signature
where possible but uses only pandas and numpy.

This module is the single source of truth for all TA indicator calculations.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def sma(series: pd.Series, length: int = 10) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=length).mean()


def ema(series: pd.Series, length: int = 10) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing (equivalent to EMA with alpha = 1/length)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    return result


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).

    Returns DataFrame with columns: MACD_{fast}_{slow}_{signal}, MACDh_{...}, MACDs_{...}
    """
    fast_ema = ema(series, length=fast)
    slow_ema = ema(series, length=slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, length=signal)
    histogram = macd_line - signal_line

    suffix = f"_{fast}_{slow}_{signal}"
    return pd.DataFrame({
        f"MACD{suffix}": macd_line,
        f"MACDh{suffix}": histogram,
        f"MACDs{suffix}": signal_line,
    })


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    """
    Average Directional Index with +DI and -DI.

    Returns DataFrame with columns: ADX_{length}, DMP_{length}, DMN_{length}
    """
    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    # Wilder's smoothing
    alpha = 1.0 / length
    atr = tr.ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, min_periods=length, adjust=False).mean()

    # Directional Indicators
    plus_di = 100.0 * plus_dm_smooth / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_smooth / atr.replace(0, np.nan)

    # DX and ADX
    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx_val = dx.ewm(alpha=alpha, min_periods=length, adjust=False).mean()

    return pd.DataFrame({
        f"ADX_{length}": adx_val,
        f"DMP_{length}": plus_di,
        f"DMN_{length}": minus_di,
    })


def stoch(high: pd.Series, low: pd.Series, close: pd.Series,
          k: int = 14, d: int = 3, smooth_k: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator.

    Returns DataFrame with columns: STOCHk_{k}_{d}_{smooth_k}, STOCHd_{k}_{d}_{smooth_k}
    """
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()

    range_ = highest_high - lowest_low
    fast_k = 100.0 * (close - lowest_low) / range_.replace(0, np.nan)

    # Smooth %K
    slow_k = fast_k.rolling(window=smooth_k).mean()
    slow_d = slow_k.rolling(window=d).mean()

    suffix = f"_{k}_{d}_{smooth_k}"
    return pd.DataFrame({
        f"STOCHk{suffix}": slow_k,
        f"STOCHd{suffix}": slow_d,
    })


def bbands(series: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands.

    Returns DataFrame with columns: BBL_{length}_{std}, BBM_{...}, BBU_{...}, BBB_{...}, BBP_{...}
    """
    mid = sma(series, length=length)
    stdev = series.rolling(window=length).std()

    upper = mid + std * stdev
    lower = mid - std * stdev

    bandwidth = (upper - lower) / mid.replace(0, np.nan) * 100
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)

    suffix = f"_{length}_{std}"
    return pd.DataFrame({
        f"BBL{suffix}": lower,
        f"BBM{suffix}": mid,
        f"BBU{suffix}": upper,
        f"BBB{suffix}": bandwidth,
        f"BBP{suffix}": pct_b,
    })


def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
             tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> tuple:
    """
    Ichimoku Cloud.

    Returns tuple of (ichimoku_df, span_df):
      ichimoku_df: ITS_{tenkan}, IKS_{kijun}, ISA_{tenkan}, ISB_{kijun}, ICS_{kijun}
      span_df: forward-projected Senkou spans (ISA, ISB shifted forward by kijun periods)
    """
    # Tenkan-sen (Conversion Line)
    tenkan_high = high.rolling(window=tenkan).max()
    tenkan_low = low.rolling(window=tenkan).min()
    its = (tenkan_high + tenkan_low) / 2

    # Kijun-sen (Base Line)
    kijun_high = high.rolling(window=kijun).max()
    kijun_low = low.rolling(window=kijun).min()
    iks = (kijun_high + kijun_low) / 2

    # Senkou Span A (Leading Span A) — average of Tenkan and Kijun
    isa = (its + iks) / 2

    # Senkou Span B (Leading Span B) — midpoint of 52-period high/low
    senkou_high = high.rolling(window=senkou).max()
    senkou_low = low.rolling(window=senkou).min()
    isb = (senkou_high + senkou_low) / 2

    # Chikou Span (Lagging Span) — close shifted back kijun periods
    ics = close.shift(-kijun)

    ichimoku_df = pd.DataFrame({
        f"ITS_{tenkan}": its,
        f"IKS_{kijun}": iks,
        f"ISA_{tenkan}": isa.shift(kijun),   # Shifted forward to match chart cloud position
        f"ISB_{kijun}": isb.shift(kijun),    # Shifted forward to match chart cloud position
        f"ICS_{kijun}": ics,
    })

    # Forward-projected spans beyond current data (for future cloud visualization)
    # These extend kijun periods past the last bar
    future_index = pd.RangeIndex(start=len(high), stop=len(high) + kijun)
    span_df = pd.DataFrame({
        f"ISA_{tenkan}": pd.concat([isa.iloc[-kijun:].reset_index(drop=True)], ignore_index=True),
        f"ISB_{kijun}": pd.concat([isb.iloc[-kijun:].reset_index(drop=True)], ignore_index=True),
    }, index=future_index)

    return (ichimoku_df, span_df)
