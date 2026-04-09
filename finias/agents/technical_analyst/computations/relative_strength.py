"""
Relative Strength Module (Dimension 4: Relative Performance)

Answers: "Is this stock outperforming or underperforming its peers?"

This is one of the most documented alpha factors in quantitative finance.
Our backtest found that trend regime alone has modest alpha. Relative
strength answers the critical question: "Among oversold stocks, which ones
are starting to turn?"

Key metrics:
  - Stock vs Sector RS Ratio: is the stock outperforming its sector?
  - Sector vs SPY RS Ratio: is the sector outperforming the market?
  - RS Momentum: is relative strength improving or deteriorating?
  - RS Percentile: where does this stock rank vs the universe?

All computation is pure Python. No API calls.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class RelativeStrengthAnalysis:
    """Complete relative strength analysis for a single symbol."""

    symbol: str = ""
    sector: str = "unknown"

    # Stock vs Sector
    rs_vs_sector: Optional[float] = None  # Current RS ratio (normalized to 1.0 at start)
    rs_vs_sector_sma20: Optional[float] = None  # 20-day SMA of RS ratio
    rs_vs_sector_trend: str = "neutral"  # improving, deteriorating, neutral
    rs_vs_sector_momentum: Optional[float] = None  # 20-day ROC of RS ratio

    # Sector vs SPY (market context)
    sector_vs_spy: Optional[float] = None
    sector_vs_spy_trend: str = "neutral"  # improving, deteriorating, neutral
    sector_momentum_20d: Optional[float] = None  # Sector 20d return relative to SPY

    # Universe Ranking
    rs_percentile: Optional[float] = None  # 0-100, where this stock ranks in 20d return vs peers

    # Combined Assessment
    rs_regime: str = "neutral"  # leading, improving, lagging, deteriorating
    rs_score: float = 0.0  # -1 (lagging & deteriorating) to +1 (leading & improving)

    def to_dict(self) -> dict:
        return {
            "sector": self.sector,
            "vs_sector": {
                "rs_ratio": round(self.rs_vs_sector, 4) if self.rs_vs_sector is not None else None,
                "rs_sma20": round(self.rs_vs_sector_sma20, 4) if self.rs_vs_sector_sma20 is not None else None,
                "trend": self.rs_vs_sector_trend,
                "momentum": round(self.rs_vs_sector_momentum, 4) if self.rs_vs_sector_momentum is not None else None,
            },
            "sector_vs_spy": {
                "rs_ratio": round(self.sector_vs_spy, 4) if self.sector_vs_spy is not None else None,
                "trend": self.sector_vs_spy_trend,
                "momentum_20d": round(self.sector_momentum_20d, 4) if self.sector_momentum_20d is not None else None,
            },
            "rs_percentile": round(self.rs_percentile, 1) if self.rs_percentile is not None else None,
            "rs_regime": self.rs_regime,
            "rs_score": round(self.rs_score, 4),
        }


# Map GICS sectors to their ETF proxy
SECTOR_ETF_MAP = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}


def analyze_relative_strength(
    df: pd.DataFrame,
    symbol: str = "",
    sector: str = "unknown",
    sector_etf_df: pd.DataFrame = None,
    spy_df: pd.DataFrame = None,
    universe_returns_20d: dict = None,
) -> RelativeStrengthAnalysis:
    """
    Compute relative strength for a single symbol.

    Args:
        df: Symbol's OHLCV DataFrame, sorted chronologically.
        symbol: Ticker symbol.
        sector: GICS sector name.
        sector_etf_df: Sector ETF's OHLCV DataFrame (e.g., XLK for tech stocks).
        spy_df: SPY's OHLCV DataFrame (market benchmark).
        universe_returns_20d: Dict of {symbol: 20d_return} for percentile ranking.

    Returns:
        RelativeStrengthAnalysis with all RS signals.
    """
    result = RelativeStrengthAnalysis(symbol=symbol, sector=sector)

    if df is None or len(df) < 30:
        return result

    # === Stock vs Sector RS ===
    if sector_etf_df is not None and len(sector_etf_df) >= 30:
        _compute_rs_vs_benchmark(
            df, sector_etf_df, result,
            target_attr_prefix="rs_vs_sector",
        )

    # === Sector vs SPY RS ===
    if sector_etf_df is not None and spy_df is not None and len(spy_df) >= 30:
        _compute_sector_vs_spy(sector_etf_df, spy_df, result)

    # === Universe Percentile Ranking ===
    if universe_returns_20d and symbol in universe_returns_20d:
        all_returns = list(universe_returns_20d.values())
        my_return = universe_returns_20d[symbol]
        if all_returns and my_return is not None:
            below_count = sum(1 for r in all_returns if r is not None and r < my_return)
            total_valid = sum(1 for r in all_returns if r is not None)
            if total_valid > 0:
                result.rs_percentile = (below_count / total_valid) * 100

    # === RS Regime Classification ===
    _classify_rs_regime(result)

    return result


def _compute_rs_vs_benchmark(
    stock_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    result: RelativeStrengthAnalysis,
    target_attr_prefix: str,
):
    """Compute relative strength ratio of stock vs a benchmark."""
    # Align by finding common date range
    stock_close = stock_df["close"].values
    bench_close = bench_df["close"].values

    # Use the shorter length
    min_len = min(len(stock_close), len(bench_close))
    if min_len < 30:
        return

    stock_close = stock_close[-min_len:]
    bench_close = bench_close[-min_len:]

    # RS ratio: stock price / benchmark price (normalized so start = 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs_ratio = (stock_close / stock_close[0]) / (bench_close / bench_close[0])
        rs_ratio = np.where(np.isfinite(rs_ratio), rs_ratio, np.nan)

    rs_series = pd.Series(rs_ratio)

    # Current RS ratio
    current_rs = float(rs_series.iloc[-1]) if not np.isnan(rs_series.iloc[-1]) else None
    setattr(result, target_attr_prefix, current_rs)

    # 20-day SMA of RS ratio
    if len(rs_series) >= 20:
        rs_sma = rs_series.rolling(20).mean()
        sma_val = float(rs_sma.iloc[-1]) if not np.isnan(rs_sma.iloc[-1]) else None
        setattr(result, f"{target_attr_prefix}_sma20", sma_val)

        # RS trend: is current RS above or below its 20-day SMA?
        if current_rs is not None and sma_val is not None:
            if current_rs > sma_val * 1.005:
                setattr(result, f"{target_attr_prefix}_trend", "improving")
            elif current_rs < sma_val * 0.995:
                setattr(result, f"{target_attr_prefix}_trend", "deteriorating")
            else:
                setattr(result, f"{target_attr_prefix}_trend", "neutral")

    # RS momentum: 20-day rate of change of RS ratio
    if len(rs_series.dropna()) >= 21:
        rs_20d_ago = float(rs_series.iloc[-21])
        if rs_20d_ago > 0 and not np.isnan(rs_20d_ago):
            rs_momentum = (current_rs / rs_20d_ago) - 1.0 if current_rs is not None else None
            setattr(result, f"{target_attr_prefix}_momentum", rs_momentum)


def _compute_sector_vs_spy(
    sector_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    result: RelativeStrengthAnalysis,
):
    """Compute sector ETF performance relative to SPY."""
    sect_close = sector_df["close"].values
    spy_close = spy_df["close"].values

    min_len = min(len(sect_close), len(spy_close))
    if min_len < 30:
        return

    sect_close = sect_close[-min_len:]
    spy_close = spy_close[-min_len:]

    # Sector vs SPY RS ratio (normalized)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = (sect_close / sect_close[0]) / (spy_close / spy_close[0])
        rs = np.where(np.isfinite(rs), rs, np.nan)

    rs_series = pd.Series(rs)
    result.sector_vs_spy = float(rs_series.iloc[-1]) if not np.isnan(rs_series.iloc[-1]) else None

    # Sector vs SPY trend (20d SMA)
    if len(rs_series) >= 20:
        rs_sma = rs_series.rolling(20).mean()
        sma_val = float(rs_sma.iloc[-1]) if not np.isnan(rs_sma.iloc[-1]) else None
        if result.sector_vs_spy is not None and sma_val is not None:
            if result.sector_vs_spy > sma_val * 1.005:
                result.sector_vs_spy_trend = "improving"
            elif result.sector_vs_spy < sma_val * 0.995:
                result.sector_vs_spy_trend = "deteriorating"

    # Sector 20d return vs SPY 20d return
    if min_len >= 21:
        sect_ret = sect_close[-1] / sect_close[-21] - 1
        spy_ret = spy_close[-1] / spy_close[-21] - 1
        result.sector_momentum_20d = sect_ret - spy_ret


def _classify_rs_regime(result: RelativeStrengthAnalysis):
    """
    Classify relative strength regime.

    Leading: outperforming AND improving
    Improving: underperforming BUT getting better (early reversal signal)
    Lagging: underperforming AND getting worse
    Deteriorating: outperforming BUT starting to weaken
    """
    score = 0.0

    # RS vs sector contribution
    if result.rs_vs_sector_trend == "improving":
        score += 0.3
    elif result.rs_vs_sector_trend == "deteriorating":
        score -= 0.3

    if result.rs_vs_sector_momentum is not None:
        mom = max(-0.1, min(0.1, result.rs_vs_sector_momentum))  # Clamp
        score += mom * 3.0  # Scale to ~+/-0.3

    # Percentile contribution
    if result.rs_percentile is not None:
        pctl_score = (result.rs_percentile - 50) / 50  # -1 to +1
        score += pctl_score * 0.2

    # Sector context
    if result.sector_vs_spy_trend == "improving":
        score += 0.1
    elif result.sector_vs_spy_trend == "deteriorating":
        score -= 0.1

    result.rs_score = max(-1.0, min(1.0, score))

    # Classify regime
    is_outperforming = result.rs_percentile is not None and result.rs_percentile >= 60
    is_improving = result.rs_vs_sector_trend == "improving"

    if is_outperforming and is_improving:
        result.rs_regime = "leading"
    elif not is_outperforming and is_improving:
        result.rs_regime = "improving"
    elif is_outperforming and not is_improving:
        result.rs_regime = "deteriorating"
    else:
        result.rs_regime = "lagging"


def compute_universe_returns(dfs: dict[str, pd.DataFrame]) -> dict[str, Optional[float]]:
    """
    Compute 20-day returns for all symbols in the universe.
    Used for RS percentile ranking.
    """
    returns = {}
    for symbol, df in dfs.items():
        if df is not None and len(df) >= 21:
            ret = float(df["close"].iloc[-1] / df["close"].iloc[-21] - 1)
            returns[symbol] = ret
        else:
            returns[symbol] = None
    return returns
