"""
Support & Resistance Module (Key Price Levels)

Answers: "Where are the important price levels for entry, exit, and stop-loss?"

Key intelligence features:
  - Classic and Fibonacci pivot points from OHLC
  - Bollinger Band dynamic levels
  - Donchian channel boundaries (rolling high/low)
  - Key level clustering: when multiple methods identify the same zone,
    that zone becomes a high-confidence level
  - Distance metrics: current price distance to nearest support/resistance
  - Risk/reward ratio from current price

All computation is pure Python + pandas-ta. No API calls.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import pandas_ta as ta


@dataclass
class KeyLevel:
    """A single support or resistance level."""
    price: float
    level_type: str  # support, resistance
    source: str  # pivot, fibonacci, bollinger, donchian, cluster
    strength: float = 1.0  # 1.0 = single source, higher = multiple sources agree


@dataclass
class LevelsAnalysis:
    """Complete support/resistance analysis for a single symbol."""

    symbol: str = ""
    current_price: float = 0.0

    # Pivot Points
    pivot: Optional[float] = None
    pivot_r1: Optional[float] = None
    pivot_r2: Optional[float] = None
    pivot_r3: Optional[float] = None
    pivot_s1: Optional[float] = None
    pivot_s2: Optional[float] = None
    pivot_s3: Optional[float] = None

    # Fibonacci Pivots
    fib_r1: Optional[float] = None
    fib_r2: Optional[float] = None
    fib_r3: Optional[float] = None
    fib_s1: Optional[float] = None
    fib_s2: Optional[float] = None
    fib_s3: Optional[float] = None

    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_pct_b: Optional[float] = None  # 0-1, where price is within bands
    bb_bandwidth: Optional[float] = None

    # Donchian Channels
    donchian_20_high: Optional[float] = None
    donchian_20_low: Optional[float] = None
    donchian_50_high: Optional[float] = None
    donchian_50_low: Optional[float] = None

    # Key Levels (clustered from all sources)
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    nearest_support_distance_pct: Optional[float] = None
    nearest_resistance_distance_pct: Optional[float] = None
    risk_reward_ratio: Optional[float] = None  # distance_to_resistance / distance_to_support
    key_levels: list[KeyLevel] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "current_price": round(self.current_price, 2),
            "pivots": {
                "pivot": round(self.pivot, 2) if self.pivot else None,
                "r1": round(self.pivot_r1, 2) if self.pivot_r1 else None,
                "r2": round(self.pivot_r2, 2) if self.pivot_r2 else None,
                "r3": round(self.pivot_r3, 2) if self.pivot_r3 else None,
                "s1": round(self.pivot_s1, 2) if self.pivot_s1 else None,
                "s2": round(self.pivot_s2, 2) if self.pivot_s2 else None,
                "s3": round(self.pivot_s3, 2) if self.pivot_s3 else None,
            },
            "fibonacci": {
                "r1": round(self.fib_r1, 2) if self.fib_r1 else None,
                "r2": round(self.fib_r2, 2) if self.fib_r2 else None,
                "r3": round(self.fib_r3, 2) if self.fib_r3 else None,
                "s1": round(self.fib_s1, 2) if self.fib_s1 else None,
                "s2": round(self.fib_s2, 2) if self.fib_s2 else None,
                "s3": round(self.fib_s3, 2) if self.fib_s3 else None,
            },
            "bollinger": {
                "upper": round(self.bb_upper, 2) if self.bb_upper else None,
                "middle": round(self.bb_middle, 2) if self.bb_middle else None,
                "lower": round(self.bb_lower, 2) if self.bb_lower else None,
                "pct_b": round(self.bb_pct_b, 4) if self.bb_pct_b is not None else None,
                "bandwidth": round(self.bb_bandwidth, 4) if self.bb_bandwidth else None,
            },
            "donchian": {
                "20d_high": round(self.donchian_20_high, 2) if self.donchian_20_high else None,
                "20d_low": round(self.donchian_20_low, 2) if self.donchian_20_low else None,
                "50d_high": round(self.donchian_50_high, 2) if self.donchian_50_high else None,
                "50d_low": round(self.donchian_50_low, 2) if self.donchian_50_low else None,
            },
            "nearest_support": round(self.nearest_support, 2) if self.nearest_support else None,
            "nearest_resistance": round(self.nearest_resistance, 2) if self.nearest_resistance else None,
            "support_distance_pct": round(self.nearest_support_distance_pct, 2) if self.nearest_support_distance_pct is not None else None,
            "resistance_distance_pct": round(self.nearest_resistance_distance_pct, 2) if self.nearest_resistance_distance_pct is not None else None,
            "risk_reward_ratio": round(self.risk_reward_ratio, 2) if self.risk_reward_ratio else None,
            "key_levels_count": len(self.key_levels),
        }


def analyze_levels(df: pd.DataFrame, symbol: str = "") -> LevelsAnalysis:
    """
    Compute support/resistance levels from multiple methods.

    Args:
        df: OHLCV DataFrame, sorted chronologically.
        symbol: Ticker symbol.

    Returns:
        LevelsAnalysis with all level data and clustering.
    """
    result = LevelsAnalysis(symbol=symbol)

    if df is None or len(df) < 20:
        return result

    result.current_price = float(df["close"].iloc[-1])

    # === Classic Pivot Points ===
    _compute_pivots(df, result)

    # === Fibonacci Pivots ===
    _compute_fibonacci_pivots(df, result)

    # === Bollinger Bands ===
    _compute_bollinger(df, result)

    # === Donchian Channels ===
    _compute_donchian(df, result)

    # === Key Level Clustering ===
    _cluster_levels(result)

    return result


def _compute_pivots(df: pd.DataFrame, result: LevelsAnalysis):
    """Compute classic pivot points from the most recent complete day."""
    # Use the previous day's OHLC (not the current incomplete day)
    if len(df) < 2:
        return

    prev = df.iloc[-2]
    h = float(prev["high"])
    l = float(prev["low"])
    c = float(prev["close"])

    pp = (h + l + c) / 3
    result.pivot = pp
    result.pivot_r1 = 2 * pp - l
    result.pivot_r2 = pp + (h - l)
    result.pivot_r3 = h + 2 * (pp - l)
    result.pivot_s1 = 2 * pp - h
    result.pivot_s2 = pp - (h - l)
    result.pivot_s3 = l - 2 * (h - pp)


def _compute_fibonacci_pivots(df: pd.DataFrame, result: LevelsAnalysis):
    """Compute Fibonacci pivot levels."""
    if len(df) < 2:
        return

    prev = df.iloc[-2]
    h = float(prev["high"])
    l = float(prev["low"])
    c = float(prev["close"])
    r = h - l  # Range

    pp = (h + l + c) / 3
    result.fib_r1 = pp + 0.382 * r
    result.fib_r2 = pp + 0.618 * r
    result.fib_r3 = pp + 1.000 * r
    result.fib_s1 = pp - 0.382 * r
    result.fib_s2 = pp - 0.618 * r
    result.fib_s3 = pp - 1.000 * r


def _compute_bollinger(df: pd.DataFrame, result: LevelsAnalysis):
    """Compute Bollinger Bands."""
    bbands = ta.bbands(df["close"], length=20, std=2.0)
    if bbands is None or bbands.empty:
        return

    upper_col = [c for c in bbands.columns if c.startswith("BBU_")]
    mid_col = [c for c in bbands.columns if c.startswith("BBM_")]
    lower_col = [c for c in bbands.columns if c.startswith("BBL_")]
    pctb_col = [c for c in bbands.columns if c.startswith("BBP_")]
    bw_col = [c for c in bbands.columns if c.startswith("BBB_")]

    if upper_col:
        result.bb_upper = float(bbands[upper_col[0]].iloc[-1])
    if mid_col:
        result.bb_middle = float(bbands[mid_col[0]].iloc[-1])
    if lower_col:
        result.bb_lower = float(bbands[lower_col[0]].iloc[-1])
    if pctb_col:
        result.bb_pct_b = float(bbands[pctb_col[0]].iloc[-1])
    if bw_col:
        result.bb_bandwidth = float(bbands[bw_col[0]].iloc[-1])


def _compute_donchian(df: pd.DataFrame, result: LevelsAnalysis):
    """Compute Donchian channel boundaries."""
    if len(df) >= 20:
        result.donchian_20_high = float(df["high"].iloc[-20:].max())
        result.donchian_20_low = float(df["low"].iloc[-20:].min())
    if len(df) >= 50:
        result.donchian_50_high = float(df["high"].iloc[-50:].max())
        result.donchian_50_low = float(df["low"].iloc[-50:].min())


def _cluster_levels(result: LevelsAnalysis):
    """
    Identify key levels by clustering levels from multiple sources.

    When multiple methods (pivots, Fibonacci, Bollinger, Donchian) identify
    price levels within a narrow band (0.75% of price), those levels cluster
    into a high-confidence key level.
    """
    price = result.current_price
    if price <= 0:
        return

    cluster_threshold = price * 0.0075  # 0.75% of price

    # Collect all computed levels with their source
    all_levels = []

    # Pivot levels
    for level, source in [
        (result.pivot_r1, "pivot"), (result.pivot_r2, "pivot"), (result.pivot_r3, "pivot"),
        (result.pivot_s1, "pivot"), (result.pivot_s2, "pivot"), (result.pivot_s3, "pivot"),
        (result.pivot, "pivot"),
    ]:
        if level is not None:
            all_levels.append((level, source))

    # Fibonacci levels
    for level in [result.fib_r1, result.fib_r2, result.fib_r3,
                  result.fib_s1, result.fib_s2, result.fib_s3]:
        if level is not None:
            all_levels.append((level, "fibonacci"))

    # Bollinger levels
    for level in [result.bb_upper, result.bb_middle, result.bb_lower]:
        if level is not None:
            all_levels.append((level, "bollinger"))

    # Donchian levels
    for level in [result.donchian_20_high, result.donchian_20_low,
                  result.donchian_50_high, result.donchian_50_low]:
        if level is not None:
            all_levels.append((level, "donchian"))

    if not all_levels:
        return

    # Sort by price
    all_levels.sort(key=lambda x: x[0])

    # Cluster nearby levels
    clusters = []
    current_cluster = [all_levels[0]]

    for i in range(1, len(all_levels)):
        if all_levels[i][0] - current_cluster[-1][0] <= cluster_threshold:
            current_cluster.append(all_levels[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [all_levels[i]]
    clusters.append(current_cluster)

    # Build key levels from clusters
    supports = []
    resistances = []

    for cluster in clusters:
        avg_price = np.mean([l[0] for l in cluster])
        sources = set(l[1] for l in cluster)
        strength = len(sources)  # More independent sources = stronger level
        level_type = "support" if avg_price < price else "resistance"

        key_level = KeyLevel(
            price=float(avg_price),
            level_type=level_type,
            source="cluster" if strength > 1 else cluster[0][1],
            strength=float(strength),
        )
        result.key_levels.append(key_level)

        if level_type == "support":
            supports.append(key_level)
        else:
            resistances.append(key_level)

    # Find nearest support and resistance
    if supports:
        nearest_sup = max(supports, key=lambda l: l.price)
        result.nearest_support = nearest_sup.price
        result.nearest_support_distance_pct = ((price - nearest_sup.price) / price) * 100

    if resistances:
        nearest_res = min(resistances, key=lambda l: l.price)
        result.nearest_resistance = nearest_res.price
        result.nearest_resistance_distance_pct = ((nearest_res.price - price) / price) * 100

    # Risk/reward ratio
    if (result.nearest_support_distance_pct is not None
            and result.nearest_resistance_distance_pct is not None
            and result.nearest_support_distance_pct > 0):
        result.risk_reward_ratio = (
            result.nearest_resistance_distance_pct / result.nearest_support_distance_pct
        )
