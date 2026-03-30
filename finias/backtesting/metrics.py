"""
Backtest validation metrics.

Computes statistical measures of signal quality:
- Correlation between signal and forward returns
- Directional hit rate
- Strategy Sharpe ratio
- Maximum drawdown
"""

from __future__ import annotations
from typing import Optional
import numpy as np


def correlation(signal: list[float], returns: list[float]) -> Optional[float]:
    """Pearson correlation between signal values and forward returns."""
    if len(signal) < 10 or len(signal) != len(returns):
        return None
    sig = np.array(signal)
    ret = np.array(returns)
    # Remove NaN pairs
    mask = ~(np.isnan(sig) | np.isnan(ret))
    if mask.sum() < 10:
        return None
    corr = np.corrcoef(sig[mask], ret[mask])[0, 1]
    return float(corr) if not np.isnan(corr) else None


def directional_hit_rate(
    signal: list[float],
    returns: list[float],
    signal_threshold: float = 0.0,
) -> dict:
    """
    When signal is above/below threshold, how often do returns agree?

    Returns dict with bullish_hits, bearish_hits, counts.
    """
    sig = np.array(signal)
    ret = np.array(returns)
    mask = ~(np.isnan(sig) | np.isnan(ret))
    sig, ret = sig[mask], ret[mask]

    bullish_mask = sig > signal_threshold
    bearish_mask = sig < -signal_threshold

    result = {}

    if bullish_mask.sum() > 0:
        bullish_rets = ret[bullish_mask]
        result["bullish_count"] = int(bullish_mask.sum())
        result["bullish_avg_return"] = float(np.mean(bullish_rets))
        result["bullish_hit_rate"] = float(np.mean(bullish_rets > 0))
    else:
        result["bullish_count"] = 0

    if bearish_mask.sum() > 0:
        bearish_rets = ret[bearish_mask]
        result["bearish_count"] = int(bearish_mask.sum())
        result["bearish_avg_return"] = float(np.mean(bearish_rets))
        result["bearish_hit_rate"] = float(np.mean(bearish_rets < 0))
    else:
        result["bearish_count"] = 0

    neutral_mask = ~bullish_mask & ~bearish_mask
    if neutral_mask.sum() > 0:
        result["neutral_count"] = int(neutral_mask.sum())
        result["neutral_avg_return"] = float(np.mean(ret[neutral_mask]))
    else:
        result["neutral_count"] = 0

    return result


def strategy_performance(
    signal: list[float],
    returns: list[float],
    bull_threshold: float = 0.15,
    bear_threshold: float = -0.15,
    transaction_cost_bps: float = 10,
) -> dict:
    """
    Simple long/cash strategy performance.

    Rules:
    - Composite > bull_threshold: 100% long SPY
    - Composite between bear and bull: 50% long
    - Composite < bear_threshold: 100% cash

    Assumes 1-period execution lag (signal on Friday, execute Monday).
    """
    sig = np.array(signal)
    ret = np.array(returns)

    # 1-period execution lag
    positions = np.zeros(len(sig))
    for i in range(1, len(sig)):
        if sig[i - 1] > bull_threshold:
            positions[i] = 1.0
        elif sig[i - 1] < bear_threshold:
            positions[i] = 0.0
        else:
            positions[i] = 0.5

    # Strategy returns with transaction costs
    position_changes = np.abs(np.diff(positions, prepend=positions[0]))
    costs = position_changes * transaction_cost_bps / 10000
    strategy_returns = positions * ret / 100 - costs  # ret is in percent

    # Buy and hold returns
    bnh_returns = ret / 100

    # Remove NaN
    mask = ~np.isnan(strategy_returns) & ~np.isnan(bnh_returns)
    strat = strategy_returns[mask]
    bnh = bnh_returns[mask]

    if len(strat) < 10:
        return {"error": "insufficient data"}

    # Annualize (assuming weekly observations)
    periods_per_year = 52

    strat_total = float(np.prod(1 + strat) - 1)
    bnh_total = float(np.prod(1 + bnh) - 1)

    strat_ann = float((1 + strat_total) ** (periods_per_year / len(strat)) - 1) if len(strat) > 0 else 0
    bnh_ann = float((1 + bnh_total) ** (periods_per_year / len(bnh)) - 1) if len(bnh) > 0 else 0

    strat_vol = float(np.std(strat) * np.sqrt(periods_per_year))
    bnh_vol = float(np.std(bnh) * np.sqrt(periods_per_year))

    strat_sharpe = strat_ann / strat_vol if strat_vol > 0 else 0
    bnh_sharpe = bnh_ann / bnh_vol if bnh_vol > 0 else 0

    # Max drawdown
    strat_cumulative = np.cumprod(1 + strat)
    strat_peak = np.maximum.accumulate(strat_cumulative)
    strat_dd = float(np.min(strat_cumulative / strat_peak - 1))

    bnh_cumulative = np.cumprod(1 + bnh)
    bnh_peak = np.maximum.accumulate(bnh_cumulative)
    bnh_dd = float(np.min(bnh_cumulative / bnh_peak - 1))

    return {
        "strategy_total_return": strat_total,
        "buyhold_total_return": bnh_total,
        "strategy_annual_return": strat_ann,
        "buyhold_annual_return": bnh_ann,
        "strategy_sharpe": strat_sharpe,
        "buyhold_sharpe": bnh_sharpe,
        "strategy_max_drawdown": strat_dd,
        "buyhold_max_drawdown": bnh_dd,
        "strategy_volatility": strat_vol,
        "buyhold_volatility": bnh_vol,
        "total_trades": int(np.sum(position_changes > 0)),
        "pct_time_fully_invested": float(np.mean(positions == 1.0)),
        "pct_time_cash": float(np.mean(positions == 0.0)),
        "observations": len(strat),
    }
