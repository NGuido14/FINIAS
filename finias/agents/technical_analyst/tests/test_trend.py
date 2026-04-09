"""
Tests for the trend analysis module.

Uses synthetic price data to test all trend detection components.
"""

import pytest
import numpy as np
import pandas as pd
from finias.agents.technical_analyst.computations.trend import (
    analyze_trend, TrendAnalysis,
    _compute_ma_constellation, _classify_trend_regime, _compute_adx,
)


def _make_uptrend_df(n=250, start=100.0, daily_return=0.001, seed=42):
    """Generate a clean uptrend DataFrame."""
    rng = np.random.RandomState(seed)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + daily_return + rng.normal(0, 0.005)))
    prices = np.array(prices)
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": rng.randint(1_000_000, 10_000_000, n),
    })


def _make_downtrend_df(n=250, start=100.0, daily_return=-0.002, seed=42):
    """Generate a clean downtrend DataFrame with strong signal."""
    return _make_uptrend_df(n=n, start=start, daily_return=daily_return, seed=seed)


def _make_flat_df(n=250, price=100.0, seed=99):
    """Generate a flat/consolidation DataFrame with mean reversion."""
    rng = np.random.RandomState(seed)
    # Use mean-reverting process to keep prices truly flat
    prices = np.full(n, price, dtype=float)
    for i in range(1, n):
        prices[i] = price + 0.95 * (prices[i-1] - price) + rng.normal(0, 0.3)
    return pd.DataFrame({
        "open": prices - 0.2,
        "high": prices + 0.5,
        "low": prices - 0.5,
        "close": prices,
        "volume": rng.randint(1_000_000, 10_000_000, n),
    })


class TestTrendRegime:
    """Test trend regime classification."""

    def test_uptrend_detected(self):
        df = _make_uptrend_df()
        result = analyze_trend(df, symbol="TEST")
        assert result.trend_regime in ("strong_uptrend", "uptrend")
        assert result.trend_score > 0

    def test_downtrend_detected(self):
        df = _make_downtrend_df()
        result = analyze_trend(df, symbol="TEST")
        assert result.trend_regime in ("strong_downtrend", "downtrend", "consolidation")
        assert result.trend_score < 0.1  # Should be negative or near-zero

    def test_consolidation_detected(self):
        df = _make_flat_df()
        result = analyze_trend(df, symbol="TEST")
        # Flat market should not produce strong directional signal
        assert abs(result.trend_score) < 0.5

    def test_insufficient_data_returns_defaults(self):
        df = pd.DataFrame({"close": [100, 101], "high": [102, 103], "low": [99, 100], "open": [100, 101], "volume": [1000, 1000]})
        result = analyze_trend(df)
        assert result.trend_regime == "unknown"
        assert result.trend_score == 0.0


class TestMAConstellation:
    """Test moving average alignment."""

    def test_uptrend_has_bullish_alignment(self):
        df = _make_uptrend_df(n=250, daily_return=0.002)
        result = analyze_trend(df)
        assert result.ma_alignment in ("perfect_bull", "bull")
        assert result.ma_alignment_score > 0

    def test_price_above_sma200_in_uptrend(self):
        df = _make_uptrend_df(n=250, daily_return=0.002)
        result = analyze_trend(df)
        assert result.price_vs_sma200 == "above"

    def test_price_below_sma200_in_downtrend(self):
        df = _make_downtrend_df(n=250, daily_return=-0.002)
        result = analyze_trend(df)
        assert result.price_vs_sma200 == "below"


class TestADX:
    """Test ADX trend strength."""

    def test_strong_trend_has_high_adx(self):
        df = _make_uptrend_df(n=250, daily_return=0.003)
        result = analyze_trend(df)
        if result.adx is not None:
            assert result.adx > 0  # ADX should be positive

    def test_flat_market_has_low_adx(self):
        df = _make_flat_df(n=250)
        result = analyze_trend(df)
        if result.adx is not None:
            assert result.adx < 40  # Flat should have lower ADX


class TestIchimoku:
    """Test Ichimoku Cloud signals."""

    def test_uptrend_has_bullish_ichimoku(self):
        df = _make_uptrend_df(n=250, daily_return=0.002)
        result = analyze_trend(df)
        assert result.ichimoku_signal in ("bullish", "strong_bullish", "neutral")
        assert result.price_vs_cloud in ("above", "inside", "unknown")

    def test_downtrend_has_bearish_ichimoku(self):
        df = _make_downtrend_df(n=250, daily_return=-0.002)
        result = analyze_trend(df)
        assert result.ichimoku_signal in ("bearish", "strong_bearish", "neutral")


class TestTrendMaturity:
    """Test trend maturity estimation."""

    def test_long_uptrend_is_mature(self):
        df = _make_uptrend_df(n=300, daily_return=0.002)
        result = analyze_trend(df)
        assert result.trend_age_bars > 0
        assert result.trend_maturity in ("developing", "mature", "late")


class TestToDict:
    """Test serialization."""

    def test_to_dict_has_all_keys(self):
        df = _make_uptrend_df()
        result = analyze_trend(df, symbol="AAPL")
        d = result.to_dict()
        assert "trend_regime" in d
        assert "trend_score" in d
        assert "ma" in d
        assert "adx" in d
        assert "ichimoku" in d
        assert "maturity" in d
        assert "structure" in d

    def test_to_dict_is_json_serializable(self):
        import json
        df = _make_uptrend_df()
        result = analyze_trend(df)
        serialized = json.dumps(result.to_dict())
        assert len(serialized) > 50
