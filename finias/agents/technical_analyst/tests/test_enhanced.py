"""
Tests for the Enhanced Signal Module.

Tests cover all 5 research-backed signals:
  1. ATR Normalization Context
  2. RSI(2) Pullback Detection
  3. 52-Week High Ratio
  4. Weekly Trend Context
  5. Price Acceleration
"""

import pytest
import numpy as np
import pandas as pd
from finias.agents.technical_analyst.computations.enhanced import (
    compute_enhanced_signals, EnhancedSignals, MEDIAN_ATR_RATIO,
)


def _make_uptrend_df(n=300, start=100.0, daily_return=0.002, seed=42):
    """Steady uptrend for testing."""
    rng = np.random.RandomState(seed)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + daily_return + rng.normal(0, 0.005)))
    prices = np.array(prices)
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.008,
        "low": prices * 0.992,
        "close": prices,
        "volume": rng.randint(1_000_000, 10_000_000, n),
    })


def _make_pullback_df(n=300, seed=42):
    """Uptrend with a sharp recent pullback for RSI(2) testing."""
    rng = np.random.RandomState(seed)
    prices = [100.0]
    for i in range(n - 1):
        if i < n - 5:
            prices.append(prices[-1] * (1 + 0.002 + rng.normal(0, 0.003)))
        else:
            # Sharp 3-day pullback at the end
            prices.append(prices[-1] * (1 - 0.02))
    prices = np.array(prices)
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": rng.randint(1_000_000, 10_000_000, n),
    })


def _make_high_vol_df(n=300, seed=42):
    """High volatility stock (like TSLA)."""
    rng = np.random.RandomState(seed)
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + 0.001 + rng.normal(0, 0.03)))
    prices = np.array(prices)
    return pd.DataFrame({
        "open": prices * 0.99,
        "high": prices * 1.03,
        "low": prices * 0.97,
        "close": prices,
        "volume": rng.randint(1_000_000, 10_000_000, n),
    })


def _make_accelerating_df(n=300, seed=42):
    """Price with accelerating (convex) trajectory."""
    t = np.arange(n, dtype=float)
    # Quadratic: price = 100 + 0.1*t + 0.001*t² (convex upward)
    prices = 100 + 0.1 * t + 0.001 * t * t
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, 0.3, n)
    prices = prices + noise
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": rng.randint(1_000_000, 10_000_000, n),
    })


class TestATRContext:
    """Test ATR normalization context."""

    def test_atr_ratio_computed(self):
        result = compute_enhanced_signals(_make_uptrend_df())
        assert result.atr_ratio is not None
        assert result.atr_ratio > 0

    def test_atr_ratio_percentile(self):
        result = compute_enhanced_signals(_make_uptrend_df())
        assert result.atr_ratio_percentile is not None
        assert 0 <= result.atr_ratio_percentile <= 100

    def test_high_vol_has_higher_scaling(self):
        low_vol = compute_enhanced_signals(_make_uptrend_df())
        high_vol = compute_enhanced_signals(_make_high_vol_df())
        assert high_vol.atr_scaling_factor > low_vol.atr_scaling_factor

    def test_scaling_clamped(self):
        result = compute_enhanced_signals(_make_uptrend_df())
        assert 0.5 <= result.atr_scaling_factor <= 2.0

    def test_scaling_clamped_high_vol(self):
        result = compute_enhanced_signals(_make_high_vol_df())
        assert result.atr_scaling_factor <= 2.0


class TestRSI2Pullback:
    """Test RSI(2) pullback detection."""

    def test_pullback_detected_on_sharp_decline(self):
        result = compute_enhanced_signals(_make_pullback_df())
        assert result.rsi_2 is not None
        # After 3 consecutive down days, RSI(2) should be very low
        assert result.rsi_2 < 20

    def test_no_pullback_in_steady_uptrend(self):
        result = compute_enhanced_signals(_make_uptrend_df())
        # Steady uptrend should NOT trigger pullback
        assert result.pullback_entry is False

    def test_rsi2_range(self):
        result = compute_enhanced_signals(_make_uptrend_df())
        if result.rsi_2 is not None:
            assert 0 <= result.rsi_2 <= 100


class TestHighRatio52W:
    """Test 52-week high ratio."""

    def test_uptrend_near_high(self):
        result = compute_enhanced_signals(_make_uptrend_df())
        assert result.high_ratio_52w is not None
        # Uptrend should be near 52-week high
        assert result.high_ratio_52w > 0.85
        assert result.high_nearness in ("at_high", "near_high")

    def test_ratio_between_0_and_1(self):
        result = compute_enhanced_signals(_make_uptrend_df())
        assert 0 < result.high_ratio_52w <= 1.0


class TestWeeklyTrend:
    """Test weekly trend context."""

    def test_uptrend_weekly_confirms(self):
        result = compute_enhanced_signals(
            _make_uptrend_df(), daily_trend_regime="uptrend"
        )
        assert result.weekly_trend_regime in ("uptrend", "consolidation")
        if result.weekly_trend_regime == "uptrend":
            assert result.weekly_confirms_daily is True

    def test_weekly_score_range(self):
        result = compute_enhanced_signals(_make_uptrend_df())
        assert -1.0 <= result.weekly_trend_score <= 1.0


class TestAcceleration:
    """Test price acceleration (quadratic fit)."""

    def test_accelerating_detected(self):
        result = compute_enhanced_signals(_make_accelerating_df())
        assert result.acceleration is not None
        assert result.acceleration > 0
        assert result.acceleration_regime == "accelerating"

    def test_acceleration_computed(self):
        result = compute_enhanced_signals(_make_uptrend_df())
        assert result.acceleration is not None


class TestSerialization:
    """Test to_dict serialization."""

    def test_to_dict_complete(self):
        result = compute_enhanced_signals(_make_uptrend_df(), symbol="TEST")
        d = result.to_dict()
        assert "atr_context" in d
        assert "rsi2" in d
        assert "high_52w" in d
        assert "weekly_trend" in d
        assert "acceleration" in d

    def test_json_serializable(self):
        import json
        result = compute_enhanced_signals(_make_uptrend_df())
        serialized = json.dumps(result.to_dict())
        assert len(serialized) > 50


class TestEdgeCases:
    """Test edge cases."""

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "close": [100, 101], "high": [102, 103],
            "low": [99, 100], "open": [100, 101], "volume": [1000, 1000],
        })
        result = compute_enhanced_signals(df)
        assert result.atr_ratio is None
        assert result.rsi_2 is None

    def test_none_dataframe(self):
        result = compute_enhanced_signals(None)
        assert result.atr_ratio is None
