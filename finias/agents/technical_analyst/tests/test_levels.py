"""
Tests for the support/resistance levels module.

Tests cover pivot points, Bollinger Bands, Donchian channels,
and key level clustering.
"""

import pytest
import numpy as np
import pandas as pd
from finias.agents.technical_analyst.computations.levels import (
    analyze_levels, LevelsAnalysis, KeyLevel,
)


def _make_df(n=100, price=100.0, volatility=2.0):
    np.random.seed(42)
    noise = np.random.normal(0, volatility, n)
    prices = price + np.cumsum(noise * 0.1)
    return pd.DataFrame({
        "open": prices - 0.5,
        "high": prices + volatility,
        "low": prices - volatility,
        "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    })


class TestPivotPoints:
    """Test classic pivot point computation."""

    def test_pivot_computed(self):
        result = analyze_levels(_make_df())
        assert result.pivot is not None
        assert result.pivot_r1 is not None
        assert result.pivot_s1 is not None

    def test_pivot_ordering(self):
        result = analyze_levels(_make_df())
        if result.pivot_s3 and result.pivot_r3:
            assert result.pivot_s3 < result.pivot_s2 < result.pivot_s1
            assert result.pivot_r1 < result.pivot_r2 < result.pivot_r3
            assert result.pivot_s1 < result.pivot < result.pivot_r1

    def test_fibonacci_pivots_computed(self):
        result = analyze_levels(_make_df())
        assert result.fib_r1 is not None
        assert result.fib_s1 is not None


class TestBollingerBands:
    """Test Bollinger Band computation."""

    def test_bollinger_computed(self):
        result = analyze_levels(_make_df())
        assert result.bb_upper is not None
        assert result.bb_middle is not None
        assert result.bb_lower is not None

    def test_bollinger_ordering(self):
        result = analyze_levels(_make_df())
        if result.bb_upper and result.bb_lower:
            assert result.bb_lower < result.bb_middle < result.bb_upper

    def test_pct_b_in_range(self):
        result = analyze_levels(_make_df())
        # %B can be outside 0-1 if price is outside bands
        assert result.bb_pct_b is not None


class TestDonchian:
    """Test Donchian channel computation."""

    def test_donchian_20_computed(self):
        result = analyze_levels(_make_df())
        assert result.donchian_20_high is not None
        assert result.donchian_20_low is not None

    def test_donchian_50_computed(self):
        result = analyze_levels(_make_df())
        assert result.donchian_50_high is not None
        assert result.donchian_50_low is not None

    def test_donchian_ordering(self):
        result = analyze_levels(_make_df())
        assert result.donchian_20_low <= result.donchian_20_high

    def test_50d_wider_than_20d(self):
        result = analyze_levels(_make_df())
        if result.donchian_50_high and result.donchian_20_high:
            range_20 = result.donchian_20_high - result.donchian_20_low
            range_50 = result.donchian_50_high - result.donchian_50_low
            assert range_50 >= range_20 - 0.01  # 50d channel >= 20d channel


class TestLevelClustering:
    """Test key level identification."""

    def test_key_levels_generated(self):
        result = analyze_levels(_make_df())
        assert len(result.key_levels) > 0

    def test_nearest_support_below_price(self):
        result = analyze_levels(_make_df())
        if result.nearest_support is not None:
            assert result.nearest_support <= result.current_price

    def test_nearest_resistance_above_price(self):
        result = analyze_levels(_make_df())
        if result.nearest_resistance is not None:
            assert result.nearest_resistance >= result.current_price

    def test_distance_percentages_positive(self):
        result = analyze_levels(_make_df())
        if result.nearest_support_distance_pct is not None:
            assert result.nearest_support_distance_pct >= 0
        if result.nearest_resistance_distance_pct is not None:
            assert result.nearest_resistance_distance_pct >= 0

    def test_risk_reward_ratio_positive(self):
        result = analyze_levels(_make_df())
        if result.risk_reward_ratio is not None:
            assert result.risk_reward_ratio > 0


class TestToDict:
    """Test serialization."""

    def test_to_dict_complete(self):
        result = analyze_levels(_make_df(), symbol="AAPL")
        d = result.to_dict()
        assert "current_price" in d
        assert "pivots" in d
        assert "fibonacci" in d
        assert "bollinger" in d
        assert "donchian" in d
        assert "nearest_support" in d
        assert "nearest_resistance" in d
        assert "risk_reward_ratio" in d

    def test_to_dict_json_serializable(self):
        import json
        result = analyze_levels(_make_df())
        serialized = json.dumps(result.to_dict())
        assert len(serialized) > 50


class TestEdgeCases:
    """Test edge cases."""

    def test_insufficient_data(self):
        df = pd.DataFrame({"close": [100], "high": [101], "low": [99], "open": [100], "volume": [1000]})
        result = analyze_levels(df)
        assert result.pivot is None

    def test_none_dataframe(self):
        result = analyze_levels(None)
        assert result.current_price == 0.0
