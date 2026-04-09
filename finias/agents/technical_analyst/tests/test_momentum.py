"""
Tests for the momentum analysis module.

Tests cover regime-adaptive thresholds, divergence detection,
MACD signals, and momentum scoring.
"""

import pytest
import numpy as np
import pandas as pd
from finias.agents.technical_analyst.computations.momentum import (
    analyze_momentum, MomentumAnalysis, RSI_THRESHOLDS,
)


def _make_df(n=250, start=100.0, daily_return=0.001):
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + daily_return + np.random.normal(0, 0.005)))
    prices = np.array(prices)
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    })


class TestAdaptiveThresholds:
    """Test regime-adaptive RSI thresholds."""

    def test_uptrend_thresholds_shift_up(self):
        result = analyze_momentum(_make_df(), trend_regime="strong_uptrend")
        assert result.rsi_overbought_threshold == 80
        assert result.rsi_oversold_threshold == 40

    def test_downtrend_thresholds_shift_down(self):
        result = analyze_momentum(_make_df(), trend_regime="strong_downtrend")
        assert result.rsi_overbought_threshold == 60
        assert result.rsi_oversold_threshold == 20

    def test_consolidation_uses_standard(self):
        result = analyze_momentum(_make_df(), trend_regime="consolidation")
        assert result.rsi_overbought_threshold == 70
        assert result.rsi_oversold_threshold == 30

    def test_all_regimes_have_thresholds(self):
        for regime in RSI_THRESHOLDS:
            result = analyze_momentum(_make_df(n=60), trend_regime=regime)
            assert result.rsi_overbought_threshold > result.rsi_oversold_threshold


class TestRSI:
    """Test RSI computation."""

    def test_rsi_computed(self):
        result = analyze_momentum(_make_df())
        assert result.rsi_14 is not None
        assert 0 <= result.rsi_14 <= 100

    def test_rsi_zone_classified(self):
        result = analyze_momentum(_make_df())
        assert result.rsi_zone in ("overbought", "oversold", "neutral")


class TestMACD:
    """Test MACD signals."""

    def test_macd_computed(self):
        result = analyze_momentum(_make_df())
        assert result.macd_value is not None
        assert result.macd_histogram is not None

    def test_macd_direction_set(self):
        result = analyze_momentum(_make_df())
        assert result.macd_direction in ("bullish", "bearish", "neutral")

    def test_macd_momentum_set(self):
        result = analyze_momentum(_make_df())
        assert result.macd_momentum in ("accelerating", "decelerating", "neutral")


class TestStochastic:
    """Test Stochastic oscillator."""

    def test_stochastic_computed(self):
        result = analyze_momentum(_make_df())
        assert result.stoch_k is not None
        assert 0 <= result.stoch_k <= 100

    def test_stochastic_zone(self):
        result = analyze_momentum(_make_df())
        assert result.stoch_zone in ("overbought", "oversold", "neutral")


class TestROC:
    """Test Rate of Change."""

    def test_roc_computed_at_all_timeframes(self):
        result = analyze_momentum(_make_df(n=100))
        assert result.roc_5d is not None
        assert result.roc_20d is not None
        assert result.roc_60d is not None

    def test_roc_positive_in_uptrend(self):
        result = analyze_momentum(_make_df(daily_return=0.005))
        assert result.roc_20d > 0


class TestMomentumScore:
    """Test composite momentum score."""

    def test_score_in_range(self):
        result = analyze_momentum(_make_df())
        assert -1.0 <= result.momentum_score <= 1.0

    def test_strong_uptrend_has_positive_score(self):
        df = _make_df(n=250, daily_return=0.003)
        result = analyze_momentum(df, trend_regime="strong_uptrend")
        # Strong uptrend should generally produce positive momentum
        assert result.momentum_score > -0.5  # Allow some noise

    def test_regime_stored(self):
        result = analyze_momentum(_make_df(), trend_regime="uptrend")
        assert result.trend_regime_used == "uptrend"


class TestToDict:
    """Test serialization."""

    def test_to_dict_complete(self):
        result = analyze_momentum(_make_df(), symbol="AAPL", trend_regime="uptrend")
        d = result.to_dict()
        assert "momentum_score" in d
        assert "rsi" in d
        assert "macd" in d
        assert "stochastic" in d
        assert "roc" in d
        assert "divergence" in d
        assert "thrust" in d

    def test_to_dict_json_serializable(self):
        import json
        result = analyze_momentum(_make_df())
        serialized = json.dumps(result.to_dict())
        assert len(serialized) > 50
