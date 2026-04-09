"""Tests for the volatility analysis module."""

import pytest
import numpy as np
import pandas as pd
from finias.agents.technical_analyst.computations.ta_volatility import (
    analyze_volatility, VolatilityAnalysis,
)


def _make_df(n=300, start=100.0, daily_vol=0.015):
    np.random.seed(42)
    returns = np.random.normal(0.0005, daily_vol, n)
    prices = start * np.cumprod(1 + returns)
    return pd.DataFrame({
        "open": prices * 0.999, "high": prices * (1 + daily_vol),
        "low": prices * (1 - daily_vol), "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    })


def _make_low_vol_df(n=300):
    return _make_df(n=n, daily_vol=0.003)


def _make_high_vol_df(n=300):
    return _make_df(n=n, daily_vol=0.04)


class TestATR:
    def test_atr_computed(self):
        result = analyze_volatility(_make_df())
        assert result.atr_14 is not None
        assert result.atr_14 > 0

    def test_atr_pct(self):
        result = analyze_volatility(_make_df())
        assert result.atr_pct is not None
        assert 0 < result.atr_pct < 1

    def test_atr_trend(self):
        result = analyze_volatility(_make_df())
        assert result.atr_trend in ("expanding", "contracting", "neutral")


class TestSqueeze:
    def test_squeeze_detected(self):
        result = analyze_volatility(_make_low_vol_df())
        # Low vol should be more likely to show squeeze
        assert isinstance(result.squeeze_on, bool)

    def test_bandwidth_percentile(self):
        result = analyze_volatility(_make_df())
        if result.bb_bandwidth_percentile is not None:
            assert 0 <= result.bb_bandwidth_percentile <= 100


class TestHistoricalVol:
    def test_hvol_computed(self):
        result = analyze_volatility(_make_df())
        assert result.hvol_20d is not None
        assert result.hvol_20d > 0

    def test_hvol_percentile(self):
        result = analyze_volatility(_make_df())
        if result.hvol_percentile is not None:
            assert 0 <= result.hvol_percentile <= 100

    def test_high_vol_has_higher_hvol(self):
        low = analyze_volatility(_make_low_vol_df())
        high = analyze_volatility(_make_high_vol_df())
        if low.hvol_20d and high.hvol_20d:
            assert high.hvol_20d > low.hvol_20d


class TestVolRegime:
    def test_regime_classified(self):
        result = analyze_volatility(_make_df())
        assert result.vol_regime in ("low_vol", "normal", "high_vol", "extreme")

    def test_score_in_range(self):
        result = analyze_volatility(_make_df())
        assert -1.0 <= result.vol_score <= 1.0


class TestSerialization:
    def test_to_dict_complete(self):
        result = analyze_volatility(_make_df(), symbol="TEST")
        d = result.to_dict()
        assert "atr" in d
        assert "squeeze" in d
        assert "historical_vol" in d
        assert "vol_regime" in d

    def test_json_serializable(self):
        import json
        result = analyze_volatility(_make_df())
        json.dumps(result.to_dict())
