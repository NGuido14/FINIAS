"""Tests for the volume analysis module."""

import pytest
import numpy as np
import pandas as pd
from finias.agents.technical_analyst.computations.volume import (
    analyze_volume, VolumeAnalysis,
)


def _make_df(n=250, start=100.0, daily_return=0.001, volume_trend=0):
    """Generate test DataFrame with optional volume trend."""
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + daily_return + np.random.normal(0, 0.005)))
    prices = np.array(prices)
    base_vol = 5_000_000
    volumes = base_vol + np.arange(n) * volume_trend + np.random.randint(-500_000, 500_000, n)
    volumes = np.maximum(volumes, 100_000)
    return pd.DataFrame({
        "open": prices * 0.999, "high": prices * 1.005,
        "low": prices * 0.995, "close": prices,
        "volume": volumes.astype(int),
    })


class TestOBV:
    def test_obv_computed(self):
        result = analyze_volume(_make_df())
        assert result.obv_trend in ("rising", "falling", "neutral")

    def test_obv_slope_exists(self):
        result = analyze_volume(_make_df())
        assert result.obv_slope_20d is not None


class TestRelativeVolume:
    def test_relative_volume_computed(self):
        result = analyze_volume(_make_df())
        assert result.relative_volume is not None
        assert result.relative_volume > 0

    def test_volume_zone_classified(self):
        result = analyze_volume(_make_df())
        assert result.volume_zone in ("high", "low", "normal")


class TestMFI:
    def test_mfi_computed(self):
        result = analyze_volume(_make_df())
        assert result.mfi_14 is not None
        assert 0 <= result.mfi_14 <= 100


class TestConfirmation:
    def test_score_in_range(self):
        result = analyze_volume(_make_df(), trend_regime="uptrend")
        assert -1.0 <= result.volume_confirmation_score <= 1.0

    def test_regime_stored(self):
        result = analyze_volume(_make_df(), trend_regime="downtrend")
        assert result.trend_regime_used == "downtrend"


class TestSerialization:
    def test_to_dict_complete(self):
        result = analyze_volume(_make_df(), symbol="TEST")
        d = result.to_dict()
        assert "volume_confirmation_score" in d
        assert "obv" in d
        assert "relative_volume" in d
        assert "mfi" in d

    def test_json_serializable(self):
        import json
        result = analyze_volume(_make_df())
        json.dumps(result.to_dict())
