"""Tests for volatility analysis computation."""

import pytest
from finias.agents.macro_strategist.computations.volatility import (
    analyze_volatility, _classify_vol_regime, _compute_vol_risk_score
)


def _make_vix_series(values):
    return [{"date": f"2024-01-{i+1:02d}", "value": v} for i, v in enumerate(values)]


def _make_prices(values):
    return [{"date": f"2024-01-{i+1:02d}", "close": v} for i, v in enumerate(values)]


def test_low_vol_regime():
    """Test classification of low volatility."""
    assert _classify_vol_regime(12.0, 10.0, False) == "low"


def test_normal_vol_regime():
    """Test classification of normal volatility."""
    assert _classify_vol_regime(18.0, 50.0, False) == "normal"


def test_elevated_vol_regime():
    """Test classification of elevated volatility."""
    assert _classify_vol_regime(28.0, 80.0, False) == "elevated"


def test_extreme_vol_regime():
    """Test classification of extreme volatility."""
    assert _classify_vol_regime(40.0, 95.0, True) == "extreme"


def test_vol_risk_score_low():
    """Low VIX should produce low risk score."""
    score = _compute_vol_risk_score(12.0, 10.0, False, 5.0)
    assert score < 0.2


def test_vol_risk_score_high():
    """High VIX with spike should produce high risk score."""
    score = _compute_vol_risk_score(38.0, 95.0, True, -2.0)
    assert score > 0.7
