"""Tests for yield curve analysis computation."""

import pytest
from finias.agents.macro_strategist.computations.yield_curve import (
    analyze_yield_curve, _compute_recession_score
)


def _make_series(values):
    """Helper to create a series from a list of values."""
    return [{"date": f"2024-01-{i+1:02d}", "value": v} for i, v in enumerate(values)]


def test_normal_yield_curve():
    """Test analysis with a normal (upward sloping) yield curve."""
    result = analyze_yield_curve(
        yields_2y=_make_series([4.0, 4.1, 4.2]),
        yields_5y=_make_series([4.2, 4.3, 4.4]),
        yields_10y=_make_series([4.5, 4.6, 4.7]),
        yields_30y=_make_series([4.8, 4.9, 5.0]),
        yields_3m=_make_series([5.0, 5.0, 5.0]),
        fed_funds=_make_series([5.25, 5.25, 5.25]),
    )

    assert result.spread_2s10s == pytest.approx(0.5)
    assert result.is_2s10s_inverted is False
    assert result.curve_shape == "normal"


def test_inverted_yield_curve():
    """Test analysis with an inverted yield curve."""
    result = analyze_yield_curve(
        yields_2y=_make_series([5.0, 5.0, 5.0]),
        yields_5y=_make_series([4.5, 4.5, 4.5]),
        yields_10y=_make_series([4.0, 4.0, 4.0]),
        yields_30y=_make_series([4.2, 4.2, 4.2]),
        yields_3m=_make_series([5.3, 5.3, 5.3]),
        fed_funds=_make_series([5.25, 5.25, 5.25]),
    )

    assert result.spread_2s10s == pytest.approx(-1.0)
    assert result.is_2s10s_inverted is True
    assert result.curve_shape == "inverted"
    assert result.recession_signal_score > 0


def test_empty_series():
    """Test with empty data."""
    result = analyze_yield_curve(
        yields_2y=[], yields_5y=[], yields_10y=[],
        yields_30y=[], yields_3m=[], fed_funds=[],
    )
    assert result.spread_2s10s is None
    assert result.curve_shape == "unknown"


def test_recession_score_inverted():
    """Test recession scoring for inverted curve."""
    score = _compute_recession_score(-0.5, -0.3, 100, None)
    assert score > 0.3  # Inverted + sustained should give meaningful score
