"""Tests for hierarchical regime detection computation (Phase 1)."""

import pytest
from finias.agents.macro_strategist.computations.regime import (
    detect_regime, _compute_growth_cycle_score, _compute_market_signals_score,
    _classify_primary_regime, _compute_stress_index, _identify_binding_constraint,
)
from finias.agents.macro_strategist.computations.yield_curve import YieldCurveAnalysis
from finias.agents.macro_strategist.computations.volatility import VolatilityAnalysis
from finias.agents.macro_strategist.computations.breadth import BreadthAnalysis
from finias.agents.macro_strategist.computations.cross_asset import CrossAssetAnalysis
from finias.agents.macro_strategist.computations.monetary_policy import MonetaryPolicyAnalysis
from finias.agents.macro_strategist.computations.business_cycle import BusinessCycleAnalysis
from finias.agents.macro_strategist.computations.inflation import InflationAnalysis
from finias.core.agents.models import MarketRegime


def _make_yc(shape="normal", recession_score=0.0):
    return YieldCurveAnalysis(
        t3m=5.0, t2y=4.5, t5y=4.2, t10y=4.0, t30y=4.3,
        spread_2s10s=-0.5 if shape == "inverted" else 0.5,
        spread_3m10y=-1.0 if shape == "inverted" else 0.3,
        spread_2s30s=0.2,
        spread_2s10s_change_30d=0.0,
        spread_2s10s_change_90d=0.0,
        is_2s10s_inverted=(shape == "inverted"),
        is_3m10y_inverted=(shape == "inverted"),
        inversion_depth_2s10s=-0.5 if shape == "inverted" else 0.5,
        days_inverted_2s10s=100 if shape == "inverted" else 0,
        curve_shape=shape,
        recession_signal_score=recession_score,
    )


def _make_vol(regime="normal", risk_score=0.3):
    return VolatilityAnalysis(
        vix_current=18.0,
        vix_percentile_1y=50.0,
        vix_change_1d=0.5,
        vix_change_5d=1.0,
        vix_change_20d=2.0,
        vix_sma_20=17.0,
        vix_is_elevated=False,
        vix_is_spike=False,
        realized_vol_20d=15.0,
        realized_vol_60d=14.0,
        iv_rv_spread=3.0,
        vol_regime=regime,
        vol_risk_score=risk_score,
    )


def _make_breadth(score=0.5):
    result = BreadthAnalysis()
    result.breadth_health = "data_unavailable"
    result.breadth_score = score
    return result


def _make_cross_asset(score=0.0):
    result = CrossAssetAnalysis()
    result.dxy_level = 104.0
    result.dxy_trend = "stable"
    result.dxy_change_30d = 0.5
    result.hy_spread = 3.5
    result.hy_spread_trend = "stable"
    result.hy_spread_change_30d = -0.1
    result.credit_stress = False
    result.breakeven_5y = 2.3
    result.breakeven_10y = 2.2
    result.inflation_expectations = "anchored"
    result.cross_asset_score = score
    return result


def test_regime_detection_risk_on():
    """Test that favorable conditions produce risk-on or positive regime."""
    yc = _make_yc(shape="normal", recession_score=0.0)
    vol = _make_vol(regime="low", risk_score=0.1)
    br = _make_breadth(score=0.8)
    ca = _make_cross_asset(score=0.3)

    result = detect_regime(yc, vol, br, ca)
    assert result.primary_regime in (MarketRegime.RISK_ON, MarketRegime.TRANSITION)
    assert result.confidence > 0.3
    assert result.composite_score > 0


def test_regime_detection_risk_off():
    """Test that stressed conditions produce risk-off or transition regime.

    Note: Without monetary/inflation data, their scores default to 0.0
    which dilutes the composite. With full Phase 1 data, stressed market
    signals + inverted curve produce clearer risk-off.
    """
    yc = _make_yc(shape="inverted", recession_score=0.7)
    vol = _make_vol(regime="elevated", risk_score=0.7)
    br = _make_breadth(score=0.2)
    ca = _make_cross_asset(score=-0.5)

    result = detect_regime(yc, vol, br, ca)
    # Without monetary/inflation data, composite is diluted — transition or risk-off
    assert result.primary_regime in (MarketRegime.RISK_OFF, MarketRegime.TRANSITION, MarketRegime.CRISIS)
    assert result.composite_score < 0  # Directionally bearish
    assert result.stress_index > 0.2


def test_regime_with_full_phase1_analyses():
    """Test regime detection with all Phase 1 analyses included."""
    yc = _make_yc(shape="normal", recession_score=0.1)
    vol = _make_vol(regime="normal", risk_score=0.3)
    br = _make_breadth(score=0.6)
    ca = _make_cross_asset(score=0.1)

    mp = MonetaryPolicyAnalysis(
        fed_funds_current=4.5,
        policy_stance="hawkish",
        policy_direction="on_hold",
        liquidity_regime="adequate",
        policy_score=-0.2,
        liquidity_score=0.1,
    )

    cycle = BusinessCycleAnalysis(
        cycle_phase="mid_cycle",
        phase_confidence=0.6,
        composite_leading=0.2,
        recession_probability=0.1,
        sahm_value=0.1,
        sahm_triggered=False,
    )

    infl = InflationAnalysis(
        core_pce_yoy=2.8,
        inflation_regime="stable",
        inflation_trend="decelerating",
        inflation_score=0.1,
    )

    result = detect_regime(
        yc, vol, br, ca,
        monetary_policy=mp,
        business_cycle=cycle,
        inflation_analysis=infl,
    )

    # With mid-cycle, adequate liquidity, and stable inflation — should be positive
    assert result.primary_regime in (MarketRegime.RISK_ON, MarketRegime.TRANSITION)
    assert result.cycle_phase == "mid_cycle"
    assert result.liquidity_regime == "adequate"
    assert result.inflation_regime == "stable"
    assert result.binding_constraint in (
        "growth_cycle", "monetary_liquidity", "inflation", "market_signals"
    )


def test_binding_constraint_identification():
    """Test that binding constraint identifies the most negative category."""
    weights = {"growth": 0.25, "monetary": 0.25, "inflation": 0.25, "market": 0.25}

    # Inflation is most negative
    binding = _identify_binding_constraint(0.3, 0.2, -0.5, 0.1, weights)
    assert binding == "inflation"

    # Market signals most negative
    binding = _identify_binding_constraint(0.1, 0.2, 0.0, -0.4, weights)
    assert binding == "market_signals"


def test_classify_primary_regime_crisis():
    """High stress should produce crisis regime."""
    vol = _make_vol(regime="extreme", risk_score=0.9)
    regime, confidence = _classify_primary_regime(composite=-0.5, stress=0.75, vol=vol)
    assert regime == MarketRegime.CRISIS


def test_classify_primary_regime_risk_on():
    """Positive composite above threshold should produce risk_on."""
    vol = _make_vol(regime="normal", risk_score=0.2)
    regime, confidence = _classify_primary_regime(composite=0.12, stress=0.1, vol=vol)
    assert regime == MarketRegime.RISK_ON


def test_classify_primary_regime_risk_off():
    """Negative composite below threshold should produce risk_off."""
    vol = _make_vol(regime="elevated", risk_score=0.5)
    regime, confidence = _classify_primary_regime(composite=-0.20, stress=0.3, vol=vol)
    assert regime == MarketRegime.RISK_OFF
