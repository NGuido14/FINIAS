"""Tests for cached daily refresh infrastructure."""

import pytest
from datetime import datetime, timezone, timedelta


def test_morning_refresh_prompt_exists():
    """Verify the morning refresh prompt can be imported."""
    from finias.agents.macro_strategist.prompts.refresh import MORNING_REFRESH_PROMPT
    assert len(MORNING_REFRESH_PROMPT) > 500  # Should be comprehensive
    assert "{question}" in MORNING_REFRESH_PROMPT  # Must have placeholder
    assert "REGIME STATUS" in MORNING_REFRESH_PROMPT
    assert "BINDING CONSTRAINT" in MORNING_REFRESH_PROMPT
    assert "WEB SEARCH" in MORNING_REFRESH_PROMPT


def test_morning_refresh_prompt_format():
    """Verify the prompt can be formatted with empty and non-empty questions."""
    from finias.agents.macro_strategist.prompts.refresh import MORNING_REFRESH_PROMPT
    # Empty question (morning refresh)
    formatted = MORNING_REFRESH_PROMPT.format(question="")
    assert len(formatted) > 500
    # With question (ad-hoc)
    formatted = MORNING_REFRESH_PROMPT.format(question="What about oil?")
    assert "What about oil?" in formatted


def test_director_summary_exists():
    """Verify to_director_summary method exists on RegimeAssessment."""
    from finias.agents.macro_strategist.computations.regime import RegimeAssessment
    assert hasattr(RegimeAssessment, 'to_director_summary')


def test_director_summary_is_subset():
    """Director summary should be significantly smaller than full to_dict."""
    from finias.agents.macro_strategist.computations.regime import RegimeAssessment
    from finias.core.agents.models import MarketRegime
    import json

    # Create a minimal regime assessment
    regime = RegimeAssessment(
        primary_regime=MarketRegime.TRANSITION,
        cycle_phase="mid_cycle",
        liquidity_regime="ample",
        volatility_regime="elevated",
        inflation_regime="stable",
        composite_score=0.01,
        confidence=0.6,
        stress_index=0.18,
        binding_constraint="inflation",
        key_levels={"vix": 30.0, "fed_funds": 3.64},
        yield_curve={"spreads": {"2s10s": 0.56}},
        volatility={"vix": {"current": 30.0}},
        breadth={"breadth_health": "weakening", "breadth_score": 0.49, "sector_participation": {}, "sector_rotation": {}},
        cross_asset={"credit": {}, "oil": {}, "dollar": {}, "risk_appetite": {},
                     "stock_bond_correlation": {}, "correlations": {"aggregate": {}}},
        monetary_policy={"liquidity": {"net_liquidity_millions": 5782000}},
        business_cycle={"sahm_rule": {"value": 0.267}},
        inflation={"headline": {"core_pce_yoy": 3.06}},
        trajectory={
            "forward_bias": {"bias": "neutral"},
            "position_sizing": {"max_single_position_pct": 1.0},
            "velocity": {"urgency": "high"},
            "event_calendar": {},
            "scenario_triggers": [],
            "sector_guidance": {},
            "rate_decisions": {},
            "inflation_surprise": {},
            "trajectory_signals": {},
        },
    )

    full = json.dumps(regime.to_dict(), default=str)
    summary = json.dumps(regime.to_director_summary(), default=str)

    # Summary should be significantly smaller
    assert len(summary) < len(full)
    # But should contain essential fields
    summary_dict = regime.to_director_summary()
    assert "regime" in summary_dict
    assert "key_levels" in summary_dict
    assert "trajectory" in summary_dict
    assert "scores" in summary_dict
    assert "confidence" in summary_dict


def test_director_accepts_state_param():
    """Verify Director can be constructed with state parameter."""
    from finias.agents.director.agent import Director
    from finias.core.agents.registry import ToolRegistry

    registry = ToolRegistry()
    # Should work with None state (backward compatible)
    director = Director(registry=registry, state=None)
    assert director.state is None

    # Should work without state arg (backward compatible)
    director2 = Director(registry=registry)
    assert director2.state is None
