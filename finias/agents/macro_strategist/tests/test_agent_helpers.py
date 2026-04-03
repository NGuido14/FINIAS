"""Tests for agent-level helper functions."""

import pytest


def test_strip_cite_tags():
    from finias.agents.macro_strategist.agent import _strip_cite_tags
    assert _strip_cite_tags('Oil prices <cite index="3-17">surged 36%</cite> this month') == 'Oil prices surged 36% this month'
    assert _strip_cite_tags('No tags here') == 'No tags here'
    assert _strip_cite_tags('<cite index="1-2">Full tag</cite>') == 'Full tag'
    assert _strip_cite_tags('') == ''
    assert _strip_cite_tags(None) is None


def test_extract_interpretation_json():
    from finias.agents.macro_strategist.agent import _extract_interpretation_json
    import json

    # Simple case
    text = '{"summary": "test", "key_findings": ["a"], "risks": ["b"], "watch_items": ["c"]}'
    result = _extract_interpretation_json(text)
    assert result["summary"] == "test"

    # JSON surrounded by web search text
    text = 'Web search found this. Here is the analysis: {"summary": "test", "key_findings": ["a"], "risks": [], "watch_items": []} End of response.'
    result = _extract_interpretation_json(text)
    assert result["summary"] == "test"

    # No JSON at all
    result = _extract_interpretation_json("Just plain text with no JSON")
    assert result == {}


def test_extract_interpretation_json_raw_json_text():
    """When full text is valid JSON, it should be extracted."""
    from finias.agents.macro_strategist.agent import _extract_interpretation_json
    text = '{"macro_regime": "transition", "summary": "test summary", "key_findings": ["a"], "risks": ["b"], "watch_items": ["c"], "key_metrics": {"vix": 30.0}}'
    result = _extract_interpretation_json(text)
    assert result.get("summary") == "test summary"
    assert result.get("key_metrics", {}).get("vix") == 30.0


def test_extract_interpretation_json_web_search_with_stray_braces():
    """Web search content with stray braces should not break extraction."""
    from finias.agents.macro_strategist.agent import _extract_interpretation_json
    import json

    interpretation = {
        "macro_regime": "Cautious Transition",
        "binding_constraint": "Inflation persistence",
        "summary": "Core PCE remains elevated.",
        "key_findings": ["Yield curve inverted"],
        "risks": ["Sticky inflation"],
        "watch_items": ["VIX at 28"],
        "key_metrics": {"vix": 28.0, "core_pce_yoy": 3.1}
    }

    # Simulate web search noise with unmatched/stray braces
    web_noise = (
        'Based on my web search, I found {partial data here and also '
        'some JSON-LD like {"@context": "https://schema.org"} and '
        'a JS snippet: function() { return x; } — now here is my analysis:\n\n'
    )
    text = web_noise + json.dumps(interpretation)
    result = _extract_interpretation_json(text)
    assert result.get("macro_regime") == "Cautious Transition"
    assert result.get("summary") == "Core PCE remains elevated."
    assert result.get("key_metrics", {}).get("vix") == 28.0


def test_extract_interpretation_json_json_with_trailing_text():
    """JSON followed by trailing explanation text should still extract."""
    from finias.agents.macro_strategist.agent import _extract_interpretation_json
    import json

    interpretation = {
        "macro_regime": "Risk-On",
        "summary": "Broad expansion underway.",
        "key_findings": ["Growth strong"],
        "risks": ["Overheating"],
        "watch_items": ["Oil above 90"],
        "key_metrics": {"vix": 15.0}
    }
    text = json.dumps(interpretation) + "\n\nI hope this analysis is helpful. Let me know if you need anything else."
    result = _extract_interpretation_json(text)
    assert result.get("macro_regime") == "Risk-On"
    assert result.get("summary") == "Broad expansion underway."


def test_structuring_prompt_exists():
    """Verify the structuring prompt can be imported."""
    from finias.agents.macro_strategist.prompts.interpretation import (
        MACRO_ANALYSIS_PROMPT,
        MACRO_STRUCTURING_PROMPT,
    )
    assert "regime_data" in MACRO_ANALYSIS_PROMPT or "regime_data" in str(MACRO_ANALYSIS_PROMPT)
    assert "{analysis_text}" in MACRO_STRUCTURING_PROMPT
    # Analysis prompt should NOT require JSON output
    assert "Respond with ONLY a JSON" not in MACRO_ANALYSIS_PROMPT
    # Structuring prompt SHOULD require JSON output
    assert "JSON" in MACRO_STRUCTURING_PROMPT


def test_macro_analysis_prompt_has_temporal_guidance():
    """Analysis prompt should include temporal and continuity guidance."""
    from finias.agents.macro_strategist.prompts.interpretation import MACRO_ANALYSIS_PROMPT
    assert "TEMPORAL CONTEXT" in MACRO_ANALYSIS_PROMPT
    assert "PRIOR ASSESSMENT CONTINUITY" in MACRO_ANALYSIS_PROMPT
    assert "inflection points" in MACRO_ANALYSIS_PROMPT
    assert "materialized" in MACRO_ANALYSIS_PROMPT


class TestInterpretationValidation:
    """Tests for post-hoc interpretation validation."""

    def _make_regime(self):
        """Create a minimal regime assessment for testing."""
        from finias.agents.macro_strategist.computations.regime import RegimeAssessment
        from finias.core.agents.models import MarketRegime
        return RegimeAssessment(
            primary_regime=MarketRegime.TRANSITION,
            composite_score=-0.023,
            binding_constraint="inflation",
            key_levels={
                "vix": 25.25,
                "core_pce_yoy": 3.06,
                "core_pce_3m_ann": 3.66,
                "hy_spread": 3.28,
                "fed_funds": 3.64,
                "sahm_value": 0.267,
                "net_liquidity": 5782000,
            },
            trajectory={
                "forward_bias": {"bias": "neutral", "score": 0.0, "confidence": "moderate"},
            },
        )

    def test_validation_corrects_binding_constraint(self):
        """Fabricated binding constraint should be auto-corrected."""
        from finias.agents.macro_strategist.agent import MacroStrategist

        regime = self._make_regime()
        interp = {
            "summary": "Test summary",
            "key_findings": ["a"],
            "risks": ["b"],
            "watch_items": ["c"],
            "macro_regime": "Transition",
            "binding_constraint": "market signals",  # WRONG — should be "inflation"
            "key_metrics": {"forward_bias": "neutral", "composite_score": -0.023},
        }

        # Create a minimal MacroStrategist-like object to call the method
        class MockAgent:
            _validate_interpretation = MacroStrategist._validate_interpretation

        agent = MockAgent()
        result = agent._validate_interpretation(interp, regime)

        assert result["binding_constraint"] == "inflation"
        assert result["_validation"]["corrected"] > 0
        assert any(c["field"] == "binding_constraint" for c in result["_validation"]["corrections"])

    def test_validation_passes_correct_binding(self):
        """Correct binding constraint should pass."""
        from finias.agents.macro_strategist.agent import MacroStrategist

        regime = self._make_regime()
        interp = {
            "summary": "Test",
            "key_findings": [],
            "risks": [],
            "watch_items": [],
            "macro_regime": "Transition",
            "binding_constraint": "Inflation persistence",  # Contains "inflation" — passes
            "key_metrics": {"forward_bias": "neutral", "composite_score": -0.023},
        }

        class MockAgent:
            _validate_interpretation = MacroStrategist._validate_interpretation

        agent = MockAgent()
        result = agent._validate_interpretation(interp, regime)

        # Should NOT be corrected — "Inflation persistence" contains "inflation"
        assert not any(c["field"] == "binding_constraint" for c in result["_validation"]["corrections"])

    def test_validation_corrects_forward_bias(self):
        """Wrong forward_bias should be auto-corrected."""
        from finias.agents.macro_strategist.agent import MacroStrategist

        regime = self._make_regime()
        interp = {
            "summary": "Test",
            "key_findings": [],
            "risks": [],
            "watch_items": [],
            "macro_regime": "Transition",
            "binding_constraint": "inflation",
            "key_metrics": {"forward_bias": "constructive", "composite_score": -0.023},  # WRONG
        }

        class MockAgent:
            _validate_interpretation = MacroStrategist._validate_interpretation

        agent = MockAgent()
        result = agent._validate_interpretation(interp, regime)

        assert result["key_metrics"]["forward_bias"] == "neutral"
        assert any(c["field"] == "key_metrics.forward_bias" for c in result["_validation"]["corrections"])

    def test_validation_flags_wrong_vix(self):
        """VIX that doesn't match FRED or live should be flagged."""
        from finias.agents.macro_strategist.agent import MacroStrategist

        regime = self._make_regime()
        interp = {
            "summary": "Test",
            "key_findings": [],
            "risks": [],
            "watch_items": [],
            "macro_regime": "Transition",
            "binding_constraint": "inflation",
            "key_metrics": {"vix": 35.0},  # Way off from 25.25
        }

        class MockAgent:
            _validate_interpretation = MacroStrategist._validate_interpretation

        agent = MockAgent()
        result = agent._validate_interpretation(interp, regime, live_prices={"vix": 23.87})

        assert any(w["field"] == "key_metrics.vix" for w in result["_validation"]["warnings"])

    def test_validation_accepts_live_vix(self):
        """VIX matching the live price should pass even if FRED differs."""
        from finias.agents.macro_strategist.agent import MacroStrategist

        regime = self._make_regime()  # FRED VIX = 25.25
        interp = {
            "summary": "Test",
            "key_findings": [],
            "risks": [],
            "watch_items": [],
            "macro_regime": "Transition",
            "binding_constraint": "inflation",
            "key_metrics": {"vix": 24.0},  # Close to live 23.87, not FRED 25.25
        }

        class MockAgent:
            _validate_interpretation = MacroStrategist._validate_interpretation

        agent = MockAgent()
        result = agent._validate_interpretation(interp, regime, live_prices={"vix": 23.87})

        # Should pass — 24.0 is within tolerance of live 23.87
        assert not any(w["field"] == "key_metrics.vix" for w in result["_validation"]["warnings"])

    def test_validation_produces_audit_trail(self):
        """Validation should always produce a _validation field."""
        from finias.agents.macro_strategist.agent import MacroStrategist

        regime = self._make_regime()
        interp = {
            "summary": "Test",
            "key_findings": [],
            "risks": [],
            "watch_items": [],
            "macro_regime": "Transition",
            "binding_constraint": "inflation",
            "key_metrics": {},
        }

        class MockAgent:
            _validate_interpretation = MacroStrategist._validate_interpretation

        agent = MockAgent()
        result = agent._validate_interpretation(interp, regime)

        assert "_validation" in result
        assert "validated_at" in result["_validation"]
        assert "corrections" in result["_validation"]
        assert "warnings" in result["_validation"]
        assert "passed" in result["_validation"]
        assert "total_checks" in result["_validation"]

    def test_validation_flags_regime_mismatch(self):
        """macro_regime not containing computed regime should be flagged."""
        from finias.agents.macro_strategist.agent import MacroStrategist

        regime = self._make_regime()  # primary_regime = TRANSITION
        interp = {
            "summary": "Test",
            "key_findings": [],
            "risks": [],
            "watch_items": [],
            "macro_regime": "Risk-Off Crisis",  # Doesn't contain "transition"
            "binding_constraint": "inflation",
            "key_metrics": {},
        }

        class MockAgent:
            _validate_interpretation = MacroStrategist._validate_interpretation

        agent = MockAgent()
        result = agent._validate_interpretation(interp, regime)

        assert any(w["field"] == "macro_regime" for w in result["_validation"]["warnings"])
