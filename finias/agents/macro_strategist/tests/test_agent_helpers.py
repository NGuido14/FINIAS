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
