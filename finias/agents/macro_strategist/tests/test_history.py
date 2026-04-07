"""Tests for the historical data layer."""

import pytest
from datetime import date, timedelta

from finias.agents.macro_strategist.history import (
    MATRIX_METRICS,
    MARKET_METRICS,
    ALL_METRIC_NAMES,
    auto_sample,
    compute_correlation,
    compute_cross_correlations,
    detect_inflections,
    format_trajectory_for_context,
    get_macro_history_tool_definition,
    _parse_date,
    _sample_monthly,
    _sample_weekly,
)


class TestMetricMappings:
    def test_matrix_metrics_not_empty(self):
        assert len(MATRIX_METRICS) >= 30

    def test_market_metrics_not_empty(self):
        assert len(MARKET_METRICS) >= 15

    def test_all_metric_names_combined(self):
        assert len(ALL_METRIC_NAMES) == len(MATRIX_METRICS) + len(MARKET_METRICS)

    def test_no_duplicate_metric_names(self):
        assert len(set(ALL_METRIC_NAMES)) == len(ALL_METRIC_NAMES)

    def test_key_metrics_present(self):
        for m in ["core_pce", "vix", "fed_funds", "unemployment", "oil_wti", "dxy", "spy", "xle"]:
            assert m in ALL_METRIC_NAMES, f"Missing metric: {m}"


class TestAutoSample:
    def test_short_series_unchanged(self):
        data = [{"date": f"2025-01-{i:02d}", "value": i} for i in range(1, 11)]
        assert auto_sample(data, max_points=20) == data

    def test_long_series_downsampled(self):
        data = [{"date": f"2025-01-{i:02d}", "value": i} for i in range(1, 32)]
        result = auto_sample(data, max_points=10)
        assert len(result) <= 10
        assert result[0] == data[0]  # First preserved
        assert result[-1] == data[-1]  # Last preserved

    def test_empty_series(self):
        assert auto_sample([]) == []

    def test_single_item(self):
        data = [{"date": "2025-01-01", "value": 1}]
        assert auto_sample(data) == data


class TestCorrelation:
    def test_perfect_positive(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        b = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        assert compute_correlation(a, b) == 1.0

    def test_perfect_negative(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        b = [12.0, 10.0, 8.0, 6.0, 4.0, 2.0]
        assert compute_correlation(a, b) == -1.0

    def test_uncorrelated(self):
        a = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        b = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0]
        corr = compute_correlation(a, b)
        assert abs(corr) < 0.5

    def test_too_short(self):
        assert compute_correlation([1, 2], [3, 4]) is None

    def test_mismatched_lengths(self):
        assert compute_correlation([1, 2, 3], [4, 5]) is None


class TestCrossCorrelations:
    def test_two_series(self):
        series = {
            "a": [{"date": f"2025-01-{i:02d}", "value": float(i)} for i in range(1, 21)],
            "b": [{"date": f"2025-01-{i:02d}", "value": float(i) * 2} for i in range(1, 21)],
        }
        result = compute_cross_correlations(series)
        assert "a vs b" in result
        assert result["a vs b"] == 1.0

    def test_single_series_no_correlations(self):
        series = {"a": [{"date": "2025-01-01", "value": 1.0}]}
        assert compute_cross_correlations(series) == {}


class TestInflectionDetection:
    def test_threshold_crossing_above(self):
        series = [
            {"date": "2025-01-01", "value": 2.8},
            {"date": "2025-02-01", "value": 2.9},
            {"date": "2025-03-01", "value": 3.1},
        ]
        result = detect_inflections(series, "core_pce")
        crossings = [i for i in result if i["type"] == "crossed_above"]
        assert len(crossings) >= 1
        assert crossings[0]["threshold"] == 3.0

    def test_threshold_crossing_below(self):
        series = [
            {"date": "2025-01-01", "value": 21},
            {"date": "2025-02-01", "value": 19},
        ]
        result = detect_inflections(series, "vix")
        crossings = [i for i in result if i["type"] == "crossed_below"]
        assert len(crossings) >= 1
        assert crossings[0]["threshold"] == 20

    def test_no_inflections_flat(self):
        series = [{"date": f"2025-01-{i:02d}", "value": 3.0} for i in range(1, 10)]
        result = detect_inflections(series, "core_pce")
        assert len(result) == 0

    def test_too_short_no_crash(self):
        series = [{"date": "2025-01-01", "value": 1.0}]
        result = detect_inflections(series, "vix")
        assert result == []


class TestDateParsing:
    def test_valid_iso(self):
        result = _parse_date("2025-01-20", date(2025, 1, 1))
        assert result == date(2025, 1, 20)

    def test_invalid_returns_fallback(self):
        fallback = date(2025, 1, 1)
        assert _parse_date("not-a-date", fallback) == fallback

    def test_empty_returns_fallback(self):
        fallback = date(2025, 1, 1)
        assert _parse_date("", fallback) == fallback


class TestSamplingHelpers:
    def test_sample_monthly(self):
        series = [
            {"date": "2025-01-01", "value": 1},
            {"date": "2025-01-15", "value": 2},
            {"date": "2025-02-01", "value": 3},
            {"date": "2025-02-15", "value": 4},
        ]
        result = _sample_monthly(series)
        assert len(result) == 2  # One per month

    def test_sample_weekly_short(self):
        series = [{"date": f"2025-01-{i:02d}", "value": i} for i in range(1, 11)]
        result = _sample_weekly(series)
        assert len(result) == len(series)  # Short enough, unchanged


class TestTrajectoryFormatting:
    def test_empty_trajectory(self):
        assert format_trajectory_for_context({}) == ""
        assert format_trajectory_for_context(None) == ""

    def test_basic_formatting(self):
        trajectory = {
            "metric_snapshots": {
                "vix": {
                    "label": "VIX",
                    "current": 24.48,
                    "12mo": 18.2,
                    "6mo": 22.1,
                    "3mo": 23.0,
                    "1mo": 24.0,
                }
            },
            "regime_history": [
                {"date": "2026-04-06", "regime": "transition", "binding": "inflation", "summary": "Test"}
            ],
            "inflection_points": [],
        }
        result = format_trajectory_for_context(trajectory)
        assert "VIX" in result
        assert "18.2" in result
        assert "24.48" in result
        assert "transition" in result


class TestToolDefinition:
    def test_tool_definition_structure(self):
        defn = get_macro_history_tool_definition()
        assert defn["name"] == "query_macro_history"
        assert "input_schema" in defn
        assert "metrics" in defn["input_schema"]["properties"]
        assert defn["input_schema"]["required"] == ["metrics"]

    def test_tool_description_contains_metrics(self):
        defn = get_macro_history_tool_definition()
        assert "core_pce" in defn["description"]
        assert "spy" in defn["description"]
        assert "vix" in defn["description"]
