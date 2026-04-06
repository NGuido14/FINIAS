"""Comprehensive tests for the data validation layer."""

import pytest
from datetime import date, timedelta

from finias.data.validation.quality import (
    check_series_gaps,
    check_staleness,
    check_value_bounds,
    check_outliers,
    validate_series,
    SeriesQualityReport,
    DataQualityReport,
    CONSECUTIVE_CRITICAL,
)
from finias.data.validation.bounds import (
    check_computation_bounds,
    check_value_change,
    COMPUTATION_BOUNDS,
)


# ============================================================================
# Helpers
# ============================================================================

def _monthly_series(values: list[float], start_year=2024, start_month=1) -> list[dict]:
    """Create a monthly series with correct dates."""
    result = []
    for i, v in enumerate(values):
        month = start_month + i
        year = start_year + (month - 1) // 12
        m = ((month - 1) % 12) + 1
        result.append({"date": f"{year}-{m:02d}-01", "value": v})
    return result


def _daily_series(count: int, start_date=None, skip_dates=None) -> list[dict]:
    """Create a daily series, optionally skipping specific dates."""
    if start_date is None:
        start_date = date(2025, 1, 1)
    if skip_dates is None:
        skip_dates = set()

    result = []
    current = start_date
    added = 0
    while added < count:
        if current.weekday() < 5 and current not in skip_dates:  # Mon-Fri
            result.append({"date": str(current), "value": 100.0 + added * 0.1})
            added += 1
        current += timedelta(days=1)
    return result


def _weekly_series(count: int, start_date=None, skip_indices=None) -> list[dict]:
    """Create a weekly series, optionally skipping specific weeks."""
    if start_date is None:
        start_date = date(2024, 1, 4)  # A Thursday
    if skip_indices is None:
        skip_indices = set()

    result = []
    for i in range(count):
        if i not in skip_indices:
            d = start_date + timedelta(weeks=i)
            result.append({"date": str(d), "value": 200000 + i * 1000})
    return result


# ============================================================================
# 1. Gap Detection Tests
# ============================================================================

class TestSeriesGaps:
    def test_no_gaps_monthly(self):
        """Consecutive monthly data → no gaps."""
        series = _monthly_series([4.0, 4.1, 4.2, 4.3, 4.4, 4.5])
        gaps = check_series_gaps(series, "monthly", "TEST")
        assert len(gaps) == 0

    def test_one_month_gap(self):
        """Missing one month in monthly data → 1 gap detected."""
        series = _monthly_series([4.0, 4.1, 4.2, 4.3, 4.4])
        # Remove March (index 2)
        del series[2]
        gaps = check_series_gaps(series, "monthly", "TEST")
        assert len(gaps) == 1
        assert gaps[0]["gap_periods"] == 1

    def test_two_month_gap(self):
        """Missing two consecutive months → 1 gap with gap_periods=2."""
        # Jan, Feb, [skip Mar, Apr], May
        series = [
            {"date": "2025-01-01", "value": 4.0},
            {"date": "2025-02-01", "value": 4.1},
            {"date": "2025-05-01", "value": 4.2},
        ]
        gaps = check_series_gaps(series, "monthly", "TEST")
        assert len(gaps) == 1
        assert gaps[0]["gap_periods"] == 2

    def test_no_gaps_weekly(self):
        """Consecutive weekly data → no gaps."""
        series = _weekly_series(10)
        gaps = check_series_gaps(series, "weekly", "TEST")
        assert len(gaps) == 0

    def test_weekly_gap(self):
        """Missing one week → 1 gap."""
        series = _weekly_series(10, skip_indices={4})
        gaps = check_series_gaps(series, "weekly", "TEST")
        assert len(gaps) == 1

    def test_daily_no_gaps(self):
        """Consecutive daily data (weekdays) → no gaps."""
        series = _daily_series(20)
        gaps = check_series_gaps(series, "daily", "TEST")
        assert len(gaps) == 0

    def test_daily_with_gap(self):
        """Missing a full trading week → gap detected."""
        # Skip an entire week
        skip = {date(2025, 1, 13) + timedelta(days=i) for i in range(5)}
        series = _daily_series(30, skip_dates=skip)
        gaps = check_series_gaps(series, "daily", "TEST")
        assert len(gaps) >= 1

    def test_empty_series(self):
        """Empty series → no gaps (nothing to check)."""
        gaps = check_series_gaps([], "monthly", "TEST")
        assert len(gaps) == 0

    def test_single_observation(self):
        """One observation → no gaps."""
        gaps = check_series_gaps([{"date": "2025-01-01", "value": 4.0}], "monthly", "TEST")
        assert len(gaps) == 0

    def test_unrate_october_gap(self):
        """
        Reproduce the exact UNRATE bug: October 2025 missing.
        This is the specific scenario that corrupted the Sahm Rule.
        """
        series = [
            {"date": "2025-06-01", "value": 4.1},
            {"date": "2025-07-01", "value": 4.3},
            {"date": "2025-08-01", "value": 4.3},
            {"date": "2025-09-01", "value": 4.4},
            # October MISSING
            {"date": "2025-11-01", "value": 4.5},
            {"date": "2025-12-01", "value": 4.4},
            {"date": "2026-01-01", "value": 4.3},
        ]
        gaps = check_series_gaps(series, "monthly", "UNRATE")
        assert len(gaps) == 1
        assert gaps[0]["from_date"] == "2025-09-01"
        assert gaps[0]["to_date"] == "2025-11-01"
        assert gaps[0]["gap_periods"] == 1


# ============================================================================
# 2. Staleness Tests
# ============================================================================

class TestStaleness:
    def test_fresh_daily(self):
        """Recent daily data → not stale."""
        series = [{"date": str(date.today() - timedelta(days=1)), "value": 25.0}]
        result = check_staleness(series, "daily", "VIX")
        assert result is None  # Not stale

    def test_stale_daily(self):
        """Daily data 10 days old → stale."""
        series = [{"date": str(date.today() - timedelta(days=10)), "value": 25.0}]
        result = check_staleness(series, "daily", "VIX")
        assert result is not None
        assert result["status"] == "stale"

    def test_fresh_monthly(self):
        """Monthly data 30 days old → not stale."""
        series = [{"date": str(date.today() - timedelta(days=30)), "value": 4.3}]
        result = check_staleness(series, "monthly", "UNRATE")
        assert result is None

    def test_stale_monthly(self):
        """Monthly data 90 days old → stale."""
        series = [{"date": str(date.today() - timedelta(days=90)), "value": 4.3}]
        result = check_staleness(series, "monthly", "UNRATE")
        assert result is not None

    def test_empty_series(self):
        """Empty series → missing status."""
        result = check_staleness([], "daily", "TEST")
        assert result is not None
        assert result["status"] == "missing"


# ============================================================================
# 3. Outlier Tests
# ============================================================================

class TestOutliers:
    def test_no_outliers(self):
        """Normal data → no outliers."""
        series = [{"date": f"2024-{i:02d}-01", "value": 4.0 + i * 0.01}
                  for i in range(1, 55)]
        outliers = check_outliers(series, series_id="TEST")
        assert len(outliers) == 0

    def test_extreme_outlier(self):
        """Value 10 std devs from mean → flagged."""
        series = [{"date": f"2024-{((i % 12) + 1):02d}-01", "value": 4.0 + (i % 3) * 0.01}
                  for i in range(54)]
        series.append({"date": "2028-07-01", "value": 40.0})  # 10x normal
        outliers = check_outliers(series, z_threshold=4.0, series_id="TEST")
        assert len(outliers) == 1
        assert outliers[0]["z_score"] > 4.0

    def test_insufficient_data(self):
        """Too few observations → no outlier check."""
        series = [{"date": "2025-01-01", "value": 4.0}]
        outliers = check_outliers(series, series_id="TEST")
        assert len(outliers) == 0


# ============================================================================
# 4. Validate Series Tests
# ============================================================================

class TestValidateSeries:
    def test_healthy_series(self):
        """Complete, fresh, no gaps → healthy."""
        series = _monthly_series([4.0 + i * 0.01 for i in range(24)],
                                start_year=2024, start_month=5)
        report = validate_series(series, "UNRATE", "monthly", consecutive_required=True)
        assert report.status == "healthy"
        assert report.is_consecutive is True
        assert len(report.gaps) == 0

    def test_critical_with_gap(self):
        """Gap in consecutive-required series → critical."""
        series = _monthly_series([4.0, 4.1, 4.2, 4.3, 4.4])
        del series[2]  # Remove one month
        report = validate_series(series, "UNRATE", "monthly", consecutive_required=True)
        assert report.status == "critical"
        assert report.is_consecutive is False
        assert any("consecutive" in w.lower() for w in report.warnings)

    def test_warning_with_gap_non_critical(self):
        """Gap in non-critical series → warning, not critical."""
        series = _monthly_series([4.0, 4.1, 4.2, 4.3, 4.4])
        del series[2]
        report = validate_series(series, "TCU", "monthly", consecutive_required=False)
        assert report.status == "warning"

    def test_empty_series_critical(self):
        """No data → critical."""
        report = validate_series([], "UNRATE", "monthly")
        assert report.status == "critical"

    def test_to_dict_structure(self):
        """to_dict returns expected keys."""
        series = _monthly_series([4.0, 4.1, 4.2])
        report = validate_series(series, "TEST", "monthly")
        d = report.to_dict()
        assert "series_id" in d
        assert "status" in d
        assert "gaps" in d
        assert "warnings" in d
        assert "is_consecutive" in d


# ============================================================================
# 5. Computation Bounds Tests
# ============================================================================

class TestBounds:
    def test_all_within_bounds(self):
        """Normal values → no violations."""
        key_levels = {
            "vix": 25.0,
            "hy_spread": 3.5,
            "recession_prob": 0.09,
            "core_pce_yoy": 3.06,
            "fed_funds": 3.64,
        }
        violations = check_computation_bounds(key_levels)
        assert len(violations) == 0

    def test_recession_prob_above_one(self):
        """Recession probability > 1.0 → violation."""
        violations = check_computation_bounds({"recession_prob": 1.5})
        assert len(violations) == 1
        assert "recession_prob" in violations[0]

    def test_negative_vix(self):
        """Negative VIX → violation."""
        violations = check_computation_bounds({"vix": -3.0})
        assert len(violations) == 1

    def test_extreme_sahm(self):
        """Sahm value of 5.0 → violation (max is 3.0)."""
        violations = check_computation_bounds({"sahm_value": 5.0})
        assert len(violations) == 1

    def test_missing_values_ignored(self):
        """None values → skipped, no violations."""
        violations = check_computation_bounds({"vix": None, "hy_spread": None})
        assert len(violations) == 0

    def test_bounds_dict_has_all_keys(self):
        """COMPUTATION_BOUNDS covers key fields."""
        assert "recession_prob" in COMPUTATION_BOUNDS
        assert "vix" in COMPUTATION_BOUNDS
        assert "composite_score" in COMPUTATION_BOUNDS
        assert "net_spec_percentile" in COMPUTATION_BOUNDS

    def test_value_change_normal(self):
        """Small change → no warning."""
        result = check_value_change(25.0, 24.0, "vix")
        assert result is None

    def test_value_change_extreme(self):
        """Huge change → warning."""
        result = check_value_change(250.0, 25.0, "vix")
        assert result is not None
        assert "LARGE CHANGE" in result

    def test_value_change_zero_prior(self):
        """Prior is zero → no warning (avoid division by zero)."""
        result = check_value_change(5.0, 0.0, "test")
        assert result is None


# ============================================================================
# 6. DataQualityReport Tests
# ============================================================================

class TestDataQualityReport:
    def test_empty_report(self):
        """Default report → unknown status."""
        report = DataQualityReport()
        assert report.overall_status == "unknown"
        assert len(report.critical_issues) == 0

    def test_quality_warnings_for_notes(self):
        """Critical issues generate data notes for Claude."""
        report = DataQualityReport()
        report.critical_issues.append("UNRATE: gap in monthly data, Sahm Rule affected")
        notes = report.get_quality_warnings_for_notes()
        assert len(notes) >= 1
        assert "CRITICAL" in notes[0]

    def test_freshness_warnings(self):
        """Gaps in consecutive-critical series generate freshness warnings."""
        report = DataQualityReport()
        series_report = SeriesQualityReport(
            series_id="UNRATE",
            is_consecutive=False,
            status="critical",
        )
        report.series_reports["UNRATE"] = series_report
        warnings = report.get_freshness_warnings()
        assert any("UNRATE" in w for w in warnings)

    def test_to_dict_structure(self):
        """to_dict returns expected keys."""
        report = DataQualityReport()
        d = report.to_dict()
        assert "overall_status" in d
        assert "critical_count" in d
        assert "warning_count" in d
        assert "healthy_count" in d

    def test_no_warnings_when_clean(self):
        """Clean data → no quality warnings for notes."""
        report = DataQualityReport()
        notes = report.get_quality_warnings_for_notes()
        assert len(notes) == 0


# ============================================================================
# 7. CONSECUTIVE_CRITICAL Coverage
# ============================================================================

class TestConsecutiveCritical:
    def test_unrate_is_critical(self):
        """UNRATE must be in CONSECUTIVE_CRITICAL."""
        assert "UNRATE" in CONSECUTIVE_CRITICAL

    def test_all_recession_model_inputs(self):
        """All recession model feature inputs should be consecutive-critical."""
        # The recession model uses: sahm (from UNRATE), claims (ICSA),
        # sentiment (UMCSENT), indpro (INDPRO), permits (PERMIT)
        assert "ICSA" in CONSECUTIVE_CRITICAL
        assert "UMCSENT" in CONSECUTIVE_CRITICAL
        assert "INDPRO" in CONSECUTIVE_CRITICAL
        assert "PERMIT" in CONSECUTIVE_CRITICAL

    def test_inflation_inputs(self):
        """Key inflation inputs should be consecutive-critical."""
        assert "PCEPILFE" in CONSECUTIVE_CRITICAL
        assert "CPIAUCSL" in CONSECUTIVE_CRITICAL


# ============================================================================
# 8. FRED Quality Tests (import verification)
# ============================================================================

class TestFREDQualityImports:
    def test_fred_quality_imports(self):
        """Verify FRED quality module can be imported."""
        from finias.data.validation.fred_quality import (
            validate_fred_series,
            validate_all_fred,
            detect_fred_gaps,
            FRED_FREQUENCY,
        )
        assert len(FRED_FREQUENCY) > 60
        assert "UNRATE" in FRED_FREQUENCY
        assert FRED_FREQUENCY["UNRATE"] == "monthly"
        assert FRED_FREQUENCY["VIXCLS"] == "daily"
        assert FRED_FREQUENCY["WALCL"] == "weekly"

    def test_polygon_quality_imports(self):
        """Verify Polygon quality module can be imported."""
        from finias.data.validation.polygon_quality import (
            validate_price_bars,
            validate_all_polygon,
            EXPECTED_SYMBOLS,
        )
        assert len(EXPECTED_SYMBOLS) == 19
        assert "SPY" in EXPECTED_SYMBOLS

    def test_bounds_imports(self):
        """Verify bounds module can be imported."""
        from finias.data.validation.bounds import (
            check_computation_bounds,
            check_value_change,
            COMPUTATION_BOUNDS,
        )
        assert len(COMPUTATION_BOUNDS) > 10
