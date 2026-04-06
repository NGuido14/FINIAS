"""Integration tests for data validation layer wiring."""

import pytest
import json


class TestMigrationV009:
    """Verify v009 migration file exists and is valid SQL."""

    def test_migration_file_exists(self):
        """v009 migration SQL file should exist."""
        from pathlib import Path
        migration_path = Path(__file__).parent.parent.parent.parent / "core" / "database" / "schemas" / "v009_data_quality.sql"
        assert migration_path.exists(), f"Migration file not found at {migration_path}"

    def test_migration_has_alter(self):
        """v009 should add data_quality_json column."""
        from pathlib import Path
        migration_path = Path(__file__).parent.parent.parent.parent / "core" / "database" / "schemas" / "v009_data_quality.sql"
        content = migration_path.read_text()
        assert "data_quality_json" in content
        assert "JSONB" in content


class TestAgentQualityIntegration:
    """Verify agent.py has quality check integration points."""

    def test_agent_imports_validation(self):
        """Agent should be able to import validation modules."""
        from finias.data.validation.quality import (
            check_series_gaps,
            validate_series,
            DataQualityReport,
            CONSECUTIVE_CRITICAL,
        )
        from finias.data.validation.bounds import check_computation_bounds
        assert callable(check_series_gaps)
        assert callable(validate_series)
        assert callable(check_computation_bounds)
        assert len(CONSECUTIVE_CRITICAL) >= 7

    def test_quality_report_to_dict_serializable(self):
        """DataQualityReport.to_dict() should be JSON serializable."""
        from finias.data.validation.quality import DataQualityReport, SeriesQualityReport
        report = DataQualityReport()
        report.critical_issues.append("UNRATE: gap detected")
        report.series_reports["UNRATE"] = SeriesQualityReport(
            series_id="UNRATE", status="critical", is_consecutive=False
        )
        report.overall_status = "critical"
        # Must be JSON serializable for data_quality_json column
        serialized = json.dumps(report.to_dict(), default=str)
        parsed = json.loads(serialized)
        assert parsed["overall_status"] == "critical"
        assert parsed["critical_count"] == 1

    def test_quality_warnings_for_data_notes(self):
        """Quality warnings should generate Claude-readable data notes."""
        from finias.data.validation.quality import DataQualityReport
        report = DataQualityReport()
        report.critical_issues.append("UNRATE: gap detected — Sahm Rule may be inaccurate")
        notes = report.get_quality_warnings_for_notes()
        assert len(notes) >= 1
        assert "CRITICAL" in notes[0]
        assert "UNRATE" in notes[0]

    def test_freshness_warnings_for_downstream(self):
        """Freshness warnings should flow to MacroContext."""
        from finias.data.validation.quality import (
            DataQualityReport, SeriesQualityReport, CONSECUTIVE_CRITICAL,
        )
        report = DataQualityReport()
        report.series_reports["UNRATE"] = SeriesQualityReport(
            series_id="UNRATE", status="critical", is_consecutive=False,
        )
        warnings = report.get_freshness_warnings()
        assert any("UNRATE" in w for w in warnings)
        assert any("Sahm" in w for w in warnings)

    def test_bounds_check_with_real_structure(self):
        """Bounds check against a realistic key_levels dict."""
        from finias.data.validation.bounds import check_computation_bounds
        key_levels = {
            "vix": 24.54,
            "hy_spread": 3.17,
            "recession_prob": 0.09,
            "core_pce_yoy": 3.06,
            "fed_funds": 3.64,
            "net_liquidity": 5825519.0,  # Raw (not trillion)
            "sahm_value": 0.2,
            "spread_2s10s": 0.52,
            "nfci": -0.43,
        }
        violations = check_computation_bounds(key_levels)
        # net_liquidity in raw form is ~5.8M (millions) which is outside
        # the trillion-scaled bounds. This is expected — the bounds are
        # for the trillion-scaled value, not raw.
        # All other values should be within bounds.
        non_liq_violations = [v for v in violations if "net_liquidity" not in v]
        assert len(non_liq_violations) == 0


class TestDiagnosticsIntegration:
    """Verify diagnostics can import and call quality checks."""

    def test_diagnostics_can_import_quality(self):
        """Diagnostics should be able to import quality modules."""
        from finias.data.validation.fred_quality import validate_all_fred, detect_fred_gaps
        from finias.data.validation.polygon_quality import validate_all_polygon
        from finias.data.validation.bounds import check_computation_bounds
        assert callable(validate_all_fred)
        assert callable(detect_fred_gaps)
        assert callable(validate_all_polygon)
        assert callable(check_computation_bounds)


class TestMorningRefreshIntegration:
    """Verify morning refresh can import quality modules."""

    def test_morning_refresh_can_import(self):
        """Morning refresh should import gap detection for auto-backfill."""
        from finias.data.validation.fred_quality import detect_fred_gaps
        from finias.data.validation.quality import CONSECUTIVE_CRITICAL
        assert callable(detect_fred_gaps)
        assert "UNRATE" in CONSECUTIVE_CRITICAL

    def test_fred_frequency_covers_all_macro_series(self):
        """FRED_FREQUENCY should have entries for all series used in agent.py."""
        from finias.data.validation.fred_quality import FRED_FREQUENCY
        from finias.data.providers.fred_client import MACRO_SERIES
        # Not all MACRO_SERIES need frequency mapping (some are reference only)
        # But all CONSECUTIVE_CRITICAL must be mapped
        from finias.data.validation.quality import CONSECUTIVE_CRITICAL
        for sid in CONSECUTIVE_CRITICAL:
            assert sid in FRED_FREQUENCY, f"{sid} is consecutive-critical but not in FRED_FREQUENCY"
