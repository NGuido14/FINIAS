"""
Tests for TA history tool and backfill script.
"""

import pytest
from datetime import date, timedelta


class TestHistoryToolDefinition:
    """Test the TA history tool definition."""

    def test_tool_definition_importable(self):
        from finias.agents.technical_analyst.history import get_ta_history_tool_definition
        td = get_ta_history_tool_definition()
        assert td["name"] == "query_ta_history"
        assert "input_schema" in td

    def test_tool_has_query_types(self):
        from finias.agents.technical_analyst.history import get_ta_history_tool_definition
        td = get_ta_history_tool_definition()
        props = td["input_schema"]["properties"]
        assert "query_type" in props
        assert "symbols" in props
        assert "trend_regime" in props
        assert "divergence_type" in props

    def test_query_types_enum(self):
        from finias.agents.technical_analyst.history import get_ta_history_tool_definition
        td = get_ta_history_tool_definition()
        enum = td["input_schema"]["properties"]["query_type"]["enum"]
        assert "symbol_history" in enum
        assert "divergence_scan" in enum
        assert "regime_scan" in enum
        assert "accuracy" in enum
        assert "strongest_signals" in enum


class TestDateParsing:
    """Test date parsing helpers."""

    def test_parse_relative_3mo(self):
        from finias.agents.technical_analyst.history import _parse_relative_date
        ref = date(2026, 4, 8)
        result = _parse_relative_date("3mo", ref)
        assert result < ref
        assert (ref - result).days == 90

    def test_parse_relative_1yr(self):
        from finias.agents.technical_analyst.history import _parse_relative_date
        ref = date(2026, 4, 8)
        result = _parse_relative_date("1yr", ref)
        assert (ref - result).days == 365

    def test_parse_iso_date(self):
        from finias.agents.technical_analyst.history import _parse_date
        result = _parse_date("2025-06-15", date(2026, 1, 1))
        assert result == date(2025, 6, 15)

    def test_parse_invalid_falls_back(self):
        from finias.agents.technical_analyst.history import _parse_date
        fallback = date(2026, 1, 1)
        result = _parse_date("not-a-date", fallback)
        assert result == fallback


class TestBackfillImports:
    """Test backfill script can be imported."""

    def test_backfill_imports(self):
        from finias.scripts.backfill_ta_signals import (
            backfill_signals,
            compute_forward_returns,
            generate_accuracy_report,
        )
        assert callable(backfill_signals)
        assert callable(compute_forward_returns)
        assert callable(generate_accuracy_report)


class TestMigration:
    """Test v012 migration file."""

    def test_migration_exists(self):
        from pathlib import Path
        p = Path(__file__).parent.parent / "finias" / "core" / "database" / "schemas" / "v012_ta_forward_returns.sql"
        assert p.exists()

    def test_migration_adds_columns(self):
        from pathlib import Path
        p = Path(__file__).parent.parent / "finias" / "core" / "database" / "schemas" / "v012_ta_forward_returns.sql"
        content = p.read_text()
        assert "fwd_return_5d" in content
        assert "fwd_return_20d" in content
        assert "fwd_return_60d" in content
        assert "close_price" in content
