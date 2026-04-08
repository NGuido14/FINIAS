"""
Tests for the FINIAS symbol universe management.

Tests cover:
- Macro ETF list integrity
- Universe module helper functions (imports, signatures)
- Migration file validity
- Batch loading methods exist
- Settings and Polygon client rate limit
- Script imports

NOTE: Tests that require Wikipedia access or a live database are NOT included.
Those are integration tests that run during seed_universe --check.
"""

import pytest
from finias.data.universe import (
    MACRO_ETFS,
    ETF_SECTOR_MAP,
    TIER_MACRO,
    TIER_SP500,
    fetch_sp500_from_wikipedia,
    get_active_symbols,
    populate_macro_etfs,
    populate_sp500_from_list,
)


class TestMacroETFs:
    """Test the macro ETF list."""

    def test_macro_etf_count(self):
        """Should have exactly 19 ETFs (matching current system)."""
        assert len(MACRO_ETFS) == 19

    def test_macro_contains_spy(self):
        assert "SPY" in MACRO_ETFS

    def test_macro_contains_all_sector_etfs(self):
        sector_etfs = ["XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLC", "XLY", "XLRE", "XLB"]
        for etf in sector_etfs:
            assert etf in MACRO_ETFS, f"Missing sector ETF: {etf}"

    def test_macro_no_duplicates(self):
        assert len(MACRO_ETFS) == len(set(MACRO_ETFS))


class TestETFSectorMap:
    """Test ETF sector mapping."""

    def test_eleven_sector_etfs_mapped(self):
        assert len(ETF_SECTOR_MAP) == 11

    def test_all_mapped_etfs_in_macro_list(self):
        for etf in ETF_SECTOR_MAP:
            assert etf in MACRO_ETFS, f"{etf} in sector map but not in MACRO_ETFS"

    def test_sector_names_are_standard(self):
        expected_sectors = {
            "Communication Services", "Consumer Discretionary", "Consumer Staples",
            "Energy", "Financials", "Health Care", "Industrials",
            "Information Technology", "Materials", "Real Estate", "Utilities",
        }
        assert set(ETF_SECTOR_MAP.values()) == expected_sectors


class TestTierConstants:
    """Test tier constants."""

    def test_tier_values(self):
        assert TIER_MACRO == "macro"
        assert TIER_SP500 == "sp500"


class TestFetchFunction:
    """Test that the Wikipedia fetch function exists and has correct signature."""

    def test_fetch_function_callable(self):
        assert callable(fetch_sp500_from_wikipedia)

    def test_fetch_function_is_sync(self):
        """fetch_sp500_from_wikipedia should be a regular function, not async."""
        import asyncio
        assert not asyncio.iscoroutinefunction(fetch_sp500_from_wikipedia)


class TestDBHelpers:
    """Test that database helper functions exist with correct signatures."""

    def test_get_active_symbols_is_async(self):
        import asyncio
        assert asyncio.iscoroutinefunction(get_active_symbols)

    def test_populate_macro_is_async(self):
        import asyncio
        assert asyncio.iscoroutinefunction(populate_macro_etfs)

    def test_populate_sp500_is_async(self):
        import asyncio
        assert asyncio.iscoroutinefunction(populate_sp500_from_list)


class TestMigration:
    """Verify the v010 migration file."""

    def test_migration_file_exists(self):
        from pathlib import Path
        p = Path(__file__).parent.parent / "finias" / "core" / "database" / "schemas" / "v010_symbol_universe.sql"
        assert p.exists(), f"Migration not found: {p}"

    def test_migration_creates_table(self):
        from pathlib import Path
        p = Path(__file__).parent.parent / "finias" / "core" / "database" / "schemas" / "v010_symbol_universe.sql"
        content = p.read_text()
        assert "symbol_universe" in content
        assert "CREATE TABLE" in content
        assert "tier" in content
        assert "is_active" in content
        assert "sector" in content


class TestBatchLoadingImport:
    """Verify batch loading methods exist on MarketDataCache."""

    def test_batch_method_exists(self):
        from finias.data.cache.market_cache import MarketDataCache
        assert hasattr(MarketDataCache, "get_batch_daily_bars")

    def test_staleness_method_exists(self):
        from finias.data.cache.market_cache import MarketDataCache
        assert hasattr(MarketDataCache, "get_universe_staleness")


class TestSettingsRateLimit:
    """Verify Polygon rate limit setting."""

    def test_settings_has_polygon_rate_limit(self):
        from finias.core.config.settings import Settings
        s = Settings()
        assert hasattr(s, "polygon_rate_limit")
        assert isinstance(s.polygon_rate_limit, int)
        assert s.polygon_rate_limit >= 1

    def test_default_rate_limit_is_5(self):
        from finias.core.config.settings import Settings
        s = Settings()
        assert s.polygon_rate_limit == 5


class TestPolygonClientRateLimit:
    """Verify PolygonClient reads rate limit from settings."""

    def test_polygon_client_uses_settings(self):
        from finias.data.providers.polygon_client import PolygonClient
        client = PolygonClient(api_key="test")
        assert client.max_calls_per_minute >= 1

    def test_polygon_client_override(self):
        from finias.data.providers.polygon_client import PolygonClient
        client = PolygonClient(api_key="test", max_calls_per_minute=200)
        assert client.max_calls_per_minute == 200


class TestScriptImports:
    """Verify scripts can be imported."""

    def test_seed_script(self):
        from finias.scripts.seed_universe import seed_universe_table, seed_price_data
        assert callable(seed_universe_table)
        assert callable(seed_price_data)

    def test_refresh_script(self):
        from finias.scripts.refresh_universe import refresh_prices
        assert callable(refresh_prices)


class TestRequirements:
    """Verify new dependencies are importable."""

    def test_pandas_importable(self):
        import pandas
        assert hasattr(pandas, "read_html")

    def test_lxml_importable(self):
        import lxml
