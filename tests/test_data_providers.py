"""
Data provider tests.

Tests that data clients can be instantiated and have correct structure.
Integration tests with actual APIs require valid API keys.
"""

import pytest
from finias.data.providers.fred_client import FredClient, MACRO_SERIES


def test_fred_macro_series():
    """Verify MACRO_SERIES has all required series."""
    required = ["DGS2", "DGS10", "DGS30", "VIXCLS", "T10Y2Y", "FEDFUNDS"]
    for series_id in required:
        assert series_id in MACRO_SERIES, f"Missing required FRED series: {series_id}"


def test_polygon_client_init():
    """Test PolygonClient can be instantiated."""
    from finias.data.providers.polygon_client import PolygonClient
    client = PolygonClient(api_key="test_key")
    assert client.api_key == "test_key"
    assert client.max_calls_per_minute >= 1  # Default 5 or overridden by .env


def test_fred_client_init():
    """Test FredClient can be instantiated."""
    client = FredClient(api_key="test_key")
    assert client.api_key == "test_key"
