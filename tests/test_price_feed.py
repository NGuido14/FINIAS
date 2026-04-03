"""Tests for yfinance live price feed — shared infrastructure."""

import pytest


class TestPriceFeedModule:
    def test_price_feed_imports(self):
        """Verify the price feed module can be imported."""
        from finias.data.providers.price_feed import (
            fetch_live_prices, store_live_prices,
            get_live_prices, get_current_prices,
            LIVE_INSTRUMENTS, PRICES_REDIS_KEY,
        )
        assert len(LIVE_INSTRUMENTS) == 7
        assert PRICES_REDIS_KEY == "prices:live"

    def test_live_instruments_mapping(self):
        """Verify all instrument ticker mappings."""
        from finias.data.providers.price_feed import LIVE_INSTRUMENTS
        assert LIVE_INSTRUMENTS["vix"] == "^VIX"
        assert LIVE_INSTRUMENTS["skew"] == "^SKEW"
        assert LIVE_INSTRUMENTS["wti"] == "CL=F"
        assert LIVE_INSTRUMENTS["brent"] == "BZ=F"
        assert LIVE_INSTRUMENTS["gold"] == "GC=F"
        assert LIVE_INSTRUMENTS["dxy"] == "DX-Y.NYB"
        assert LIVE_INSTRUMENTS["spx"] == "^GSPC"

    def test_skew_not_in_fred_series(self):
        """SKEW should NOT be in MACRO_SERIES — yfinance is the source."""
        from finias.data.providers.fred_client import MACRO_SERIES
        assert "SKEW" not in MACRO_SERIES

    def test_build_data_notes_accepts_live_prices(self):
        """_build_data_notes should accept live_prices parameter."""
        import inspect
        from finias.agents.macro_strategist.agent import MacroStrategist
        sig = inspect.signature(MacroStrategist._build_data_notes)
        assert "live_prices" in sig.parameters


class TestPriceFeedArchitecture:
    def test_prices_key_separate_from_regime(self):
        """prices:live Redis key must be separate from regime:current."""
        from finias.data.providers.price_feed import PRICES_REDIS_KEY
        assert PRICES_REDIS_KEY == "prices:live"
        assert PRICES_REDIS_KEY != "regime:current"

    def test_regime_dict_does_not_contain_live_prices(self):
        """Regime to_dict() should NOT include live_prices — clean separation."""
        from finias.agents.macro_strategist.computations.regime import RegimeAssessment
        from finias.core.agents.models import MarketRegime
        regime = RegimeAssessment(primary_regime=MarketRegime.TRANSITION)
        d = regime.to_dict()
        assert "live_prices" not in d

    def test_volatility_handles_single_skew_value(self):
        """Volatility module works with single-value SKEW from yfinance."""
        from finias.agents.macro_strategist.computations.volatility import analyze_volatility
        vix = [{"date": "2026-01-01", "value": 25.0}] * 260
        spy = [{"date": f"2026-01-{i:02d}", "close": 500.0 + i * 0.1} for i in range(1, 70)]
        skew = [{"date": "2026-04-02", "value": 146.95}]
        result = analyze_volatility(vix, spy, skew_series=skew)
        assert result.skew_current == 146.95
        assert result.skew_regime == "elevated"

    def test_get_current_prices_function_exists(self):
        """get_current_prices should be importable for downstream agents."""
        from finias.data.providers.price_feed import get_current_prices
        import inspect
        sig = inspect.signature(get_current_prices)
        assert "state" in sig.parameters
        assert "max_age_seconds" in sig.parameters
