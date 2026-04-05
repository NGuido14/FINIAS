"""Tests for forward-looking intelligence enhancements."""

import pytest


class TestSectorReturns:
    """Test sector absolute return computation."""

    def test_sector_returns_field_exists(self):
        """BreadthAnalysis should have sector_absolute_returns field."""
        from finias.agents.macro_strategist.computations.breadth import BreadthAnalysis
        result = BreadthAnalysis()
        assert hasattr(result, 'sector_absolute_returns')
        assert result.sector_absolute_returns == {}

    def test_sector_returns_computed(self):
        """Sector returns should be computed from price data."""
        from finias.agents.macro_strategist.computations.breadth import analyze_breadth

        # Create realistic sector data with known returns
        sector_prices = {}
        for sym in ["XLK", "XLE", "XLF", "XLV", "XLI", "XLP", "XLU", "XLC", "XLY", "XLRE", "XLB"]:
            # 201 days of data, price goes from 100 to 110 (10% over 200 days)
            sector_prices[sym] = [
                {"date": f"2024-{(i//30)+1:02d}-{(i%30)+1:02d}", "close": 100.0 + i * 0.05}
                for i in range(201)
            ]

        spy_prices = [
            {"date": f"2024-{(i//30)+1:02d}-{(i%30)+1:02d}", "close": 500.0 + i * 0.25}
            for i in range(201)
        ]

        result = analyze_breadth(spy_prices, sector_prices, None)

        assert len(result.sector_absolute_returns) > 0
        # Check structure
        for sym, rets in result.sector_absolute_returns.items():
            assert "5d" in rets or "20d" in rets
            if "20d" in rets:
                assert isinstance(rets["20d"], float)

    def test_sector_returns_in_to_dict(self):
        """Sector returns should appear in to_dict() output."""
        from finias.agents.macro_strategist.computations.breadth import BreadthAnalysis
        result = BreadthAnalysis()
        result.sector_absolute_returns = {"XLE": {"5d": 4.1, "20d": 18.2}}
        d = result.to_dict()
        assert "sector_returns" in d
        assert d["sector_returns"]["XLE"]["20d"] == 18.2


class TestRecessionDrivers:
    """Test recession probability driver decomposition."""

    def test_recession_drivers_field_exists(self):
        """BusinessCycleAnalysis should have recession_drivers field."""
        from finias.agents.macro_strategist.computations.business_cycle import BusinessCycleAnalysis
        result = BusinessCycleAnalysis()
        assert hasattr(result, 'recession_drivers')

    def test_model_returns_dict_with_drivers(self):
        """predict_recession_probability should return dict with drivers."""
        from finias.agents.macro_strategist.models.recession_model import (
            predict_recession_probability, _COEFFICIENTS_PATH
        )
        if not _COEFFICIENTS_PATH.exists():
            pytest.skip("Coefficients not trained yet")

        result = predict_recession_probability(
            sahm_value=0.3,
            yield_curve_3m10y=1.5,
            claims_yoy_pct=5.0,
            permits_yoy_pct=-10.0,
            sentiment_yoy_pct=-5.0,
            indpro_yoy_pct=1.0,
        )

        assert isinstance(result, dict)
        assert "probability" in result
        assert "drivers" in result
        assert "base_rate" in result
        assert len(result["drivers"]) > 0
        # Drivers should be sorted by absolute contribution
        contribs = [abs(d["contribution"]) for d in result["drivers"]]
        assert contribs == sorted(contribs, reverse=True)


class TestForwardLookingFields:
    """Test that new interpretation fields are handled correctly."""

    def test_structuring_prompt_has_scenarios(self):
        """Structuring prompt should include scenarios field."""
        from finias.agents.macro_strategist.prompts.interpretation import MACRO_STRUCTURING_PROMPT
        assert "scenarios" in MACRO_STRUCTURING_PROMPT
        assert "catalysts" in MACRO_STRUCTURING_PROMPT
        assert "opportunities" in MACRO_STRUCTURING_PROMPT
        assert "regime_change_conditions" in MACRO_STRUCTURING_PROMPT

    def test_refresh_prompt_has_forward_looking(self):
        """Refresh prompt should include forward-looking analysis section."""
        from finias.agents.macro_strategist.prompts.refresh import MORNING_REFRESH_PROMPT
        assert "SCENARIOS" in MORNING_REFRESH_PROMPT
        assert "CATALYSTS" in MORNING_REFRESH_PROMPT
        assert "OPPORTUNITIES" in MORNING_REFRESH_PROMPT
        assert "REGIME CHANGE CONDITIONS" in MORNING_REFRESH_PROMPT

    def test_build_data_notes_has_sector_returns(self):
        """_build_data_notes should include sector returns when available."""
        import inspect
        from finias.agents.macro_strategist.agent import MacroStrategist
        source = inspect.getsource(MacroStrategist._build_data_notes)
        assert "sector_returns" in source or "sector_absolute_returns" in source or "SECTOR PERFORMANCE" in source
