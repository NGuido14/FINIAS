"""Tests for Session B: Ground-truth validation, Brent integration, SKEW index."""

import pytest


class TestGroundTruthModule:
    def test_validation_module_imports(self):
        from finias.validation.ground_truth import (
            validate_sahm_rule,
            validate_2s10s_spread,
            run_all_validations,
        )

    def test_new_fred_series_defined(self):
        from finias.data.providers.fred_client import MACRO_SERIES
        assert "SAHMREALTIME" in MACRO_SERIES
        assert "T10Y2Y" in MACRO_SERIES
        # SKEW moved to yfinance live price feed (no longer in MACRO_SERIES)
        from finias.data.providers.price_feed import LIVE_INSTRUMENTS
        assert "skew" in LIVE_INSTRUMENTS


class TestBrentIntegration:
    def test_cross_asset_has_brent_fields(self):
        from finias.agents.macro_strategist.computations.cross_asset import CrossAssetAnalysis
        ca = CrossAssetAnalysis()
        assert hasattr(ca, 'brent_price')
        assert hasattr(ca, 'brent_change_20d_pct')
        assert hasattr(ca, 'wti_brent_spread')
        assert hasattr(ca, 'wti_brent_spread_widening')

    def test_cross_asset_to_dict_includes_brent(self):
        from finias.agents.macro_strategist.computations.cross_asset import CrossAssetAnalysis
        ca = CrossAssetAnalysis(
            oil_price=100.0,
            brent_price=108.0,
            wti_brent_spread=-8.0,
            wti_brent_spread_widening=True,
        )
        d = ca.to_dict()
        assert d["oil"]["wti_price"] == 100.0
        assert d["oil"]["brent_price"] == 108.0
        assert d["oil"]["wti_brent_spread"] == -8.0
        assert d["oil"]["wti_brent_spread_widening"] is True

    def test_analyze_cross_assets_accepts_brent(self):
        """analyze_cross_assets should accept brent_series parameter."""
        import inspect
        from finias.agents.macro_strategist.computations.cross_asset import analyze_cross_assets
        sig = inspect.signature(analyze_cross_assets)
        assert "brent_series" in sig.parameters

    def test_brent_computation(self):
        """Brent price and spread should be computed correctly."""
        from finias.agents.macro_strategist.computations.cross_asset import analyze_cross_assets

        # Create minimal test data
        wti = [{"date": f"2026-01-{i:02d}", "value": 70.0 + i * 0.5} for i in range(1, 25)]
        brent = [{"date": f"2026-01-{i:02d}", "value": 78.0 + i * 0.5} for i in range(1, 25)]

        result = analyze_cross_assets(
            dxy_series=[], hy_spread_series=[], breakeven_5y=[], breakeven_10y=[],
            oil_series=wti, brent_series=brent,
        )

        assert result.oil_price is not None
        assert result.brent_price is not None
        assert result.wti_brent_spread is not None
        assert result.wti_brent_spread < 0  # Brent should be higher

    def test_spread_widening_flag(self):
        """Wide spread should set wti_brent_spread_widening flag."""
        from finias.agents.macro_strategist.computations.cross_asset import analyze_cross_assets

        wti = [{"date": f"2026-01-{i:02d}", "value": 90.0} for i in range(1, 25)]
        brent = [{"date": f"2026-01-{i:02d}", "value": 100.0} for i in range(1, 25)]

        result = analyze_cross_assets(
            dxy_series=[], hy_spread_series=[], breakeven_5y=[], breakeven_10y=[],
            oil_series=wti, brent_series=brent,
        )

        assert result.wti_brent_spread_widening is True  # |spread| > $5


class TestSkewIntegration:
    def test_volatility_has_skew_fields(self):
        from finias.agents.macro_strategist.computations.volatility import VolatilityAnalysis
        vol = VolatilityAnalysis(
            vix_current=25.0, vix_percentile_1y=90.0,
            vix_change_1d=0.5, vix_change_5d=2.0, vix_change_20d=-3.0,
            vix_sma_20=24.0, vix_is_elevated=True, vix_is_spike=False,
            realized_vol_20d=22.0, realized_vol_60d=20.0, iv_rv_spread=3.0,
            vol_regime="elevated", vol_risk_score=0.5,
        )
        assert hasattr(vol, 'skew_current')
        assert hasattr(vol, 'skew_regime')

    def test_volatility_to_dict_includes_skew(self):
        from finias.agents.macro_strategist.computations.volatility import VolatilityAnalysis
        vol = VolatilityAnalysis(
            vix_current=25.0, vix_percentile_1y=90.0,
            vix_change_1d=0.5, vix_change_5d=2.0, vix_change_20d=-3.0,
            vix_sma_20=24.0, vix_is_elevated=True, vix_is_spike=False,
            realized_vol_20d=22.0, realized_vol_60d=20.0, iv_rv_spread=3.0,
            vol_regime="elevated", vol_risk_score=0.5,
            skew_current=142.0, skew_regime="elevated",
        )
        d = vol.to_dict()
        assert d["skew"]["current"] == 142.0
        assert d["skew"]["regime"] == "elevated"

    def test_analyze_volatility_accepts_skew(self):
        import inspect
        from finias.agents.macro_strategist.computations.volatility import analyze_volatility
        sig = inspect.signature(analyze_volatility)
        assert "skew_series" in sig.parameters

    def test_skew_regime_classification(self):
        """SKEW should classify into correct regimes."""
        from finias.agents.macro_strategist.computations.volatility import analyze_volatility

        vix = [{"date": "2026-01-01", "value": 20.0}] * 260
        spy = [{"date": f"2026-01-{i:02d}", "close": 500.0 + i * 0.1} for i in range(1, 70)]

        # Normal SKEW
        skew_normal = [{"date": "2026-01-01", "value": 125.0}]
        result = analyze_volatility(vix, spy, skew_series=skew_normal)
        assert result.skew_regime == "normal"

        # Elevated SKEW
        skew_elevated = [{"date": "2026-01-01", "value": 142.0}]
        result = analyze_volatility(vix, spy, skew_series=skew_elevated)
        assert result.skew_regime == "elevated"

        # Extreme SKEW
        skew_extreme = [{"date": "2026-01-01", "value": 155.0}]
        result = analyze_volatility(vix, spy, skew_series=skew_extreme)
        assert result.skew_regime == "extreme"

        # Complacent SKEW
        skew_low = [{"date": "2026-01-01", "value": 110.0}]
        result = analyze_volatility(vix, spy, skew_series=skew_low)
        assert result.skew_regime == "complacent"
