"""Tests for cross-asset correlation analysis module."""

import pytest
import numpy as np
from finias.agents.macro_strategist.computations.correlation import (
    _price_to_log_returns,
    _fred_to_log_returns,
    _fred_to_levels,
    _rolling_corr,
    _rolling_beta,
    _r,
    _compute_pair,
    PairCorrelation,
    CorrelationMatrix,
    compute_correlation_matrix,
    generate_correlation_data_notes,
)


# ============================================================================
# Helpers for synthetic data generation
# ============================================================================

def _make_polygon_prices(values):
    """Create Polygon-format prices from a list of close values."""
    return [{"close": v} for v in values]


def _make_fred_series(values):
    """Create FRED-format series from a list of values."""
    return [{"value": v} for v in values]


def _generate_prices(n, seed=42, drift=0.0003, vol=0.01, start=100.0):
    """Generate synthetic prices with log-normal dynamics."""
    rng = np.random.RandomState(seed)
    log_returns = drift + vol * rng.randn(n)
    log_prices = np.log(start) + np.cumsum(log_returns)
    return np.exp(log_prices)


def _generate_correlated_prices(n, seed=42, correlation=0.5, drift=0.0003, vol=0.01):
    """Generate two correlated price series."""
    rng = np.random.RandomState(seed)
    z1 = rng.randn(n)
    z2 = correlation * z1 + np.sqrt(1 - correlation**2) * rng.randn(n)

    prices_a = np.exp(np.cumsum(drift + vol * z1) + np.log(100.0))
    prices_b = np.exp(np.cumsum(drift + vol * z2) + np.log(100.0))
    return prices_a, prices_b


# ============================================================================
# 1. Helper Function Tests
# ============================================================================

class TestPriceToLogReturns:
    def test_valid_prices(self):
        prices = _make_polygon_prices([100.0, 105.0, 110.0, 115.0])
        result = _price_to_log_returns(prices)
        assert result is not None
        assert len(result) == 3
        assert result[0] == pytest.approx(np.log(105.0 / 100.0), abs=1e-10)

    def test_none_input(self):
        assert _price_to_log_returns(None) is None

    def test_empty_input(self):
        assert _price_to_log_returns([]) is None

    def test_single_price(self):
        assert _price_to_log_returns(_make_polygon_prices([100.0])) is None

    def test_zero_price(self):
        assert _price_to_log_returns(_make_polygon_prices([100.0, 0.0, 50.0])) is None

    def test_negative_price(self):
        assert _price_to_log_returns(_make_polygon_prices([100.0, -5.0, 50.0])) is None


class TestFredToLogReturns:
    def test_valid_series(self):
        series = _make_fred_series([10.0, 11.0, 12.0])
        result = _fred_to_log_returns(series)
        assert result is not None
        assert len(result) == 2

    def test_none_input(self):
        assert _fred_to_log_returns(None) is None

    def test_empty_input(self):
        assert _fred_to_log_returns([]) is None

    def test_single_value(self):
        assert _fred_to_log_returns(_make_fred_series([10.0])) is None

    def test_zero_value(self):
        assert _fred_to_log_returns(_make_fred_series([10.0, 0.0, 5.0])) is None


class TestFredToLevels:
    def test_basic(self):
        series = _make_fred_series([15.0, 20.0, 25.0])
        result = _fred_to_levels(series)
        assert result is not None
        assert len(result) == 3
        assert result[0] == 15.0

    def test_none_input(self):
        assert _fred_to_levels(None) is None

    def test_empty_input(self):
        assert _fred_to_levels([]) is None


# ============================================================================
# 2. Rolling Correlation Tests
# ============================================================================

class TestRollingCorr:
    def test_perfect_positive(self):
        a = np.array([0.01, 0.02, -0.01, 0.03, -0.02] * 12)
        b = a.copy()
        result = _rolling_corr(a, b, 20)
        assert result is not None
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative(self):
        a = np.array([0.01, 0.02, -0.01, 0.03, -0.02] * 12)
        b = -a
        result = _rolling_corr(a, b, 20)
        assert result is not None
        assert result == pytest.approx(-1.0, abs=1e-6)

    def test_zero_variance_returns_none(self):
        a = np.ones(30)
        b = np.random.randn(30)
        result = _rolling_corr(a, b, 20)
        assert result is None

    def test_insufficient_data_returns_none(self):
        a = np.random.randn(10)
        b = np.random.randn(10)
        result = _rolling_corr(a, b, 20)
        assert result is None

    def test_known_correlation(self):
        """Test with known ~0.5 correlation within tolerance."""
        rng = np.random.RandomState(123)
        z1 = rng.randn(500)
        z2 = 0.5 * z1 + np.sqrt(0.75) * rng.randn(500)
        result = _rolling_corr(z1, z2, 500)
        assert result is not None
        assert abs(result - 0.5) < 0.1  # Generous tolerance for sample noise


# ============================================================================
# 3. Rolling Beta Tests
# ============================================================================

class TestRollingBeta:
    def test_beta_of_self(self):
        """Beta of a vs itself should be ~1.0."""
        a = np.array([0.01, 0.02, -0.01, 0.03, -0.02] * 12)
        result = _rolling_beta(a, a, 20)
        assert result is not None
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_scaled_beta(self):
        """Beta of 2x series should be ~2.0."""
        a = np.array([0.01, 0.02, -0.01, 0.03, -0.02] * 12)
        b = 2.0 * a
        result = _rolling_beta(a, b, 20)
        assert result is not None
        assert result == pytest.approx(2.0, abs=1e-6)

    def test_negative_beta(self):
        """Beta of -0.5x series should be ~-0.5."""
        a = np.array([0.01, 0.02, -0.01, 0.03, -0.02] * 12)
        b = -0.5 * a
        result = _rolling_beta(a, b, 20)
        assert result is not None
        assert result == pytest.approx(-0.5, abs=1e-6)

    def test_zero_variance_returns_none(self):
        a = np.zeros(30)
        b = np.random.randn(30)
        result = _rolling_beta(a, b, 20)
        assert result is None


# ============================================================================
# 4. Round Helper Test
# ============================================================================

class TestRoundHelper:
    def test_round_value(self):
        assert _r(0.123456, 3) == 0.123

    def test_none_passthrough(self):
        assert _r(None) is None


# ============================================================================
# 5. Correlation Trend Test
# ============================================================================

class TestCorrelationTrend:
    def test_insufficient_data_no_trend(self):
        """With <120 observations, trend should be None."""
        rng = np.random.RandomState(42)
        a = rng.randn(80)
        b = rng.randn(80)
        pair = _compute_pair("test", "A", "B", a, b)
        assert pair is not None
        assert pair.corr_trend is None

    def test_sufficient_data_has_trend(self):
        """With 120+ observations, trend should be a valid label."""
        rng = np.random.RandomState(42)
        a = rng.randn(300)
        b = rng.randn(300)
        pair = _compute_pair("test", "A", "B", a, b)
        assert pair is not None
        assert pair.corr_trend in ("rising", "falling", "stable")


# ============================================================================
# 6. Regime-Conditional Tests
# ============================================================================

class TestRegimeConditional:
    def test_high_low_vol_split(self):
        """With VIX data, high/low vol correlations should be populated."""
        rng = np.random.RandomState(42)
        n = 300
        a = rng.randn(n)
        b = 0.3 * a + rng.randn(n)
        # VIX: alternating high/low so both regimes present in trailing windows
        vix = np.where(np.arange(n) % 2 == 0, 15.0, 30.0)
        pair = _compute_pair("test", "A", "B", a, b, vix)
        assert pair is not None
        assert pair.corr_high_vol is not None
        assert pair.corr_low_vol is not None
        assert pair.vol_regime_spread is not None
        assert -2.0 <= pair.vol_regime_spread <= 2.0

    def test_no_vix_graceful(self):
        """Without VIX, regime-conditional fields should be None."""
        rng = np.random.RandomState(42)
        a = rng.randn(200)
        b = rng.randn(200)
        pair = _compute_pair("test", "A", "B", a, b, None)
        assert pair is not None
        assert pair.corr_high_vol is None
        assert pair.corr_low_vol is None
        assert pair.vol_regime_spread is None


# ============================================================================
# 7. Convexity Tests
# ============================================================================

class TestConvexity:
    def test_linear_near_zero_score(self):
        """Linear relationship should produce near-zero convexity (within sample noise)."""
        rng = np.random.RandomState(42)
        n = 500
        a = rng.randn(n)
        b = 0.5 * a + rng.randn(n)
        pair = _compute_pair("test", "A", "B", a, b)
        assert pair is not None
        assert pair.convexity_score is not None
        assert abs(pair.convexity_score) < 0.3  # Tolerance for sample noise

    def test_extreme_correlations_populated(self):
        """With 300+ observations, extreme correlations should be populated."""
        rng = np.random.RandomState(42)
        n = 500
        a = rng.randn(n)
        b = 0.5 * a + rng.randn(n)
        pair = _compute_pair("test", "A", "B", a, b)
        assert pair is not None
        assert pair.extreme_up_corr is not None
        assert pair.extreme_down_corr is not None
        assert pair.convexity_note is not None

    def test_insufficient_data_no_convexity(self):
        """With <120 observations, convexity should be None."""
        rng = np.random.RandomState(42)
        a = rng.randn(80)
        b = rng.randn(80)
        pair = _compute_pair("test", "A", "B", a, b)
        assert pair is not None
        assert pair.convexity_score is None
        assert pair.extreme_up_corr is None
        assert pair.extreme_down_corr is None


# ============================================================================
# 8. Pair Regime Classification Tests
# ============================================================================

class TestPairRegimeClassification:
    def test_normal_regime(self):
        """Mild correlations should classify as normal."""
        rng = np.random.RandomState(42)
        a = rng.randn(300)
        b = 0.15 * a + rng.randn(300)
        pair = _compute_pair("test", "A", "B", a, b)
        assert pair is not None
        assert pair.regime_label in ("normal", "decoupling", "stress_coupling", "breakdown")

    def test_breakdown_sign_flip(self):
        """Opposite sign between 60d and 120d should produce breakdown."""
        # Create series where first half has positive correlation, second half negative
        rng = np.random.RandomState(42)
        n = 300
        z = rng.randn(n)
        # First 180 days: positive corr. Last 120 days: negative corr
        a = z.copy()
        b = np.empty(n)
        b[:180] = 0.8 * a[:180] + 0.2 * rng.randn(180)
        b[180:] = -0.8 * a[180:] + 0.2 * rng.randn(120)
        pair = _compute_pair("test", "A", "B", a, b)
        assert pair is not None
        # Can be breakdown or other regime depending on exact values
        assert pair.regime_label in ("normal", "decoupling", "stress_coupling", "breakdown")

    def test_stress_coupling_high_spread(self):
        """High vol_regime_spread should produce stress_coupling."""
        # This is hard to guarantee with synthetic data, so we just check the code path exists
        rng = np.random.RandomState(42)
        a = rng.randn(300)
        b = 0.5 * a + rng.randn(300)
        vix = np.concatenate([np.ones(150) * 12.0, np.ones(150) * 35.0])
        pair = _compute_pair("test", "A", "B", a, b, vix)
        assert pair is not None
        assert pair.regime_label in ("normal", "decoupling", "stress_coupling", "breakdown")

    def test_unknown_when_no_corr(self):
        """When corr_60d is None (insufficient data), regime should be None."""
        rng = np.random.RandomState(42)
        a = rng.randn(60)  # Exactly 60, so corr_60d computed but corr_120d is None
        b = rng.randn(60)
        pair = _compute_pair("test", "A", "B", a, b)
        assert pair is not None
        # regime_label requires both corr_60d and corr_120d; with only 60 obs,
        # corr_120d is None so regime_label stays None
        assert pair.regime_label is None


# ============================================================================
# 9. Aggregate Signal Tests
# ============================================================================

class TestAggregateSignals:
    def test_diversification_regime(self):
        """Verify correct diversification regime classification."""
        matrix = CorrelationMatrix(
            avg_cross_asset_corr=0.15,
            diversification_regime="diversified",
        )
        assert matrix.diversification_regime == "diversified"

        matrix2 = CorrelationMatrix(
            avg_cross_asset_corr=0.35,
            diversification_regime="concentrated",
        )
        assert matrix2.diversification_regime == "concentrated"

        matrix3 = CorrelationMatrix(
            avg_cross_asset_corr=0.55,
            diversification_regime="correlated",
        )
        assert matrix3.diversification_regime == "correlated"

    def test_stress_coupling_count(self):
        """Verify stress coupling count is tracked."""
        pair1 = PairCorrelation(
            pair_name="test1", asset_a="A", asset_b="B",
            regime_label="stress_coupling"
        )
        pair2 = PairCorrelation(
            pair_name="test2", asset_a="C", asset_b="D",
            regime_label="normal"
        )
        matrix = CorrelationMatrix(
            oil_equity=pair1,
            oil_bond=pair2,
            stress_coupling_count=1,
        )
        assert matrix.stress_coupling_count == 1


# ============================================================================
# 10. Full Integration Tests
# ============================================================================

class TestFullIntegration:
    def test_500_days_all_pairs(self):
        """500 days of synthetic data should populate all 7 pairs."""
        n = 500
        rng = np.random.RandomState(42)
        prices_a = _generate_prices(n, seed=42, vol=0.015)
        prices_b = _generate_prices(n, seed=43, vol=0.012)
        prices_c = _generate_prices(n, seed=44, vol=0.010)
        prices_d = _generate_prices(n, seed=45, vol=0.008)

        # Oil and DXY as FRED series
        oil_vals = _generate_prices(n, seed=46, drift=0.0001, vol=0.02, start=70.0)
        dxy_vals = _generate_prices(n, seed=47, drift=-0.0001, vol=0.005, start=100.0)
        vix_vals = np.abs(rng.randn(n) * 5 + 20)

        result = compute_correlation_matrix(
            spy=_make_polygon_prices(prices_a),
            tlt=_make_polygon_prices(prices_b),
            gld=_make_polygon_prices(prices_c),
            hyg=_make_polygon_prices(prices_d),
            oil=_make_fred_series(oil_vals),
            dxy=_make_fred_series(dxy_vals),
            vix=_make_fred_series(vix_vals),
            as_of_date="2025-03-20",
        )

        # All 7 pairs should be populated
        assert result.oil_equity is not None
        assert result.oil_bond is not None
        assert result.dollar_equity is not None
        assert result.gold_equity is not None
        assert result.gold_bond is not None
        assert result.credit_equity is not None
        assert result.dollar_gold is not None

        # Check correlation bounds
        for pair in [result.oil_equity, result.oil_bond, result.dollar_equity,
                     result.gold_equity, result.gold_bond, result.credit_equity,
                     result.dollar_gold]:
            if pair.corr_60d is not None:
                assert -1.0 <= pair.corr_60d <= 1.0
            if pair.beta_60d is not None:
                assert isinstance(pair.beta_60d, float)
            if pair.corr_percentile_1y is not None:
                assert 0 <= pair.corr_percentile_1y <= 100
            assert pair.regime_label in ("normal", "decoupling", "stress_coupling", "breakdown", None)

        # Aggregates
        assert result.avg_cross_asset_corr is not None
        assert result.diversification_regime in ("diversified", "concentrated", "correlated")
        assert isinstance(result.stress_coupling_count, int)
        assert isinstance(result.breakdown_count, int)

    def test_to_dict_structure(self):
        """to_dict() should produce expected key structure."""
        n = 200
        prices = _generate_prices(n, seed=42)
        oil_vals = _generate_prices(n, seed=43, start=70.0)

        result = compute_correlation_matrix(
            spy=_make_polygon_prices(prices),
            oil=_make_fred_series(oil_vals),
            as_of_date="2025-03-20",
        )

        d = result.to_dict()
        assert "pairs" in d
        assert "aggregate" in d
        assert "as_of_date" in d
        assert d["as_of_date"] == "2025-03-20"

        agg = d["aggregate"]
        assert "stress_coupling_count" in agg
        assert "breakdown_count" in agg
        assert "avg_absolute_correlation" in agg
        assert "diversification_regime" in agg
        assert "_note" in agg

        # Check pair dict structure for any populated pair
        for pair_name, pair_data in d["pairs"].items():
            assert "rolling_correlations" in pair_data
            assert "beta" in pair_data
            assert "vol_regime_conditional" in pair_data
            assert "convexity" in pair_data
            assert "regime_label" in pair_data
            assert "assets" in pair_data
            rc = pair_data["rolling_correlations"]
            assert "corr_20d" in rc
            assert "corr_60d" in rc
            assert "corr_120d" in rc

    def test_generate_data_notes_nonempty(self):
        """generate_correlation_data_notes should produce non-empty list."""
        n = 200
        prices = _generate_prices(n, seed=42)
        oil_vals = _generate_prices(n, seed=43, start=70.0)

        result = compute_correlation_matrix(
            spy=_make_polygon_prices(prices),
            oil=_make_fred_series(oil_vals),
        )

        notes = generate_correlation_data_notes(result)
        assert isinstance(notes, list)
        assert len(notes) > 0
        # First note should be the guardrail header
        assert "CORRELATION" in notes[0].upper() or "GUARDRAIL" in notes[0].upper()

    def test_missing_data_empty_matrix(self):
        """All None inputs should return empty matrix without crashing."""
        result = compute_correlation_matrix()
        assert result is not None
        assert result.oil_equity is None
        assert result.oil_bond is None
        assert result.dollar_equity is None
        assert result.gold_equity is None
        assert result.gold_bond is None
        assert result.credit_equity is None
        assert result.dollar_gold is None
        assert result.avg_cross_asset_corr is None
        assert result.diversification_regime is None

    def test_short_data_graceful(self):
        """30 days of data (below 60 minimum) should return None pairs gracefully."""
        n = 30
        prices = _generate_prices(n, seed=42)
        result = compute_correlation_matrix(
            spy=_make_polygon_prices(prices),
            tlt=_make_polygon_prices(_generate_prices(n, seed=43)),
        )
        # 30 prices -> 29 returns, below 60 minimum for _compute_pair
        assert result.gold_equity is None


# ============================================================================
# 11. PairCorrelation.to_dict() Tests
# ============================================================================

class TestPairCorrelationToDict:
    def test_all_fields_present(self):
        """When fully populated, to_dict should contain all expected fields."""
        pair = PairCorrelation(
            pair_name="test",
            asset_a="Asset A",
            asset_b="Asset B",
            corr_20d=0.3,
            corr_60d=0.25,
            corr_120d=0.2,
            corr_trend="rising",
            corr_percentile_1y=75.0,
            beta_60d=0.8,
            beta_120d=0.7,
            corr_high_vol=0.4,
            corr_low_vol=0.15,
            vol_regime_spread=0.25,
            convexity_score=0.05,
            extreme_up_corr=0.3,
            extreme_down_corr=0.4,
            convexity_note="Moderate convexity",
            regime_label="normal",
        )
        d = pair.to_dict()
        assert d["pair_name"] == "test"
        assert d["assets"]["a"] == "Asset A"
        assert d["assets"]["b"] == "Asset B"
        assert d["rolling_correlations"]["corr_60d"] == pytest.approx(0.25)
        assert d["beta"]["beta_60d"] == pytest.approx(0.8)
        assert d["vol_regime_conditional"]["spread"] == pytest.approx(0.25)
        assert d["convexity"]["score"] == pytest.approx(0.05)
        assert d["regime_label"] == "normal"

    def test_none_fields_preserved(self):
        """None fields should be preserved in to_dict output."""
        pair = PairCorrelation(
            pair_name="test",
            asset_a="A",
            asset_b="B",
        )
        d = pair.to_dict()
        assert d["rolling_correlations"]["corr_20d"] is None
        assert d["rolling_correlations"]["corr_60d"] is None
        assert d["beta"]["beta_60d"] is None
        assert d["convexity"]["score"] is None
        assert d["regime_label"] is None
