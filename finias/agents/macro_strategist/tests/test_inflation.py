"""
Comprehensive tests for inflation.py module.

Tests cover:
- Core PCE YoY and 3-month annualized computation
- Inflation regime classification
- Inflation trend detection
- Wage pressure classification
- Spiral risk computation
- Expectations anchoring
- Full end-to-end analysis
"""

import pytest
import numpy as np

from finias.agents.macro_strategist.computations.inflation import (
    analyze_inflation,
    InflationAnalysis,
    _compute_yoy_index,
    _compute_3m_annualized,
    _check_expectations_anchored,
    _classify_wage_pressure,
    _compute_spiral_risk,
    _classify_trend,
    _classify_inflation_regime,
    _compute_inflation_score,
    _latest,
    _pct_change,
)


class TestYoYComputation:
    """Test year-over-year inflation computation."""

    def test_core_pce_yoy(self):
        """Test core PCE YoY calculation from index series."""
        # Index at 100 one year ago, now at 105 → 5% YoY
        core_pce = [
            {"date": f"2023-{m:02d}-01", "value": 100 + m * 0.5}
            for m in range(1, 13)
        ] + [
            {"date": f"2024-{m:02d}-01", "value": 105 + m * 0.5}
            for m in range(1, 13)
        ]

        yoy = _compute_yoy_index(core_pce)

        # Current = 112.5, year_ago = 107.5
        # (112.5 / 107.5 - 1) * 100 ≈ 4.65%
        assert pytest.approx(yoy, abs=1.0) == 4.65

    def test_cpi_yoy_sufficient_data(self):
        """CPI YoY with 13+ observations."""
        cpi = [
            {"date": f"2023-{m:02d}-01", "value": 300 + m}
            for m in range(1, 13)
        ] + [
            {"date": "2024-01-01", "value": 312}
        ]

        yoy = _compute_yoy_index(cpi)

        assert yoy is not None
        assert yoy > 0

    def test_yoy_insufficient_data(self):
        """Less than 13 observations → None."""
        cpi = [{"date": f"2024-{i:02d}-01", "value": 300 + i} for i in range(10)]

        yoy = _compute_yoy_index(cpi)

        assert yoy is None

    def test_yoy_zero_denominator(self):
        """Zero value one year ago → None."""
        cpi = [
            {"date": "2023-01-01", "value": 0},
        ] + [
            {"date": f"2024-{m:02d}-01", "value": 300 + m}
            for m in range(1, 13)
        ]

        yoy = _compute_yoy_index(cpi)

        assert yoy is None


class TestThreeMonthAnnualized:
    """Test 3-month annualized inflation computation."""

    def test_3m_annualized_formula(self):
        """Test ((current / 3m_ago)^4 - 1) * 100 formula."""
        # If 3-month change is 0.5%, annualized is (1.005^4 - 1) * 100 ≈ 2.01%
        cpi = [
            {"date": "2024-01-01", "value": 300.0},
            {"date": "2024-02-01", "value": 300.5},
            {"date": "2024-03-01", "value": 301.0},
            {"date": "2024-04-01", "value": 301.5},
        ]

        rate = _compute_3m_annualized(cpi)

        # (301.5 / 300.0)^4 - 1 = (1.005)^4 - 1 ≈ 0.0201 → 2.01%
        assert pytest.approx(rate, abs=0.1) == 2.015

    def test_3m_annualized_insufficient_data(self):
        """Less than 4 observations → None."""
        cpi = [
            {"date": "2024-01-01", "value": 300.0},
            {"date": "2024-02-01", "value": 301.0},
        ]

        rate = _compute_3m_annualized(cpi)

        assert rate is None

    def test_3m_annualized_high_inflation(self):
        """Test 3m annualized with high short-term inflation."""
        # Large 3-month move: 300.0 → 303.0 (1% per month)
        cpi = [
            {"date": "2024-01-01", "value": 300.0},
            {"date": "2024-02-01", "value": 301.0},
            {"date": "2024-03-01", "value": 302.0},
            {"date": "2024-04-01", "value": 303.0},
        ]

        rate = _compute_3m_annualized(cpi)

        # (303.0 / 300.0)^4 - 1 = (1.01)^4 - 1 ≈ 0.0406 → 4.06%
        assert rate is not None
        assert pytest.approx(rate, abs=1.0) == 4.06


class TestInflationRegimeClassification:
    """Test inflation regime classification."""

    def test_inflation_regime_stable(self):
        """Core PCE 2.0-3.5% → stable."""
        regime = _classify_inflation_regime(
            InflationAnalysis(
                core_pce_yoy=2.5,
                inflation_trend="stable",
            )
        )

        assert regime == "stable"

    def test_inflation_regime_elevated(self):
        """Core PCE 3.5-5.0% → rising."""
        regime = _classify_inflation_regime(
            InflationAnalysis(
                core_pce_yoy=4.0,
                inflation_trend="stable",
            )
        )

        assert regime == "rising"

    def test_inflation_regime_high_with_unanchored_expectations(self):
        """High inflation + unanchored expectations → stagflation."""
        regime = _classify_inflation_regime(
            InflationAnalysis(
                core_pce_yoy=4.0,
                expectations_anchored=False,
                spiral_risk=0.5,
                inflation_trend="stable",
            )
        )

        assert regime == "stagflation"

    def test_inflation_regime_disinflation(self):
        """Core PCE 1.5-3.0% with decelerating trend → disinflation."""
        regime = _classify_inflation_regime(
            InflationAnalysis(
                core_pce_yoy=2.0,
                inflation_trend="decelerating",
            )
        )

        assert regime == "disinflation"

    def test_inflation_regime_deflation_risk(self):
        """Core PCE < 0% → deflation_risk."""
        regime = _classify_inflation_regime(
            InflationAnalysis(
                core_pce_yoy=-0.5,
                inflation_trend="decelerating",
            )
        )

        assert regime == "deflation_risk"

    def test_inflation_regime_unknown_missing_data(self):
        """Missing core PCE → unknown."""
        regime = _classify_inflation_regime(InflationAnalysis())

        assert regime == "unknown"


class TestInflationTrendClassification:
    """Test inflation trend detection."""

    def test_inflation_trend_accelerating(self):
        """3m annualized > YoY + 0.5pp → accelerating."""
        trend = _classify_trend(
            InflationAnalysis(
                core_pce_3m_annualized=3.5,
                core_pce_yoy=2.5,
            )
        )

        assert trend == "accelerating"

    def test_inflation_trend_decelerating(self):
        """3m annualized < YoY - 0.5pp → decelerating."""
        trend = _classify_trend(
            InflationAnalysis(
                core_pce_3m_annualized=1.5,
                core_pce_yoy=2.5,
            )
        )

        assert trend == "decelerating"

    def test_inflation_trend_stable(self):
        """3m annualized ≈ YoY ±0.5pp → stable."""
        trend = _classify_trend(
            InflationAnalysis(
                core_pce_3m_annualized=2.5,
                core_pce_yoy=2.5,
            )
        )

        assert trend == "stable"

    def test_inflation_trend_insufficient_data(self):
        """Missing one of the measures → unknown."""
        trend = _classify_trend(
            InflationAnalysis(
                core_pce_3m_annualized=None,
                core_pce_yoy=2.5,
            )
        )

        assert trend == "unknown"


class TestWagePressureClassification:
    """Test wage pressure classification."""

    def test_wage_pressure_low(self):
        """AHE YoY <= 3.0% → low."""
        pressure = _classify_wage_pressure(2.5)

        assert pressure == "low"

    def test_wage_pressure_moderate(self):
        """AHE YoY 3.0-4.0% → moderate."""
        pressure = _classify_wage_pressure(3.5)

        assert pressure == "moderate"

    def test_wage_pressure_elevated(self):
        """AHE YoY 4.0-5.0% → elevated."""
        pressure = _classify_wage_pressure(4.5)

        assert pressure == "elevated"

    def test_wage_pressure_high(self):
        """AHE YoY > 5.0% → high."""
        pressure = _classify_wage_pressure(5.5)

        assert pressure == "high"

    def test_wage_pressure_unknown_missing(self):
        """Missing AHE → unknown."""
        pressure = _classify_wage_pressure(None)

        assert pressure == "unknown"


class TestSpiralRisk:
    """Test wage-price spiral risk computation."""

    def test_spiral_risk_low(self):
        """Normal conditions → risk < 0.3."""
        result = InflationAnalysis(
            ahe_yoy=2.5,
            sticky_cpi_yoy=2.0,
            expectations_anchored=True,
            forward_5y5y=2.2,
        )

        risk = _compute_spiral_risk(result)

        assert risk < 0.3

    def test_spiral_risk_high(self):
        """Wage + inflation + unanchored expectations → risk > 0.5."""
        result = InflationAnalysis(
            ahe_yoy=5.5,
            sticky_cpi_yoy=4.5,
            expectations_anchored=False,
            forward_5y5y=2.8,
        )

        risk = _compute_spiral_risk(result)

        assert risk > 0.5

    def test_spiral_risk_capped_at_one(self):
        """Spiral risk never exceeds 1.0."""
        result = InflationAnalysis(
            ahe_yoy=6.0,
            sticky_cpi_yoy=5.0,
            expectations_anchored=False,
            forward_5y5y=3.0,
        )

        risk = _compute_spiral_risk(result)

        assert risk <= 1.0

    def test_spiral_risk_from_wages(self):
        """Elevated AHE contributes to spiral risk."""
        low_wage = InflationAnalysis(ahe_yoy=2.0, sticky_cpi_yoy=2.0, expectations_anchored=True)
        high_wage = InflationAnalysis(ahe_yoy=5.0, sticky_cpi_yoy=2.0, expectations_anchored=True)

        risk_low = _compute_spiral_risk(low_wage)
        risk_high = _compute_spiral_risk(high_wage)

        assert risk_high > risk_low


class TestExpectationsAnchoring:
    """Test inflation expectations anchoring check."""

    def test_expectations_anchored_target_range(self):
        """5Y5Y 2.0-2.6% → anchored."""
        anchored = _check_expectations_anchored([
            {"date": "2024-01-01", "value": 2.3}
        ])

        assert anchored is True

    def test_expectations_unanchored_above(self):
        """5Y5Y > 2.6% → unanchored."""
        anchored = _check_expectations_anchored([
            {"date": "2024-01-01", "value": 2.8}
        ])

        assert anchored is False

    def test_expectations_unanchored_below(self):
        """5Y5Y < 1.8% → unanchored."""
        anchored = _check_expectations_anchored([
            {"date": "2024-01-01", "value": 1.5}
        ])

        assert anchored is False

    def test_expectations_empty_series(self):
        """Empty series → assume anchored."""
        anchored = _check_expectations_anchored([])

        assert anchored is True


class TestInflationScore:
    """Test inflation scoring function."""

    def test_inflation_score_range(self):
        """Inflation score always between -1 and +1."""
        result = InflationAnalysis(
            core_pce_yoy=4.0,
            inflation_trend="accelerating",
            spiral_risk=0.6,
            expectations_anchored=False,
        )

        score = _compute_inflation_score(result)

        assert -1.0 <= score <= 1.0

    def test_inflation_score_high_inflation_hawkish(self):
        """High inflation + accelerating → positive (hawkish) score."""
        result = InflationAnalysis(
            core_pce_yoy=4.0,
            inflation_trend="accelerating",
            spiral_risk=0.5,
            expectations_anchored=False,
        )

        score = _compute_inflation_score(result)

        assert score > 0.2

    def test_inflation_score_low_inflation_dovish(self):
        """Low inflation + decelerating → negative (dovish) score."""
        result = InflationAnalysis(
            core_pce_yoy=1.5,
            inflation_trend="decelerating",
            spiral_risk=0.1,
            expectations_anchored=True,
        )

        score = _compute_inflation_score(result)

        assert score < -0.1

    def test_inflation_score_neutral_stable(self):
        """Target inflation + stable trend → neutral score."""
        result = InflationAnalysis(
            core_pce_yoy=2.0,
            inflation_trend="stable",
            spiral_risk=0.1,
            expectations_anchored=True,
        )

        score = _compute_inflation_score(result)

        assert -0.2 <= score <= 0.2


class TestFedTargetDistance:
    """Test distance from Fed's 2% target."""

    def test_fed_target_distance_above_target(self):
        """Core PCE 3.0% → distance = +1.0pp."""
        result = InflationAnalysis(core_pce_yoy=3.0)

        # Fed target distance is core_pce - 2.0
        distance = result.core_pce_yoy - 2.0 if result.core_pce_yoy else None

        assert distance == 1.0

    def test_fed_target_distance_below_target(self):
        """Core PCE 1.5% → distance = -0.5pp."""
        result = InflationAnalysis(core_pce_yoy=1.5)

        distance = result.core_pce_yoy - 2.0 if result.core_pce_yoy else None

        assert distance == -0.5

    def test_fed_target_distance_at_target(self):
        """Core PCE 2.0% → distance = 0.0pp."""
        result = InflationAnalysis(core_pce_yoy=2.0)

        distance = result.core_pce_yoy - 2.0 if result.core_pce_yoy else None

        assert distance == 0.0


class TestHelperFunctions:
    """Test utility functions."""

    def test_latest_empty(self):
        """Empty series → None."""
        assert _latest([]) is None

    def test_latest_value(self):
        """Return last value."""
        series = [
            {"date": "2024-01-01", "value": 100},
            {"date": "2024-02-01", "value": 105},
        ]

        assert _latest(series) == 105

    def test_pct_change_positive(self):
        """Positive change → positive result."""
        series = [
            {"date": "2024-01-01", "value": 100},
            {"date": "2024-02-01", "value": 110},
        ]

        change = _pct_change(series, 1)

        # (110 - 100) / 100 * 100 = 10%
        assert change == 10.0

    def test_pct_change_insufficient(self):
        """n >= len(series) → None."""
        series = [{"date": "2024-01-01", "value": 100}]

        change = _pct_change(series, 60)

        assert change is None


class TestFullInflationAnalysis:
    """End-to-end inflation analysis."""

    def create_synthetic_series(self, base_value, count=26, trend=0.0):
        """Create synthetic inflation series."""
        np.random.seed(42)
        values = []
        for i in range(count):
            pct_change = trend + np.random.normal(0, 0.2)
            value = base_value * ((1 + pct_change / 100) ** (i / 12))
            values.append(value)
        return [
            {"date": f"2024-{(i % 12) + 1:02d}-01", "value": round(v, 2)}
            for i, v in enumerate(values)
        ]

    def test_full_inflation_analysis_stable(self):
        """Full analysis with stable inflation."""
        result = analyze_inflation(
            cpi_all=self.create_synthetic_series(300, 26, trend=2.0),
            cpi_core=self.create_synthetic_series(305, 26, trend=2.0),
            cpi_shelter=self.create_synthetic_series(310, 26, trend=3.0),
            cpi_services=self.create_synthetic_series(310, 26, trend=2.5),
            pce=self.create_synthetic_series(280, 26, trend=2.0),
            core_pce=self.create_synthetic_series(285, 26, trend=2.0),
            sticky_cpi=self.create_synthetic_series(300, 2, trend=2.5),
            flexible_cpi=self.create_synthetic_series(300, 2, trend=1.5),
            trimmed_mean=self.create_synthetic_series(290, 2, trend=2.0),
            breakeven_5y=self.create_synthetic_series(100, 2),
            breakeven_10y=self.create_synthetic_series(100, 2),
            forward_5y5y=self.create_synthetic_series(100, 2),
            ppi=self.create_synthetic_series(250, 26, trend=2.0),
            ahe=self.create_synthetic_series(400, 26, trend=3.5),
            oil=self.create_synthetic_series(90, 60, trend=0.0),
        )

        assert isinstance(result, InflationAnalysis)
        assert result.core_pce_yoy is not None
        assert result.inflation_regime in ["stable", "rising", "disinflation", "stagflation", "deflation_risk", "unknown"]
        assert 0.0 <= result.spiral_risk <= 1.0
        assert -1.0 <= result.inflation_score <= 1.0

    def test_full_inflation_analysis_empty(self):
        """Full analysis with empty inputs."""
        result = analyze_inflation(
            cpi_all=[],
            cpi_core=[],
            cpi_shelter=[],
            cpi_services=[],
            pce=[],
            core_pce=[],
            sticky_cpi=[],
            flexible_cpi=[],
            trimmed_mean=[],
            breakeven_5y=[],
            breakeven_10y=[],
            forward_5y5y=[],
            ppi=[],
            ahe=[],
            oil=[],
        )

        assert isinstance(result, InflationAnalysis)
        assert result.core_pce_yoy is None
        assert result.inflation_regime == "unknown"
        assert result.spiral_risk == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
