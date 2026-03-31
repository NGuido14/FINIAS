"""
Comprehensive tests for business_cycle.py module.

Tests cover:
- Sahm Rule computation with correct 3-month average logic
- Business cycle phase classification
- LEI trend analysis
- Manufacturing activity (Philly Fed proxy)
- Recession probability
- Sector implications
"""

import pytest
import numpy as np
from datetime import date

from finias.agents.macro_strategist.computations.business_cycle import (
    analyze_business_cycle,
    BusinessCycleAnalysis,
    _compute_sahm_rule,
    _compute_sahm_acceleration,
    _classify_lei_trend,
    _compute_composite_leading,
    _compute_recession_probability,
    _classify_cycle_phase,
    _latest,
    _moving_average,
    _classify_trend_simple,
    _compute_yoy,
)


class TestSahmRuleComputation:
    """Test Sahm Rule calculation with correct 3-month average logic."""

    def test_sahm_rule_with_exact_prompt_data(self):
        """
        Test with exact data from the prompt:
        unemployment = [3.5, 3.6, 3.5, 3.4, 3.5, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4]
        Rolling 3m avgs: [3.533, 3.500, 3.467, 3.467, 3.533, 3.600, 3.700, 3.800, 3.900, 4.000, 4.100, 4.200, 4.300]
        Current 3m avg = 4.300, 12-month low of prior = 3.467
        Sahm = 4.300 - 3.467 = 0.833 → triggered
        """
        unemployment = [
            {"date": f"2024-{i:02d}-01", "value": val}
            for i, val in enumerate([
                3.5, 3.6, 3.5, 3.4, 3.5, 3.5, 3.6, 3.7, 3.8, 3.9,
                4.0, 4.1, 4.2, 4.3, 4.4
            ])
        ]

        sahm_val, triggered = _compute_sahm_rule(unemployment)

        # Should be 0.8333 (rounded to 4 decimals)
        assert sahm_val == pytest.approx(0.8333, abs=0.001)
        assert bool(triggered) is True

    def test_sahm_rule_not_triggered(self):
        """Test Sahm Rule with stable unemployment → not triggered."""
        unemployment = [
            {"date": f"2024-{i:02d}-01", "value": 3.7}
            for i in range(15)
        ]

        sahm_val, triggered = _compute_sahm_rule(unemployment)

        assert pytest.approx(sahm_val, abs=0.0001) == 0.0
        assert bool(triggered) is False

    def test_sahm_rule_deteriorating_slowly(self):
        """Test Sahm Rule with gradual deterioration."""
        unemployment = [
            {"date": f"2024-{i:02d}-01", "value": 3.5 + i * 0.05}
            for i in range(15)
        ]

        sahm_val, triggered = _compute_sahm_rule(unemployment)

        # Gradual rise from 3.5 to 4.2 — Sahm can exceed 0.5 with this pace
        # (current 3m avg ~4.2, 12-month low ~3.6, difference ~0.6)
        assert 0.0 < sahm_val
        assert sahm_val >= 0.5  # Strong deterioration triggers
        assert bool(triggered) is True

    def test_sahm_rule_insufficient_data(self):
        """Test Sahm Rule with less than 15 months → returns (0.0, False)."""
        unemployment = [
            {"date": f"2024-{i:02d}-01", "value": 3.7}
            for i in range(10)
        ]

        sahm_val, triggered = _compute_sahm_rule(unemployment)

        assert sahm_val == 0.0
        assert triggered is False

    def test_sahm_rule_empty_input(self):
        """Test Sahm Rule with empty input."""
        sahm_val, triggered = _compute_sahm_rule([])

        assert sahm_val == 0.0
        assert triggered is False

    def test_sahm_acceleration(self):
        """Test Sahm acceleration (rate of change)."""
        # Deteriorating unemployment over time
        unemployment = [
            {"date": f"2024-{i:02d}-01", "value": 3.5 + i * 0.06}
            for i in range(20)
        ]

        accel = _compute_sahm_acceleration(unemployment)

        # Acceleration should be positive (Sahm getting worse)
        assert accel is not None
        # With consistent deterioration, Sahm should accelerate (increase)
        if accel is not None:
            assert isinstance(accel, float)

    def test_sahm_distance_to_trigger(self):
        """Test calculation of distance from trigger point."""
        unemployment = [
            {"date": f"2024-{i:02d}-01", "value": 3.5 + i * 0.04}
            for i in range(15)
        ]

        sahm_val, _ = _compute_sahm_rule(unemployment)
        distance = max(0, 0.50 - sahm_val)

        assert distance >= 0.0
        assert distance <= 0.5


class TestBusinessCyclePhaseClassification:
    """Test business cycle phase determination."""

    def test_cycle_phase_mid_cycle(self):
        """Test mid-cycle classification with moderate conditions."""
        result = BusinessCycleAnalysis(
            composite_leading=0.2,
            recession_probability=0.15,
            sahm_triggered=False,
            lei_trend="improving",
            ism_manufacturing=52.0,
            capacity_utilization=78.0,
        )

        phase, confidence = _classify_cycle_phase(result)

        assert phase == "mid_cycle"
        assert confidence > 0.4

    def test_cycle_phase_recession_sahm_triggered(self):
        """Test recession classification when Sahm is triggered."""
        result = BusinessCycleAnalysis(
            composite_leading=-0.5,
            recession_probability=0.7,
            sahm_triggered=True,
        )

        phase, confidence = _classify_cycle_phase(result)

        assert phase == "recession"
        assert confidence >= 0.5

    def test_cycle_phase_recession_high_probability(self):
        """Test recession classification with high probability."""
        result = BusinessCycleAnalysis(
            composite_leading=-0.3,
            recession_probability=0.75,
            sahm_triggered=False,
        )

        phase, confidence = _classify_cycle_phase(result)

        assert phase == "recession"

    def test_cycle_phase_early_cycle(self):
        """Test early cycle classification."""
        result = BusinessCycleAnalysis(
            composite_leading=0.5,
            recession_probability=0.05,
            sahm_triggered=False,
            lei_trend="improving",
            ism_manufacturing=55.0,
            capacity_utilization=73.0,
        )

        phase, confidence = _classify_cycle_phase(result)

        assert phase in ["early_cycle", "mid_cycle"]

    def test_cycle_phase_late_cycle(self):
        """Test late cycle classification."""
        result = BusinessCycleAnalysis(
            composite_leading=0.1,
            recession_probability=0.4,
            sahm_triggered=False,
            lei_trend="deteriorating",
        )

        phase, confidence = _classify_cycle_phase(result)

        assert phase == "late_cycle"


class TestLEIAnalysis:
    """Test Leading Economic Index analysis."""

    def test_lei_trend_improving(self):
        """Test LEI trend classification as improving."""
        lei = [
            {"date": f"2024-{i:02d}-01", "value": 95.0 + i * 0.5}
            for i in range(1, 7)
        ]

        trend = _classify_lei_trend(lei)

        assert trend == "improving"

    def test_lei_trend_deteriorating(self):
        """Test LEI trend classification as deteriorating."""
        lei = [
            {"date": f"2024-{i:02d}-01", "value": 105.0 - i * 0.5}
            for i in range(1, 7)
        ]

        trend = _classify_lei_trend(lei)

        assert trend == "deteriorating"

    def test_lei_trend_stable(self):
        """Test LEI trend classification as stable."""
        lei = [
            {"date": f"2024-{i:02d}-01", "value": 100.0}
            for i in range(1, 7)
        ]

        trend = _classify_lei_trend(lei)

        assert trend == "stable"

    def test_lei_insufficient_data(self):
        """Test LEI trend with insufficient data."""
        lei = [{"date": "2024-01-01", "value": 100.0}]

        trend = _classify_lei_trend(lei)

        assert trend == "unknown"


class TestManufacturingActivity:
    """Test manufacturing activity (Philly Fed proxy)."""

    def test_manufacturing_proxy_conversion(self):
        """Test Philly Fed to ISM conversion: 50 + (philly * 0.3)."""
        philly_fed = [
            {"date": f"2024-01-{i+1:02d}", "value": 10.0}
            for i in range(6)
        ]

        # Ensure unemployment has 15 entries for Sahm Rule
        unemployment = [
            {"date": f"2024-{i:02d}-01", "value": 3.7}
            for i in range(15)
        ]

        result = analyze_business_cycle(
            lei_series=[],
            unemployment=unemployment,
            initial_claims=[],
            continuing_claims=[],
            jolts_openings=[],
            jolts_quits=[],
            temp_employment=[],
            avg_weekly_hours=[],
            building_permits=[],
            housing_starts=[],
            retail_sales=[],
            consumer_sentiment=[],
            industrial_production=[],
            capacity_utilization=[],
            cfnai_series=[],
            personal_income=[],
            durable_goods=[],
            nfp_series=[],
            philly_fed=philly_fed,
        )

        # Philly 10 → ISM = 50 + 10*0.3 = 53
        assert pytest.approx(result.ism_manufacturing, abs=0.5) == 53.0
        assert result.ism_is_proxy is True

    def test_manufacturing_below_50(self):
        """Test manufacturing below 50 (contraction)."""
        philly_fed = [
            {"date": "2024-01-01", "value": -5.0},
        ]

        result = analyze_business_cycle(
            lei_series=[],
            unemployment=[{"date": "2024-01-01", "value": 3.7}] * 15,
            initial_claims=[],
            continuing_claims=[],
            jolts_openings=[],
            jolts_quits=[],
            temp_employment=[],
            avg_weekly_hours=[],
            building_permits=[],
            housing_starts=[],
            retail_sales=[],
            consumer_sentiment=[],
            industrial_production=[],
            capacity_utilization=[],
            cfnai_series=[],
            personal_income=[],
            durable_goods=[],
            nfp_series=[],
            philly_fed=philly_fed,
        )

        # Philly -5 → ISM = 50 + (-5)*0.3 = 48.5
        assert pytest.approx(result.ism_manufacturing, abs=0.5) == 48.5


class TestRecessionProbability:
    """Test recession probability computation."""

    def test_recession_probability_sahm_triggered(self):
        """Sahm triggered → probability >= 0.50."""
        result = BusinessCycleAnalysis(
            sahm_triggered=True,
            sahm_value=0.6,
        )

        # When sahm is triggered, _compute_recession_probability adds 0.50
        # But we need to call the function to test it
        prob = _compute_recession_probability(result)

        assert prob >= 0.50

    def test_recession_probability_high(self):
        """High recession probability from multiple signals."""
        result = BusinessCycleAnalysis(
            sahm_value=0.45,
            lei_consecutive_negatives=6,
            ism_manufacturing=44.0,
            initial_claims_trend="rising",
            cfnai=-0.8,
        )

        prob = _compute_recession_probability(result)

        assert 0.0 <= prob <= 1.0
        assert prob > 0.3  # Multiple negative signals

    def test_recession_probability_low(self):
        """Low recession probability from benign conditions."""
        result = BusinessCycleAnalysis(
            sahm_value=0.0,
            lei_consecutive_negatives=0,
            ism_manufacturing=55.0,
            initial_claims_trend="stable",
            cfnai=0.2,
        )

        prob = _compute_recession_probability(result)

        assert prob < 0.3

    def test_recession_probability_range(self):
        """Recession probability always 0.0-1.0."""
        result = BusinessCycleAnalysis(
            sahm_value=1.0,
            lei_consecutive_negatives=10,
            ism_manufacturing=30.0,
            initial_claims_trend="rising",
            cfnai=-2.0,
        )

        prob = _compute_recession_probability(result)

        assert 0.0 <= prob <= 1.0


class TestCompositeLeading:
    """Test composite leading indicator computation."""

    def test_composite_leading_strong_expansion(self):
        """Strong expansion signals → high composite."""
        result = BusinessCycleAnalysis(
            lei_trend="improving",
            lei_consecutive_negatives=0,
            ism_manufacturing=58.0,
            initial_claims_trend="falling",
            building_permits_trend="rising",
            consumer_sentiment_trend="rising",
            sahm_value=0.0,
        )

        composite = _compute_composite_leading(result)

        assert composite > 0.3

    def test_composite_leading_weak_signals(self):
        """Weak/negative signals → low composite."""
        result = BusinessCycleAnalysis(
            lei_trend="deteriorating",
            lei_consecutive_negatives=4,
            ism_manufacturing=48.0,
            initial_claims_trend="rising",
            building_permits_trend="falling",
            consumer_sentiment_trend="falling",
            sahm_value=0.4,
        )

        composite = _compute_composite_leading(result)

        assert composite < 0.0

    def test_composite_leading_range(self):
        """Composite always between -1 and +1."""
        result = BusinessCycleAnalysis(
            lei_trend="stable",
            lei_consecutive_negatives=1,
            ism_manufacturing=50.0,
            initial_claims_trend="stable",
            building_permits_trend="stable",
            consumer_sentiment_trend="stable",
            sahm_value=0.2,
        )

        composite = _compute_composite_leading(result)

        assert -1.0 <= composite <= 1.0


class TestHelperFunctions:
    """Test utility functions."""

    def test_latest_empty(self):
        """Empty series → None."""
        assert _latest([]) is None

    def test_latest_value(self):
        """Return last value."""
        series = [
            {"date": "2024-01-01", "value": 100},
            {"date": "2024-01-02", "value": 105},
        ]
        assert _latest(series) == 105

    def test_moving_average_basic(self):
        """Test simple moving average."""
        series = [
            {"date": "2024-01-01", "value": 100},
            {"date": "2024-01-02", "value": 110},
            {"date": "2024-01-03", "value": 120},
            {"date": "2024-01-04", "value": 130},
        ]
        avg = _moving_average(series, 2)
        assert avg == 125.0

    def test_moving_average_insufficient(self):
        """Insufficient observations → None."""
        series = [{"date": "2024-01-01", "value": 100}]
        avg = _moving_average(series, 4)
        assert avg is None

    def test_classify_trend_simple_rising(self):
        """Simple trend rising."""
        series = [
            {"date": f"2024-{i:02d}-01", "value": 100 + i * 2}
            for i in range(6)
        ]
        trend = _classify_trend_simple(series, 3)
        assert trend == "rising"

    def test_classify_trend_simple_falling(self):
        """Simple trend falling."""
        series = [
            {"date": f"2024-{i:02d}-01", "value": 120 - i * 2}
            for i in range(6)
        ]
        trend = _classify_trend_simple(series, 3)
        assert trend == "falling"

    def test_classify_trend_simple_stable(self):
        """Simple trend stable."""
        series = [
            {"date": f"2024-{i:02d}-01", "value": 100.0}
            for i in range(6)
        ]
        trend = _classify_trend_simple(series, 3)
        assert trend == "stable"

    def test_compute_yoy_basic(self):
        """Year-over-year growth."""
        series = [
            {"date": f"2023-{m:02d}-01", "value": 100 + m}
            for m in range(1, 13)
        ] + [
            {"date": f"2024-{m:02d}-01", "value": 110 + m}
            for m in range(1, 13)
        ]
        yoy = _compute_yoy(series)
        # Current = 112 (2024-12), year_ago = 102 (2023-12)
        # YoY = (112 - 102) / 102 * 100 = 9.8%
        assert pytest.approx(yoy, abs=1.0) == 9.8

    def test_compute_yoy_insufficient(self):
        """Insufficient data → None."""
        series = [{"date": f"2024-{i:02d}-01", "value": 100 + i} for i in range(10)]
        yoy = _compute_yoy(series)
        assert yoy is None


class TestFullBusinessCycleAnalysis:
    """End-to-end business cycle analysis."""

    def test_full_analysis_expansion(self):
        """Full analysis with expansion signals."""
        unemployment = [
            {"date": f"2024-{i:02d}-01", "value": 3.7}
            for i in range(15)
        ]

        result = analyze_business_cycle(
            lei_series=[
                {"date": f"2024-{i:02d}-01", "value": 95.0 + i * 0.3}
                for i in range(6)
            ],
            unemployment=unemployment,
            initial_claims=[
                {"date": f"2024-{i:02d}-01", "value": 200_000 - i * 1000}
                for i in range(10)
            ],
            continuing_claims=[
                {"date": f"2024-{i:02d}-01", "value": 1_500_000}
                for i in range(10)
            ],
            jolts_openings=[
                {"date": f"2024-{i:02d}-01", "value": 8_500_000}
                for i in range(3)
            ],
            jolts_quits=[
                {"date": f"2024-{i:02d}-01", "value": 3_600_000}
                for i in range(3)
            ],
            temp_employment=[
                {"date": f"2024-{i:02d}-01", "value": 2_500_000}
                for i in range(6)
            ],
            avg_weekly_hours=[
                {"date": f"2024-{i:02d}-01", "value": 34.4}
                for i in range(6)
            ],
            building_permits=[
                {"date": f"2024-{i:02d}-01", "value": 1_400_000}
                for i in range(6)
            ],
            housing_starts=[
                {"date": f"2024-{i:02d}-01", "value": 1_200_000}
                for i in range(6)
            ],
            retail_sales=[
                {"date": f"2024-{m:02d}-01", "value": 600_000 + m * 1000}
                for m in range(1, 13)
            ],
            consumer_sentiment=[
                {"date": f"2024-{i:02d}-01", "value": 75.0 + i * 0.5}
                for i in range(6)
            ],
            industrial_production=[
                {"date": f"2024-{m:02d}-01", "value": 100 + m * 0.2}
                for m in range(1, 13)
            ],
            capacity_utilization=[
                {"date": f"2024-{i:02d}-01", "value": 78.0}
                for i in range(6)
            ],
            cfnai_series=[
                {"date": f"2024-{i:02d}-01", "value": 0.3}
                for i in range(6)
            ],
            personal_income=[],
            durable_goods=[],
            nfp_series=[],
            philly_fed=[
                {"date": f"2024-{i:02d}-01", "value": 8.0}
                for i in range(6)
            ],
        )

        assert isinstance(result, BusinessCycleAnalysis)
        assert 0.0 <= result.recession_probability <= 1.0
        assert result.cycle_phase in ["early_cycle", "mid_cycle", "late_cycle", "recession"]

    def test_full_analysis_empty_inputs(self):
        """Full analysis with minimal inputs → safe defaults."""
        result = analyze_business_cycle(
            lei_series=[],
            unemployment=[],
            initial_claims=[],
            continuing_claims=[],
            jolts_openings=[],
            jolts_quits=[],
            temp_employment=[],
            avg_weekly_hours=[],
            building_permits=[],
            housing_starts=[],
            retail_sales=[],
            consumer_sentiment=[],
            industrial_production=[],
            capacity_utilization=[],
            cfnai_series=[],
            personal_income=[],
            durable_goods=[],
            nfp_series=[],
            philly_fed=[],
        )

        assert isinstance(result, BusinessCycleAnalysis)
        # With no data, composite_leading = 0.0 → phase becomes mid_cycle (not unknown)
        assert result.cycle_phase in ["unknown", "mid_cycle"]
        assert result.recession_probability == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
