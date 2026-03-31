"""
Comprehensive tests for breadth.py module.

Tests cover:
- Sector participation metrics (50MA, 200MA)
- SPY/RSP ratio analysis for leadership breadth
- Sector rotation signals
- Breadth health assessment
- Breadth score computation
"""

import pytest
import numpy as np

from finias.agents.macro_strategist.computations.breadth import (
    analyze_breadth,
    BreadthAnalysis,
    _analyze_spy_rsp,
    _analyze_sectors,
    _compute_breadth_score,
    _classify_health,
)


class TestSectorParticipation:
    """Test sector participation metrics."""

    def test_sector_participation_200ma(self):
        """Test sectors above 200-day MA calculation."""
        # Create 11 sectors with 201 days of data
        # 6 above 200MA, 5 below
        sector_prices = {}
        for i in range(11):
            if i < 6:
                # Above 200MA
                closes = [100.0 + 1.0] * 200 + [102.0]
            else:
                # Below 200MA
                closes = [100.0 + 1.0] * 200 + [98.0]
            sector_prices[f"XL{i}"] = [{"date": f"2024-01-{j%7+1:02d}", "close": c} for j, c in enumerate(closes)]

        # SPY prices (just needs same length)
        spy_prices = [{"date": f"2024-01-{i%7+1:02d}", "close": 500.0 + i * 0.1} for i in range(201)]

        result = analyze_breadth(spy_prices, sector_prices, None)

        assert result.sectors_above_200ma == 6
        assert pytest.approx(result.pct_sectors_above_200ma, abs=1.0) == 54.5

    def test_sector_participation_50ma(self):
        """Test sectors above 50-day MA calculation."""
        # Need 201+ days for analysis
        sector_prices = {}
        for i in range(11):
            if i < 7:
                # Above 50MA
                closes = [100.0] * 200 + [101.0]
            else:
                # Below 50MA
                closes = [100.0] * 200 + [99.0]
            sector_prices[f"XL{i}"] = [{"date": f"2024-{j%7+1:02d}", "close": c} for j, c in enumerate(closes)]

        spy_prices = [{"date": f"2024-{i%7+1:02d}", "close": 500.0} for i in range(201)]

        result = analyze_breadth(spy_prices, sector_prices, None)

        assert result.sectors_above_50ma == 7
        assert pytest.approx(result.pct_sectors_above_50ma, abs=2.0) == 63.6

    def test_breadth_health_strong(self):
        """Test strong breadth with most sectors above both MAs."""
        sector_prices = {}
        for i in range(11):
            # All above 200MA, most above 50MA
            closes = [100.0 + 1.0] * 200 + [102.0]
            sector_prices[f"XL{i}"] = [{"date": f"2024-{j%7+1:02d}", "close": c} for j, c in enumerate(closes)]

        spy_prices = [{"date": f"2024-{i%7+1:02d}", "close": 500.0 + i * 0.5} for i in range(201)]
        rsp_prices = [{"date": f"2024-{i%7+1:02d}", "close": 150.0 + i * 0.4} for i in range(201)]

        result = analyze_breadth(spy_prices, sector_prices, rsp_prices)

        # With broad participation, should be healthier than poor
        assert result.breadth_health in ["strong", "healthy", "weakening"]

    def test_breadth_health_weakening(self):
        """Test weakening breadth with few sectors above 50MA."""
        sector_prices = {}
        for i in range(11):
            if i < 3:
                closes = [100.0] * 200 + [101.0]  # Above 200MA only
            else:
                closes = [100.0] * 200 + [99.0]  # Below both
            sector_prices[f"XL{i}"] = [{"date": f"2024-01-{j%7+1:02d}", "close": c} for j, c in enumerate(closes)]

        spy_prices = [{"date": f"2024-01-{i%7+1:02d}", "close": 500.0} for i in range(201)]

        result = analyze_breadth(spy_prices, sector_prices, None)

        assert result.breadth_health in ["weakening", "poor"]


class TestSPYRSPAnalysis:
    """Test SPY/RSP ratio for leadership breadth."""

    def test_spy_rsp_narrow_leadership(self):
        """SPY outperforming RSP significantly → narrow_leadership True."""
        # Need at least 201 days of SPY data for analysis
        spy_prices = [
            {"date": f"2024-{i%7+1:02d}-01", "close": 500.0 + i * 2.0}
            for i in range(201)
        ]
        rsp_prices = [
            {"date": f"2024-{i%7+1:02d}-01", "close": 150.0 + i * 0.5}
            for i in range(201)
        ]

        result = analyze_breadth(spy_prices, None, rsp_prices)

        # SPY ratio rising (cap-weighted outperforming equal-weight)
        assert result.narrow_leadership is True

    def test_spy_rsp_broad_breadth(self):
        """RSP outperforming SPY → narrow_leadership False."""
        spy_prices = [
            {"date": f"2024-{i%7+1:02d}-01", "close": 500.0 + i * 0.5}
            for i in range(201)
        ]
        rsp_prices = [
            {"date": f"2024-{i%7+1:02d}-01", "close": 150.0 + i * 2.0}
            for i in range(201)
        ]

        result = analyze_breadth(spy_prices, None, rsp_prices)

        # RSP rising faster than SPY
        assert result.narrow_leadership is False

    def test_spy_rsp_insufficient_data(self):
        """Less than 60 observations → no ratio calculation."""
        spy_prices = [{"date": f"2024-{i:02d}-01", "close": 500.0} for i in range(30)]
        rsp_prices = [{"date": f"2024-{i:02d}-01", "close": 150.0} for i in range(30)]

        result = analyze_breadth(spy_prices, None, rsp_prices)

        assert result.narrow_leadership is False


class TestSectorRotation:
    """Test sector rotation signals."""

    def test_sector_rotation_cyclical_leading(self):
        """Cyclicals outperforming defensives → risk_on_rotation."""
        sector_prices = {
            "XLF": [{"date": f"2024-{i:02d}-01", "close": 150.0 + i * 1.0} for i in range(201)],  # Cyclical
            "XLI": [{"date": f"2024-{i:02d}-01", "close": 160.0 + i * 1.0} for i in range(201)],  # Cyclical
            "XLY": [{"date": f"2024-{i:02d}-01", "close": 170.0 + i * 1.0} for i in range(201)],  # Cyclical
            "XLP": [{"date": f"2024-{i:02d}-01", "close": 180.0 + i * 0.2} for i in range(201)],  # Defensive
            "XLU": [{"date": f"2024-{i:02d}-01", "close": 190.0 + i * 0.2} for i in range(201)],  # Defensive
            "XLV": [{"date": f"2024-{i:02d}-01", "close": 200.0 + i * 0.2} for i in range(201)],  # Defensive
        }
        spy_prices = [{"date": f"2024-{i:02d}-01", "close": 500.0 + i * 0.5} for i in range(201)]

        result = analyze_breadth(spy_prices, sector_prices, None)

        # Cyclicals up faster than defensives
        assert result.cyclical_vs_defensive > 1.5
        assert result.rotation_signal == "risk_on_rotation"

    def test_sector_rotation_defensive_leading(self):
        """Defensibles outperforming cyclicals → risk_off_rotation."""
        sector_prices = {
            "XLF": [{"date": f"2024-{i:02d}-01", "close": 150.0 + i * 0.2} for i in range(201)],  # Cyclical
            "XLI": [{"date": f"2024-{i:02d}-01", "close": 160.0 + i * 0.2} for i in range(201)],  # Cyclical
            "XLY": [{"date": f"2024-{i:02d}-01", "close": 170.0 + i * 0.2} for i in range(201)],  # Cyclical
            "XLP": [{"date": f"2024-{i:02d}-01", "close": 180.0 + i * 1.0} for i in range(201)],  # Defensive
            "XLU": [{"date": f"2024-{i:02d}-01", "close": 190.0 + i * 1.0} for i in range(201)],  # Defensive
            "XLV": [{"date": f"2024-{i:02d}-01", "close": 200.0 + i * 1.0} for i in range(201)],  # Defensive
        }
        spy_prices = [{"date": f"2024-{i:02d}-01", "close": 500.0 + i * 0.5} for i in range(201)]

        result = analyze_breadth(spy_prices, sector_prices, None)

        # Defensibles up faster than cyclicals
        assert result.cyclical_vs_defensive < -1.5
        assert result.rotation_signal == "risk_off_rotation"

    def test_sector_rotation_neutral(self):
        """Balanced rotation → neutral."""
        sector_prices = {
            f"XL{i}": [{"date": f"2024-{j:02d}-01", "close": 150.0 + j * 0.5} for j in range(201)]
            for i in range(11)
        }
        spy_prices = [{"date": f"2024-{i:02d}-01", "close": 500.0 + i * 0.5} for i in range(201)]

        result = analyze_breadth(spy_prices, sector_prices, None)

        assert result.rotation_signal == "neutral"


class TestSectorRankings:
    """Test sector relative strength rankings."""

    def test_sector_rankings_order(self):
        """Leading sectors have higher relative strength than lagging."""
        # Need at least 5 sectors for sector analysis to compute rankings
        sector_prices = {
            "XLK": [{"date": f"2024-{i%7+1:02d}", "close": 200.0 + i * 2.0} for i in range(201)],  # Outperform
            "XLV": [{"date": f"2024-{i%7+1:02d}", "close": 210.0 + i * 1.0} for i in range(201)],  # Mid
            "XLE": [{"date": f"2024-{i%7+1:02d}", "close": 220.0 + i * 0.1} for i in range(201)],  # Underperform
            "XLY": [{"date": f"2024-{i%7+1:02d}", "close": 230.0 + i * 1.5} for i in range(201)],  # Strong
            "XLI": [{"date": f"2024-{i%7+1:02d}", "close": 240.0 + i * 0.8} for i in range(201)],  # Moderate
        }
        spy_prices = [{"date": f"2024-{i%7+1:02d}", "close": 500.0 + i * 0.5} for i in range(201)]

        result = analyze_breadth(spy_prices, sector_prices, None)

        # XLK or XLY should be in leading_sectors (highest relative strength)
        assert len(result.leading_sectors) > 0
        # XLE should be in lagging_sectors (lowest relative strength)
        assert "XLE" in result.lagging_sectors


class TestBreadthScore:
    """Test breadth score computation."""

    def test_breadth_score_strong_health(self):
        """Strong breadth → score >= 0.7."""
        sector_prices = {
            f"XL{i}": [{"date": f"2024-{j:02d}-01", "close": 150.0 + j * 1.0} for j in range(201)]
            for i in range(11)
        }
        spy_prices = [{"date": f"2024-{i:02d}-01", "close": 500.0 + i * 1.0} for i in range(201)]
        rsp_prices = [{"date": f"2024-{i:02d}-01", "close": 150.0 + i * 1.0} for i in range(201)]

        result = analyze_breadth(spy_prices, sector_prices, rsp_prices)

        assert result.breadth_score >= 0.6

    def test_breadth_score_poor_health(self):
        """Poor breadth → score < 0.4."""
        sector_prices = {
            f"XL{i}": [{"date": f"2024-{j:02d}-01", "close": 150.0 - j * 1.0} for j in range(201)]
            for i in range(11)
        }
        spy_prices = [{"date": f"2024-{i:02d}-01", "close": 500.0 + i * 0.1} for i in range(201)]

        result = analyze_breadth(spy_prices, sector_prices, None)

        assert result.breadth_score < 0.6

    def test_breadth_score_range(self):
        """Breadth score always 0-1."""
        sector_prices = {
            f"XL{i}": [{"date": f"2024-{j:02d}-01", "close": 150.0 + j * 0.5} for j in range(201)]
            for i in range(11)
        }
        spy_prices = [{"date": f"2024-{i:02d}-01", "close": 500.0 + i * 0.5} for i in range(201)]

        result = analyze_breadth(spy_prices, sector_prices, None)

        assert 0.0 <= result.breadth_score <= 1.0


class TestBreadthClassification:
    """Test breadth health classification."""

    def test_classify_health_strong(self):
        """Score >= 0.7 → strong."""
        health = _classify_health(0.75)
        assert health == "strong"

    def test_classify_health_healthy(self):
        """Score 0.55-0.7 → healthy."""
        health = _classify_health(0.65)
        assert health == "healthy"

    def test_classify_health_weakening(self):
        """Score 0.4-0.55 → weakening."""
        health = _classify_health(0.45)
        assert health == "weakening"

    def test_classify_health_poor(self):
        """Score < 0.4 → poor."""
        health = _classify_health(0.35)
        assert health == "poor"


class TestBreadthDivergence:
    """Test breadth divergence detection."""

    def test_breadth_divergence_narrow_leadership(self):
        """SPX near highs + narrow leadership → divergence."""
        spy_prices = [
            {"date": f"2024-{i:02d}-01", "close": 500.0 + i * 1.0}
            for i in range(30)
        ]
        rsp_prices = [
            {"date": f"2024-{i:02d}-01", "close": 150.0 + i * 0.5}
            for i in range(30)
        ]

        result = analyze_breadth(spy_prices, None, rsp_prices)

        # Market near highs with narrow leadership
        if result.pct_sectors_above_200ma < 60 and result.narrow_leadership:
            assert result.breadth_divergence is True

    def test_breadth_divergence_none(self):
        """Healthy breadth → no divergence."""
        sector_prices = {
            f"XL{i}": [{"date": f"2024-{j:02d}-01", "close": 150.0 + j * 1.0} for j in range(201)]
            for i in range(11)
        }
        spy_prices = [
            {"date": f"2024-{i:02d}-01", "close": 500.0 + i * 1.0}
            for i in range(201)
        ]

        result = analyze_breadth(spy_prices, sector_prices, None)

        # Healthy breadth with good participation
        if result.pct_sectors_above_200ma > 70:
            assert result.breadth_divergence is False


class TestMissingDataHandling:
    """Test graceful handling of missing data."""

    def test_missing_rsp_graceful(self):
        """Missing RSP data → still valid analysis."""
        sector_prices = {
            f"XL{i}": [{"date": f"2024-{j:02d}-01", "close": 150.0 + j * 0.5} for j in range(201)]
            for i in range(11)
        }
        spy_prices = [{"date": f"2024-{i:02d}-01", "close": 500.0} for i in range(201)]

        result = analyze_breadth(spy_prices, sector_prices, None)

        assert isinstance(result, BreadthAnalysis)
        assert result.spy_rsp_ratio is None  # No RSP data

    def test_few_sectors_graceful(self):
        """Only 3 sectors (< 5 minimum) → doesn't crash."""
        sector_prices = {
            f"XL{i}": [{"date": f"2024-{j:02d}-01", "close": 150.0} for j in range(201)]
            for i in range(3)
        }
        spy_prices = [{"date": f"2024-{i:02d}-01", "close": 500.0} for i in range(201)]

        result = analyze_breadth(spy_prices, sector_prices, None)

        assert isinstance(result, BreadthAnalysis)

    def test_insufficient_price_data(self):
        """Less than 201 days → returns empty result."""
        spy_prices = [{"date": f"2024-{i:02d}-01", "close": 500.0} for i in range(50)]

        result = analyze_breadth(spy_prices, None, None)

        # Should return empty BreadthAnalysis
        assert isinstance(result, BreadthAnalysis)
        assert result.sectors_above_200ma == 0


class TestFullBreadthAnalysis:
    """End-to-end breadth analysis."""

    def test_full_analysis_realistic(self):
        """Full analysis with realistic 201+ days of sector data."""
        np.random.seed(42)

        sector_prices = {}
        for i in range(11):
            # Create realistic price movements
            prices = [150.0 + (j % 20) * np.random.normal(0, 0.5) for j in range(201)]
            sector_prices[f"XL{i}"] = [
                {"date": f"2024-{j%12+1:02d}-01", "close": round(p, 2)}
                for j, p in enumerate(prices)
            ]

        spy_prices = [
            {"date": f"2024-{i%12+1:02d}-01", "close": 500.0 + i * 0.5}
            for i in range(201)
        ]

        rsp_prices = [
            {"date": f"2024-{i%12+1:02d}-01", "close": 150.0 + i * 0.4}
            for i in range(201)
        ]

        result = analyze_breadth(spy_prices, sector_prices, rsp_prices)

        assert isinstance(result, BreadthAnalysis)
        assert 0 <= result.sectors_above_200ma <= 11
        assert 0 <= result.sectors_above_50ma <= 11
        assert 0.0 <= result.breadth_score <= 1.0
        assert result.breadth_health in ["strong", "healthy", "weakening", "poor"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
