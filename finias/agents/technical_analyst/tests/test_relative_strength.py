"""Tests for the relative strength module."""

import pytest
import numpy as np
import pandas as pd
from finias.agents.technical_analyst.computations.relative_strength import (
    analyze_relative_strength, RelativeStrengthAnalysis,
    compute_universe_returns,
)


def _make_df(n=250, start=100.0, daily_return=0.001):
    np.random.seed(42)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + daily_return + np.random.normal(0, 0.005)))
    prices = np.array(prices)
    return pd.DataFrame({
        "open": prices * 0.999, "high": prices * 1.005,
        "low": prices * 0.995, "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    })


def _make_outperformer(n=250):
    return _make_df(n=n, daily_return=0.003)

def _make_underperformer(n=250):
    return _make_df(n=n, daily_return=-0.001)

def _make_benchmark(n=250):
    return _make_df(n=n, daily_return=0.001)


class TestRSvsSector:
    def test_rs_computed(self):
        result = analyze_relative_strength(
            _make_outperformer(), symbol="TEST", sector="Tech",
            sector_etf_df=_make_benchmark(),
        )
        assert result.rs_vs_sector is not None
        assert result.rs_vs_sector > 1.0  # Outperformer should have RS > 1

    def test_underperformer_has_low_rs(self):
        result = analyze_relative_strength(
            _make_underperformer(), symbol="TEST", sector="Tech",
            sector_etf_df=_make_benchmark(),
        )
        if result.rs_vs_sector is not None:
            assert result.rs_vs_sector < 1.0

    def test_rs_trend_classified(self):
        result = analyze_relative_strength(
            _make_outperformer(), symbol="TEST",
            sector_etf_df=_make_benchmark(),
        )
        assert result.rs_vs_sector_trend in ("improving", "deteriorating", "neutral")


class TestSectorVsSPY:
    def test_sector_vs_spy_computed(self):
        result = analyze_relative_strength(
            _make_df(), symbol="TEST",
            sector_etf_df=_make_outperformer(),
            spy_df=_make_benchmark(),
        )
        assert result.sector_vs_spy is not None

    def test_sector_momentum(self):
        result = analyze_relative_strength(
            _make_df(), symbol="TEST",
            sector_etf_df=_make_outperformer(),
            spy_df=_make_benchmark(),
        )
        assert result.sector_momentum_20d is not None


class TestPercentile:
    def test_percentile_with_universe(self):
        universe = {"A": 0.05, "B": 0.02, "C": -0.01, "D": -0.03, "TEST": 0.03}
        result = analyze_relative_strength(
            _make_df(), symbol="TEST",
            universe_returns_20d=universe,
        )
        assert result.rs_percentile is not None
        assert 0 <= result.rs_percentile <= 100

    def test_top_performer_high_percentile(self):
        universe = {"A": -0.05, "B": -0.02, "C": -0.01, "TEST": 0.10}
        result = analyze_relative_strength(
            _make_df(), symbol="TEST",
            universe_returns_20d=universe,
        )
        assert result.rs_percentile >= 70


class TestRSRegime:
    def test_regime_classified(self):
        result = analyze_relative_strength(
            _make_outperformer(), symbol="TEST",
            sector_etf_df=_make_benchmark(),
            spy_df=_make_benchmark(),
            universe_returns_20d={"TEST": 0.08, "A": 0.01, "B": -0.01},
        )
        assert result.rs_regime in ("leading", "improving", "lagging", "deteriorating")

    def test_score_in_range(self):
        result = analyze_relative_strength(_make_df(), symbol="TEST")
        assert -1.0 <= result.rs_score <= 1.0


class TestUniverseReturns:
    def test_compute_returns(self):
        dfs = {
            "A": _make_df(n=30, daily_return=0.002),
            "B": _make_df(n=30, daily_return=-0.001),
        }
        returns = compute_universe_returns(dfs)
        assert "A" in returns
        assert "B" in returns
        assert returns["A"] > returns["B"]

    def test_insufficient_data_returns_none(self):
        dfs = {"A": _make_df(n=5)}
        returns = compute_universe_returns(dfs)
        assert returns["A"] is None


class TestSerialization:
    def test_to_dict_complete(self):
        result = analyze_relative_strength(_make_df(), symbol="TEST", sector="Tech")
        d = result.to_dict()
        assert "sector" in d
        assert "vs_sector" in d
        assert "sector_vs_spy" in d
        assert "rs_percentile" in d
        assert "rs_regime" in d
        assert "rs_score" in d

    def test_json_serializable(self):
        import json
        result = analyze_relative_strength(_make_df())
        json.dumps(result.to_dict())
