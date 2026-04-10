"""Tests for the signal confluence synthesis engine."""

import pytest
from finias.agents.technical_analyst.computations.signals import (
    synthesize_signals, SignalSynthesis,
)


def _base_signals():
    """Return neutral baseline signals."""
    return {
        "trend": {"trend_regime": "consolidation", "trend_score": 0.0},
        "momentum": {"momentum_score": 0.0, "divergence": {"type": "none"}, "rsi": {"zone": "neutral"}, "macd": {"momentum": "neutral"}},
        "levels": {"risk_reward_ratio": 1.5, "nearest_support": 95.0, "nearest_resistance": 105.0},
        "volume": {"volume_confirmation_score": 0.0, "obv": {"divergence": "none"}, "regime_volume_trend": {"direction": "neutral"}},
        "relative_strength": {"rs_score": 0.0, "rs_regime": "neutral"},
        "volatility": {"vol_score": 0.0, "squeeze": {"active": False, "just_released": False}},
    }


class TestMeanReversionSetup:
    def test_strong_downtrend_with_divergence(self):
        sigs = _base_signals()
        sigs["trend"]["trend_regime"] = "strong_downtrend"
        sigs["trend"]["trend_score"] = -0.7
        sigs["momentum"]["divergence"]["type"] = "bullish"
        sigs["momentum"]["momentum_score"] = 0.3
        sigs["volume"]["volume_confirmation_score"] = 0.3
        sigs["volume"]["obv"]["divergence"] = "bullish"
        sigs["relative_strength"]["rs_regime"] = "improving"
        sigs["relative_strength"]["rs_score"] = 0.3

        result = synthesize_signals(**sigs, symbol="TEST", macro_regime="crisis")
        assert result.setup_type == "mean_reversion_buy"
        assert result.macro_alignment == "aligned"
        assert result.conviction == "high"
        assert result.action in ("strong_buy", "buy")

    def test_crisis_amplifies_mean_reversion(self):
        sigs = _base_signals()
        sigs["trend"]["trend_regime"] = "strong_downtrend"
        sigs["trend"]["trend_score"] = -0.6
        sigs["momentum"]["divergence"]["type"] = "bullish"
        sigs["volume"]["obv"]["divergence"] = "bullish"
        sigs["volume"]["regime_volume_trend"]["direction"] = "contracting"

        crisis = synthesize_signals(**sigs, macro_regime="crisis")
        risk_on = synthesize_signals(**sigs, macro_regime="risk_on")

        assert crisis.macro_alignment == "aligned"
        assert risk_on.macro_alignment == "opposed"
        assert crisis.conviction_score > risk_on.conviction_score

    def test_high_stress_boosts_mean_reversion(self):
        """High macro stress should upgrade mean-reversion alignment."""
        sigs = _base_signals()
        sigs["trend"]["trend_regime"] = "strong_downtrend"
        sigs["trend"]["trend_score"] = -0.6
        sigs["momentum"]["divergence"]["type"] = "bullish"
        sigs["volume"]["obv"]["divergence"] = "bullish"
        sigs["volume"]["regime_volume_trend"]["direction"] = "contracting"

        # Use "unknown" regime so primary alignment is neutral, then stress upgrades it
        with_stress = synthesize_signals(**sigs, macro_regime="unknown", macro_stress=0.6)
        without_stress = synthesize_signals(**sigs, macro_regime="unknown", macro_stress=0.1)

        assert with_stress.macro_alignment == "aligned"
        assert without_stress.macro_alignment == "neutral"
        assert with_stress.conviction_score > without_stress.conviction_score

    def test_extreme_vol_reduces_conviction(self):
        """Extreme volatility should downgrade aligned signals."""
        sigs = _base_signals()
        sigs["trend"]["trend_regime"] = "strong_downtrend"
        sigs["trend"]["trend_score"] = -0.6
        sigs["momentum"]["divergence"]["type"] = "bullish"
        sigs["volume"]["obv"]["divergence"] = "bullish"
        sigs["volume"]["regime_volume_trend"]["direction"] = "contracting"

        extreme = synthesize_signals(**sigs, macro_regime="crisis", macro_volatility="extreme")
        normal = synthesize_signals(**sigs, macro_regime="crisis", macro_volatility="normal")

        # Extreme vol should downgrade from aligned to neutral
        assert extreme.macro_alignment == "neutral"
        assert normal.macro_alignment == "aligned"

    def test_transition_does_not_align_mean_reversion(self):
        """Transition + mean_reversion should NOT be aligned (only +0.02% excess = noise)."""
        sigs = _base_signals()
        sigs["trend"]["trend_regime"] = "strong_downtrend"
        sigs["trend"]["trend_score"] = -0.6
        sigs["momentum"]["divergence"]["type"] = "bullish"
        sigs["volume"]["obv"]["divergence"] = "bullish"
        sigs["volume"]["regime_volume_trend"]["direction"] = "contracting"

        result = synthesize_signals(**sigs, macro_regime="transition")
        assert result.macro_alignment == "neutral"  # NOT aligned


class TestTrendContinuation:
    def test_uptrend_with_confirmation(self):
        sigs = _base_signals()
        sigs["trend"]["trend_regime"] = "strong_uptrend"
        sigs["trend"]["trend_score"] = 0.7
        sigs["momentum"]["momentum_score"] = 0.5
        sigs["momentum"]["macd"]["momentum"] = "accelerating"
        sigs["volume"]["volume_confirmation_score"] = 0.3
        sigs["relative_strength"]["rs_regime"] = "leading"
        sigs["relative_strength"]["rs_score"] = 0.5

        result = synthesize_signals(**sigs, macro_regime="risk_on")
        assert result.setup_type == "trend_continuation"
        assert result.macro_alignment == "aligned"


class TestSqueezeBreakout:
    def test_squeeze_released_bullish(self):
        sigs = _base_signals()
        sigs["volatility"]["squeeze"]["just_released"] = True
        sigs["momentum"]["momentum_score"] = 0.4

        result = synthesize_signals(**sigs)
        assert result.setup_type == "squeeze_breakout"
        assert "bullish" in result.setup_description


class TestDistributionWarning:
    def test_uptrend_with_bearish_signals(self):
        sigs = _base_signals()
        sigs["trend"]["trend_regime"] = "uptrend"
        sigs["trend"]["trend_score"] = 0.3
        sigs["momentum"]["divergence"]["type"] = "bearish"
        sigs["volume"]["obv"]["divergence"] = "bearish"
        sigs["relative_strength"]["rs_regime"] = "deteriorating"

        result = synthesize_signals(**sigs)
        assert result.setup_type == "distribution_warning"

    def test_risk_on_aligns_distribution_warning(self):
        """Risk_on + distribution_warning should be aligned (+2.26% excess)."""
        sigs = _base_signals()
        sigs["trend"]["trend_regime"] = "uptrend"
        sigs["trend"]["trend_score"] = 0.5
        sigs["momentum"]["divergence"]["type"] = "bearish"
        sigs["relative_strength"]["rs_regime"] = "deteriorating"

        result = synthesize_signals(**sigs, macro_regime="risk_on")
        assert result.setup_type == "distribution_warning"
        assert result.macro_alignment == "aligned"


class TestConfluence:
    def test_all_bullish(self):
        sigs = _base_signals()
        sigs["trend"]["trend_score"] = 0.5
        sigs["momentum"]["momentum_score"] = 0.5
        sigs["volume"]["volume_confirmation_score"] = 0.5
        sigs["relative_strength"]["rs_score"] = 0.5
        sigs["volatility"]["vol_score"] = 0.5

        result = synthesize_signals(**sigs)
        # Only 3 validated dimensions count (trend, momentum, rs)
        # Volume and volatility are excluded from confluence counting
        assert result.bullish_dimensions == 3
        assert result.bearish_dimensions == 0
        assert result.confluence_score == 1.0  # 3/3 = 1.0

    def test_mixed_signals(self):
        sigs = _base_signals()
        sigs["trend"]["trend_score"] = 0.5
        sigs["momentum"]["momentum_score"] = -0.5
        sigs["volume"]["volume_confirmation_score"] = 0.3

        result = synthesize_signals(**sigs)
        assert result.bullish_dimensions >= 1
        assert result.bearish_dimensions >= 1


class TestConviction:
    def test_high_conviction_requires_alignment(self):
        sigs = _base_signals()
        sigs["trend"]["trend_regime"] = "strong_downtrend"
        sigs["trend"]["trend_score"] = -0.7
        sigs["momentum"]["divergence"]["type"] = "bullish"
        sigs["volume"]["obv"]["divergence"] = "bullish"
        sigs["volume"]["regime_volume_trend"]["direction"] = "contracting"

        aligned = synthesize_signals(**sigs, macro_regime="crisis")
        opposed = synthesize_signals(**sigs, macro_regime="risk_on")

        assert aligned.conviction_score > opposed.conviction_score

    def test_conviction_score_range(self):
        result = synthesize_signals(**_base_signals())
        assert 0.0 <= result.conviction_score <= 1.0


class TestAction:
    def test_hold_on_no_setup(self):
        result = synthesize_signals(**_base_signals())
        assert result.action == "hold"
        assert result.position_bias == 0.0

    def test_strong_buy_on_aligned_high_conviction(self):
        sigs = _base_signals()
        sigs["trend"]["trend_regime"] = "strong_downtrend"
        sigs["trend"]["trend_score"] = -0.7
        sigs["momentum"]["divergence"]["type"] = "bullish"
        sigs["momentum"]["momentum_score"] = 0.4
        sigs["volume"]["obv"]["divergence"] = "bullish"
        sigs["volume"]["volume_confirmation_score"] = 0.4
        sigs["volume"]["regime_volume_trend"]["direction"] = "contracting"
        sigs["relative_strength"]["rs_regime"] = "improving"
        sigs["relative_strength"]["rs_score"] = 0.3
        sigs["levels"]["risk_reward_ratio"] = 3.0

        result = synthesize_signals(**sigs, macro_regime="crisis")
        assert result.action in ("strong_buy", "buy")
        assert result.position_bias > 0.5


class TestSerialization:
    def test_to_dict_complete(self):
        result = synthesize_signals(**_base_signals(), symbol="TEST")
        d = result.to_dict()
        assert "confluence" in d
        assert "conviction" in d
        assert "setup" in d
        assert "macro" in d
        assert "components" in d
        assert "action" in d

    def test_json_serializable(self):
        import json
        result = synthesize_signals(**_base_signals())
        json.dumps(result.to_dict())
