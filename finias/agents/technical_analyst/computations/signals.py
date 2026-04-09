"""
Signal Confluence Synthesis Engine

This is the intelligence layer that transforms 6 independent computation
modules into actionable conviction scores. Every weighting and threshold
in this module is derived from empirical backtest results on 99k signals
with forward returns across 2022-2025.

KEY EMPIRICAL FINDINGS ENCODED HERE:
  1. Mean-reversion dominates at stock level (strong_downtrend → +0.45% excess)
  2. Macro regime flips the optimal strategy (crisis = mean-revert, risk_on = trend-follow)
  3. Divergences add real alpha only in downtrends (+1.04% excess)
  4. Volume exhaustion (contracting vol in downtrend) identifies better mean-reversion entries
  5. RS improvement in downtrend = early reversal signal
  6. Multiple independent dimensions agreeing = higher conviction

The synthesis engine does NOT predict direction. It measures:
  - CONFLUENCE: How many independent dimensions agree?
  - CONVICTION: How confident should we be based on empirical hit rates?
  - SETUP TYPE: Mean-reversion, trend-continuation, or squeeze-breakout?
  - MACRO CONDITIONING: Which strategy does the macro environment favor?
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SignalSynthesis:
    """Synthesized signal for a single symbol."""

    symbol: str = ""

    # Confluence
    bullish_dimensions: int = 0  # Count of dimensions signaling bullish (0-6)
    bearish_dimensions: int = 0
    confluence_score: float = 0.0  # -1 (max bearish confluence) to +1 (max bullish)

    # Conviction
    conviction: str = "low"  # high, moderate, low
    conviction_score: float = 0.0  # 0-1

    # Setup Classification
    setup_type: str = "none"  # mean_reversion_buy, trend_continuation, squeeze_breakout,
    # exhaustion_sell, distribution_warning, none
    setup_description: str = ""

    # Macro Conditioning
    macro_regime_used: str = "unknown"
    macro_alignment: str = "neutral"  # aligned (macro favors this setup), opposed, neutral

    # Component Scores (for transparency)
    trend_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    rs_score: float = 0.0
    vol_score: float = 0.0
    levels_rr: Optional[float] = None  # Risk/reward from levels module

    # Actionable Output
    action: str = "hold"  # strong_buy, buy, hold, reduce, sell, strong_sell
    position_bias: float = 0.0  # -1 (full short) to +1 (full long)

    def to_dict(self) -> dict:
        return {
            "confluence": {
                "bullish_dims": self.bullish_dimensions,
                "bearish_dims": self.bearish_dimensions,
                "score": round(self.confluence_score, 4),
            },
            "conviction": {
                "level": self.conviction,
                "score": round(self.conviction_score, 4),
            },
            "setup": {
                "type": self.setup_type,
                "description": self.setup_description,
            },
            "macro": {
                "regime": self.macro_regime_used,
                "alignment": self.macro_alignment,
            },
            "components": {
                "trend": round(self.trend_score, 4),
                "momentum": round(self.momentum_score, 4),
                "volume": round(self.volume_score, 4),
                "rs": round(self.rs_score, 4),
                "volatility": round(self.vol_score, 4),
                "risk_reward": round(self.levels_rr, 2) if self.levels_rr is not None else None,
            },
            "action": self.action,
            "position_bias": round(self.position_bias, 4),
        }


def synthesize_signals(
    trend: dict,
    momentum: dict,
    levels: dict,
    volume: dict,
    relative_strength: dict,
    volatility: dict,
    symbol: str = "",
    macro_regime: str = "unknown",
) -> SignalSynthesis:
    """
    Synthesize all 6 computation modules into a single actionable signal.

    This is the core intelligence function. Every threshold and weight
    is derived from empirical backtest results.

    Args:
        trend: Output from analyze_trend().to_dict()
        momentum: Output from analyze_momentum().to_dict()
        levels: Output from analyze_levels().to_dict()
        volume: Output from analyze_volume().to_dict()
        relative_strength: Output from analyze_relative_strength().to_dict()
        volatility: Output from analyze_volatility().to_dict()
        symbol: Ticker symbol.
        macro_regime: From macro agent or proxy (crisis, risk_off, transition, moderate_bull, risk_on).

    Returns:
        SignalSynthesis with confluence, conviction, setup type, and action.
    """
    result = SignalSynthesis(symbol=symbol, macro_regime_used=macro_regime)

    # Extract component scores
    result.trend_score = trend.get("trend_score", 0)
    result.momentum_score = momentum.get("momentum_score", 0)
    result.volume_score = volume.get("volume_confirmation_score", 0)
    result.rs_score = relative_strength.get("rs_score", 0)
    result.vol_score = volatility.get("vol_score", 0)
    result.levels_rr = levels.get("risk_reward_ratio")

    # Step 1: Count confluence (independent dimensions agreeing)
    _count_confluence(result)

    # Step 2: Classify the setup type
    _classify_setup(result, trend, momentum, volume, relative_strength, volatility, macro_regime)

    # Step 3: Apply macro conditioning
    _apply_macro_conditioning(result, macro_regime)

    # Step 4: Compute conviction from confluence + macro alignment
    _compute_conviction(result)

    # Step 5: Determine action
    _determine_action(result)

    return result


def _count_confluence(result: SignalSynthesis):
    """Count how many independent dimensions are bullish vs bearish."""
    threshold = 0.15  # Must exceed this to count as directional

    scores = [
        result.trend_score,
        result.momentum_score,
        result.volume_score,
        result.rs_score,
        result.vol_score,
    ]

    for s in scores:
        if s > threshold:
            result.bullish_dimensions += 1
        elif s < -threshold:
            result.bearish_dimensions += 1

    # Confluence score: net agreement normalized to -1 to +1
    total = max(result.bullish_dimensions + result.bearish_dimensions, 1)
    result.confluence_score = (result.bullish_dimensions - result.bearish_dimensions) / 5.0


def _classify_setup(
    result: SignalSynthesis,
    trend: dict, momentum: dict, volume: dict,
    rs: dict, volatility: dict, macro_regime: str,
):
    """
    Classify the setup type based on empirical patterns.

    These patterns are the ones our backtest proved have predictive value.
    """
    trend_regime = trend.get("trend_regime", "unknown")
    div_type = momentum.get("divergence", {}).get("type", "none")
    obv_div = volume.get("obv", {}).get("divergence", "none")
    vol_trend = volume.get("regime_volume_trend", {}).get("direction", "neutral")
    rs_regime = rs.get("rs_regime", "neutral")
    squeeze = volatility.get("squeeze", {}).get("active", False)
    squeeze_released = volatility.get("squeeze", {}).get("just_released", False)

    # === MEAN REVERSION BUY ===
    # Backtest: strong_downtrend + bullish_divergence = +1.04% excess
    # Enhanced by: volume exhaustion (contracting), RS improving, OBV bullish
    if trend_regime in ("strong_downtrend", "downtrend"):
        bullish_signals = 0
        description_parts = []

        if div_type in ("bullish", "hidden_bullish"):
            bullish_signals += 2  # Divergence is the strongest signal
            description_parts.append(f"{div_type} momentum divergence")

        if obv_div == "bullish":
            bullish_signals += 2  # OBV divergence = institutional accumulation
            description_parts.append("OBV accumulation divergence")

        if vol_trend == "contracting":
            bullish_signals += 1  # Volume exhaustion
            description_parts.append("volume exhaustion (contracting)")

        if rs_regime == "improving":
            bullish_signals += 1  # Bottoming ahead of peers
            description_parts.append("RS improving vs sector")

        if bullish_signals >= 3:
            result.setup_type = "mean_reversion_buy"
            result.setup_description = (
                f"Mean-reversion setup in {trend_regime}: "
                + ", ".join(description_parts)
            )
            return

    # === TREND CONTINUATION ===
    # Backtest: only works in risk-on macro, +0.51% excess
    if trend_regime in ("strong_uptrend", "uptrend"):
        trend_confirms = 0

        if momentum.get("momentum_score", 0) > 0.3:
            trend_confirms += 1
        if volume.get("volume_confirmation_score", 0) > 0.2:
            trend_confirms += 1
        if rs.get("rs_regime") in ("leading", "improving"):
            trend_confirms += 1
        if momentum.get("macd", {}).get("momentum") == "accelerating":
            trend_confirms += 1

        if trend_confirms >= 3:
            result.setup_type = "trend_continuation"
            result.setup_description = (
                f"Trend continuation in {trend_regime}: "
                f"{trend_confirms}/4 confirmations (momentum, volume, RS, MACD)"
            )
            return

    # === SQUEEZE BREAKOUT ===
    if squeeze_released:
        direction = "bullish" if result.momentum_score > 0 else "bearish"
        result.setup_type = "squeeze_breakout"
        result.setup_description = (
            f"Squeeze released — {direction} breakout indicated by "
            f"momentum direction ({result.momentum_score:.2f})"
        )
        return

    # === DISTRIBUTION WARNING ===
    if trend_regime in ("uptrend", "strong_uptrend"):
        bearish_signals = 0

        if div_type == "bearish":
            bearish_signals += 1
        if obv_div == "bearish":
            bearish_signals += 1
        if volume.get("regime_volume_trend", {}).get("direction") == "contracting":
            bearish_signals += 1
        if rs.get("rs_regime") == "deteriorating":
            bearish_signals += 1

        if bearish_signals >= 2:
            result.setup_type = "distribution_warning"
            result.setup_description = (
                f"Distribution in {trend_regime}: "
                f"{bearish_signals} warning signals (bearish divergence, OBV, volume, RS)"
            )
            return

    # === EXHAUSTION SELL ===
    if (trend_regime in ("strong_uptrend",) and
            momentum.get("rsi", {}).get("zone") == "overbought" and
            rs.get("rs_regime") == "deteriorating"):
        result.setup_type = "exhaustion_sell"
        result.setup_description = (
            "Exhaustion in strong uptrend: overbought RSI + deteriorating relative strength"
        )
        return

    result.setup_type = "none"
    result.setup_description = "No high-conviction setup identified"


def _apply_macro_conditioning(result: SignalSynthesis, macro_regime: str):
    """
    Apply macro conditioning based on empirical backtest findings.

    Crisis + mean_reversion_buy = ALIGNED (backtest: +4.36% excess)
    Risk_on + trend_continuation = ALIGNED (backtest: +0.51% excess)
    Crisis + trend_continuation = OPPOSED (backtest: -7.86% excess)
    Risk_on + mean_reversion_buy = OPPOSED (backtest: -1.34% excess)
    """
    setup = result.setup_type

    # Define alignment rules from backtest
    aligned_combos = {
        ("crisis", "mean_reversion_buy"),
        ("risk_off", "mean_reversion_buy"),
        ("risk_on", "trend_continuation"),
        ("moderate_bull", "trend_continuation"),
    }

    opposed_combos = {
        ("crisis", "trend_continuation"),
        ("crisis", "exhaustion_sell"),  # Don't sell into crisis panic
        ("risk_on", "mean_reversion_buy"),
        ("risk_on", "distribution_warning"),
    }

    key = (macro_regime, setup)

    if key in aligned_combos:
        result.macro_alignment = "aligned"
    elif key in opposed_combos:
        result.macro_alignment = "opposed"
    elif setup in ("squeeze_breakout",):
        result.macro_alignment = "neutral"  # Squeezes work in any environment
    else:
        result.macro_alignment = "neutral"


def _compute_conviction(result: SignalSynthesis):
    """
    Compute conviction score from confluence + macro alignment + setup quality.

    High conviction requires: 3+ agreeing dimensions + macro aligned + recognized setup
    """
    score = 0.0

    # Confluence contribution (0-0.4)
    net_dims = abs(result.bullish_dimensions - result.bearish_dimensions)
    score += min(net_dims / 5.0, 0.4)

    # Setup type contribution (0-0.3)
    setup_weights = {
        "mean_reversion_buy": 0.3,
        "trend_continuation": 0.2,
        "squeeze_breakout": 0.25,
        "distribution_warning": 0.2,
        "exhaustion_sell": 0.15,
        "none": 0.0,
    }
    score += setup_weights.get(result.setup_type, 0)

    # Macro alignment contribution (0-0.3)
    alignment_weights = {
        "aligned": 0.3,
        "neutral": 0.1,
        "opposed": -0.2,
    }
    score += alignment_weights.get(result.macro_alignment, 0)

    # Risk/reward from levels (bonus)
    if result.levels_rr is not None and result.levels_rr > 2.0:
        score += 0.1

    result.conviction_score = max(0.0, min(1.0, score))

    if result.conviction_score >= 0.7:
        result.conviction = "high"
    elif result.conviction_score >= 0.4:
        result.conviction = "moderate"
    else:
        result.conviction = "low"


def _determine_action(result: SignalSynthesis):
    """Determine the actionable output based on conviction + direction."""
    setup = result.setup_type
    conviction = result.conviction
    alignment = result.macro_alignment

    # Bullish setups
    if setup in ("mean_reversion_buy", "trend_continuation", "squeeze_breakout"):
        if setup == "squeeze_breakout" and result.momentum_score < 0:
            # Bearish squeeze breakout
            if conviction == "high" and alignment != "opposed":
                result.action = "sell"
                result.position_bias = -0.6
            else:
                result.action = "reduce"
                result.position_bias = -0.3
        else:
            # Bullish setup
            if conviction == "high" and alignment == "aligned":
                result.action = "strong_buy"
                result.position_bias = 0.8
            elif conviction == "high":
                result.action = "buy"
                result.position_bias = 0.6
            elif conviction == "moderate" and alignment != "opposed":
                result.action = "buy"
                result.position_bias = 0.4
            else:
                result.action = "hold"
                result.position_bias = 0.1

    # Bearish setups
    elif setup in ("distribution_warning", "exhaustion_sell"):
        if conviction == "high" and alignment == "aligned":
            result.action = "strong_sell"
            result.position_bias = -0.8
        elif conviction == "high":
            result.action = "sell"
            result.position_bias = -0.5
        elif conviction == "moderate" and alignment != "opposed":
            result.action = "reduce"
            result.position_bias = -0.3
        else:
            result.action = "hold"
            result.position_bias = -0.1

    else:
        result.action = "hold"
        result.position_bias = 0.0
