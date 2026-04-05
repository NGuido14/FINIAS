"""
Cross-Asset Correlation Analysis Module

Analyzes correlations between major asset classes (equities, bonds, commodities, credit)
to understand portfolio diversification, stress regimes, and tail behavior.

Key metrics:
  - Rolling correlations (20d, 60d, 120d)
  - Correlation trend (is diversification improving or deteriorating?)
  - Correlation percentile vs historical 1-year baseline
  - Beta relationships (sensitivity of one asset to another)
  - Correlation asymmetry (convexity) — do assets correlate differently in up vs down markets?
  - Regime classification (normal, decoupling, stress coupling, breakdown)
  - Diversification regimes based on average correlation levels

All computation is pure Python. No API calls. No external dependencies beyond numpy.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ============================================================================
# Helper Functions: Data Conversion & Basic Computations
# ============================================================================


def _price_to_log_returns(prices: Optional[list[dict]]) -> Optional[np.ndarray]:
    """
    Convert Polygon price data to log returns.

    Args:
        prices: List of [{"close": float}], ascending by date

    Returns:
        np.ndarray of log returns, or None if input is invalid

    Guards:
        - Returns None if input is None or empty
        - Returns None if fewer than 2 observations
        - Returns None if any price is non-positive
    """
    if not prices or len(prices) < 2:
        return None

    try:
        closes = np.array([p["close"] for p in prices], dtype=np.float64)
        if np.any(closes <= 0):
            return None
        log_returns = np.diff(np.log(closes))
        return log_returns
    except (KeyError, ValueError, TypeError):
        return None


def _fred_to_log_returns(series: Optional[list[dict]]) -> Optional[np.ndarray]:
    """
    Convert FRED economic data to log returns.

    Args:
        series: List of [{"value": float}], ascending by date

    Returns:
        np.ndarray of log returns, or None if input is invalid

    Guards:
        - Returns None if input is None or empty
        - Returns None if fewer than 2 observations
        - Returns None if any value is non-positive
    """
    if not series or len(series) < 2:
        return None

    try:
        values = np.array([s["value"] for s in series], dtype=np.float64)
        if np.any(values <= 0):
            return None
        log_returns = np.diff(np.log(values))
        return log_returns
    except (KeyError, ValueError, TypeError):
        return None


def _fred_to_levels(series: Optional[list[dict]]) -> Optional[np.ndarray]:
    """
    Convert FRED economic data to raw levels (no log returns).

    Used for VIX and other indices that should not be differenced.

    Args:
        series: List of [{"value": float}], ascending by date

    Returns:
        np.ndarray of raw values, or None if input is invalid
    """
    if not series:
        return None

    try:
        values = np.array([s["value"] for s in series], dtype=np.float64)
        return values
    except (KeyError, ValueError, TypeError):
        return None


def _rolling_corr(
    a: np.ndarray,
    b: np.ndarray,
    window: int
) -> Optional[float]:
    """
    Compute Pearson correlation of the trailing window.

    Args:
        a, b: Return arrays (must be same length)
        window: Trailing window size

    Returns:
        Pearson correlation, or None if:
        - Not enough data (len < window)
        - Either series has zero variance in trailing window
    """
    if len(a) < window or len(b) < window or len(a) != len(b):
        return None

    a_tail = a[-window:]
    b_tail = b[-window:]

    # Check for zero variance (cannot correlate constant series)
    if np.var(a_tail) == 0 or np.var(b_tail) == 0:
        return None

    corr = np.corrcoef(a_tail, b_tail)[0, 1]
    if np.isnan(corr):
        return None

    return float(corr)


def _rolling_beta(
    a: np.ndarray,
    b: np.ndarray,
    window: int
) -> Optional[float]:
    """
    Compute beta: sensitivity of b to changes in a over trailing window.

    Beta = cov(a, b) / var(a)

    Args:
        a, b: Return arrays (must be same length)
        window: Trailing window size

    Returns:
        Beta, or None if:
        - Not enough data (len < window)
        - Series a has zero variance
    """
    if len(a) < window or len(b) < window or len(a) != len(b):
        return None

    a_tail = a[-window:]
    b_tail = b[-window:]

    var_a = np.var(a_tail, ddof=1)
    if var_a == 0:
        return None

    # np.cov returns (cov_matrix). With ddof=1 by default.
    cov_matrix = np.cov(a_tail, b_tail, ddof=1)
    cov_ab = cov_matrix[0, 1]

    beta = cov_ab / var_a
    return float(beta)


def _r(val: Optional[float], decimals: int = 4) -> Optional[float]:
    """
    Round an optional float to clean output.

    Args:
        val: Value to round (or None)
        decimals: Number of decimal places

    Returns:
        Rounded float, or None if input is None
    """
    if val is None:
        return None
    return float(np.round(val, decimals))


# ============================================================================
# PairCorrelation Dataclass
# ============================================================================


@dataclass
class PairCorrelation:
    """Analysis of correlation between two assets."""

    # Identifiers
    pair_name: str  # e.g., "Oil-Equity"
    asset_a: str   # e.g., "Oil (WTI)"
    asset_b: str   # e.g., "Equity (SPY)"

    # Rolling correlations (main metric)
    corr_20d: Optional[float] = None
    corr_60d: Optional[float] = None
    corr_120d: Optional[float] = None

    # Correlation dynamics
    corr_trend: Optional[str] = None  # "rising", "falling", "stable"
    corr_percentile_1y: Optional[float] = None  # 0-100: where current 60d sits in 1y history

    # Beta relationships
    beta_60d: Optional[float] = None
    beta_120d: Optional[float] = None

    # Regime-conditional correlation (vol-adjusted)
    corr_high_vol: Optional[float] = None
    corr_low_vol: Optional[float] = None
    vol_regime_spread: Optional[float] = None  # high_vol - low_vol

    # Convexity (tail behavior)
    convexity_score: Optional[float] = None      # Asymmetric correlation in tails
    extreme_up_corr: Optional[float] = None      # Corr in top 25% of asset_a returns
    extreme_down_corr: Optional[float] = None    # Corr in bottom 25% of asset_a returns
    convexity_note: Optional[str] = None         # Plain English interpretation

    # Regime classification
    regime_label: Optional[str] = None  # "normal", "decoupling", "stress_coupling", "breakdown"

    def to_dict(self) -> dict:
        """
        Convert to nested dictionary with sub-sections and explanatory notes.

        Returns:
            Dict with structure:
            {
                "pair_name": str,
                "assets": {"a": str, "b": str},
                "rolling_correlations": {...},
                "beta": {...},
                "vol_regime_conditional": {...},
                "convexity": {...},
                "regime_label": str
            }
        """
        return {
            "pair_name": self.pair_name,
            "assets": {
                "a": self.asset_a,
                "b": self.asset_b,
            },
            "rolling_correlations": {
                "corr_20d": _r(self.corr_20d),
                "corr_60d": _r(self.corr_60d),
                "corr_120d": _r(self.corr_120d),
                "trend": self.corr_trend,
                "_trend_note": (
                    "Rising: 60d corr > prior 60d corr + 0.10 (diversification falling). "
                    "Falling: 60d corr < prior 60d corr - 0.10 (diversification improving). "
                    "Stable: changes within ±0.10."
                ),
                "percentile_vs_1y": _r(self.corr_percentile_1y),
                "_percentile_note": (
                    "Where current 60d correlation sits in the distribution of all rolling 60d "
                    "correlations over the past 252 trading days (0-100 scale)."
                ),
            },
            "beta": {
                "beta_60d": _r(self.beta_60d),
                "beta_120d": _r(self.beta_120d),
                "_note": (
                    "Beta = cov(asset_a, asset_b) / var(asset_a). "
                    "Sensitivity of asset_b to moves in asset_a. "
                    "Beta > 1: amplified sensitivity. Beta < 0: inverse relationship."
                ),
            },
            "vol_regime_conditional": {
                "high_vol_corr": _r(self.corr_high_vol),
                "low_vol_corr": _r(self.corr_low_vol),
                "spread": _r(self.vol_regime_spread),
                "_note": (
                    "Split by VIX >= trailing 252d median. High_vol - low_vol spread shows if "
                    "correlation intensifies during stress. Requires 30+ obs per regime, 120+ total."
                ),
            },
            "convexity": {
                "score": _r(self.convexity_score),
                "extreme_up_corr": _r(self.extreme_up_corr),
                "extreme_down_corr": _r(self.extreme_down_corr),
                "note": self.convexity_note,
                "_note": (
                    "Convexity = mean(|extreme_up|, |extreme_down|) - |middle|. "
                    "Positive: tails more correlated than middle (risk amplification). "
                    "Negative: tails more decorrelated than middle (tail hedge). "
                    "Score computed on bottom/top/middle quartiles of asset_a (requires 120+ obs)."
                ),
            },
            "regime_label": self.regime_label,
        }


# ============================================================================
# CorrelationMatrix Dataclass
# ============================================================================


@dataclass
class CorrelationMatrix:
    """Cross-asset correlation matrix and aggregate regime statistics."""

    # Seven key pairs
    oil_equity: Optional[PairCorrelation] = None
    oil_bond: Optional[PairCorrelation] = None
    dollar_equity: Optional[PairCorrelation] = None
    gold_equity: Optional[PairCorrelation] = None
    gold_bond: Optional[PairCorrelation] = None
    credit_equity: Optional[PairCorrelation] = None
    dollar_gold: Optional[PairCorrelation] = None

    # Aggregate stats
    stress_coupling_count: int = 0  # Number of pairs in "stress_coupling" regime
    breakdown_count: int = 0         # Number of pairs in "breakdown" regime
    avg_cross_asset_corr: Optional[float] = None  # Mean absolute correlation across pairs

    # Diversification regime classification
    diversification_regime: Optional[str] = None  # "diversified", "concentrated", "correlated"

    # Snapshot date
    as_of_date: Optional[str] = None

    def to_dict(self) -> dict:
        """
        Convert to nested dictionary with pair details and aggregate summary.

        Returns:
            Dict with structure:
            {
                "pairs": {name: pair.to_dict(), ...},
                "aggregate": {
                    "stress_coupling_count": int,
                    "breakdown_count": int,
                    "avg_absolute_correlation": float,
                    "diversification_regime": str,
                    "_note": str,
                },
                "as_of_date": str
            }
        """
        pairs = {}
        for attr_name in [
            "oil_equity", "oil_bond", "dollar_equity",
            "gold_equity", "gold_bond", "credit_equity", "dollar_gold"
        ]:
            pair = getattr(self, attr_name, None)
            if pair is not None:
                pairs[pair.pair_name] = pair.to_dict()

        return {
            "pairs": pairs,
            "aggregate": {
                "stress_coupling_count": self.stress_coupling_count,
                "breakdown_count": self.breakdown_count,
                "avg_absolute_correlation": _r(self.avg_cross_asset_corr),
                "diversification_regime": self.diversification_regime,
                "_note": (
                    "Diversified: avg_corr < 0.25 (low systemic correlation). "
                    "Concentrated: 0.25-0.45 (moderate systemic correlation). "
                    "Correlated: > 0.45 (high systemic correlation, reduced diversification benefits)."
                ),
            },
            "as_of_date": self.as_of_date,
        }


# ============================================================================
# Pair Computation Function
# ============================================================================


def _compute_pair(
    name: str,
    label_a: str,
    label_b: str,
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    vix_levels: Optional[np.ndarray] = None,
) -> Optional[PairCorrelation]:
    """
    Compute comprehensive correlation analytics for a pair of assets.

    Args:
        name: Pair identifier (e.g., "Oil-Equity")
        label_a, label_b: Asset labels for output (e.g., "Oil (WTI)", "Equity (SPY)")
        returns_a, returns_b: Log return arrays (must be aligned)
        vix_levels: VIX index levels for vol-regime conditioning (optional)

    Returns:
        PairCorrelation object with all metrics computed, or None if insufficient data
    """
    # Align arrays to common length
    min_len = min(len(returns_a), len(returns_b))
    if vix_levels is not None:
        min_len = min(min_len, len(vix_levels))

    if min_len < 60:  # Minimum requirement
        return None

    returns_a = returns_a[-min_len:]
    returns_b = returns_b[-min_len:]
    if vix_levels is not None:
        vix_levels = vix_levels[-min_len:]

    # Rolling correlations
    corr_20d = _rolling_corr(returns_a, returns_b, 20)
    corr_60d = _rolling_corr(returns_a, returns_b, 60)
    corr_120d = _rolling_corr(returns_a, returns_b, 120)

    # Correlation trend: compare current 60d vs 60d from 60 days ago
    corr_trend = None
    if len(returns_a) >= 120 and corr_60d is not None:
        prior_corr_60d = _rolling_corr(returns_a[:-60], returns_b[:-60], 60)
        if prior_corr_60d is not None:
            delta = corr_60d - prior_corr_60d
            if delta > 0.10:
                corr_trend = "rising"
            elif delta < -0.10:
                corr_trend = "falling"
            else:
                corr_trend = "stable"

    # Percentile rank vs 1-year history (trailing 252 days)
    corr_percentile_1y = None
    if len(returns_a) >= 252 and corr_60d is not None:
        rolling_corrs = []
        for i in range(60, min(252, len(returns_a)) + 1):
            rc = _rolling_corr(returns_a[-i:-i+60], returns_b[-i:-i+60], 60) if i >= 60 else None
            if rc is not None:
                rolling_corrs.append(rc)

        if rolling_corrs:
            # Scipy-free percentile: count how many are <= current
            rolling_corrs = np.array(rolling_corrs)
            percentile = float(np.sum(rolling_corrs <= corr_60d) / len(rolling_corrs) * 100)
            corr_percentile_1y = percentile

    # Beta relationships
    beta_60d = _rolling_beta(returns_a, returns_b, 60)
    beta_120d = _rolling_beta(returns_a, returns_b, 120)

    # Regime-conditional correlation (VIX-based vol split)
    corr_high_vol = None
    corr_low_vol = None
    vol_regime_spread = None

    if vix_levels is not None and len(vix_levels) >= 120:
        vix_median = np.median(vix_levels[-252:]) if len(vix_levels) >= 252 else np.median(vix_levels)

        high_vol_mask = vix_levels[-120:] >= vix_median
        low_vol_mask = vix_levels[-120:] < vix_median

        high_vol_count = np.sum(high_vol_mask)
        low_vol_count = np.sum(low_vol_mask)

        # Require 30+ observations per regime, 120+ total
        if high_vol_count >= 30 and low_vol_count >= 30:
            high_vol_returns_a = returns_a[-120:][high_vol_mask]
            high_vol_returns_b = returns_b[-120:][high_vol_mask]
            low_vol_returns_a = returns_a[-120:][low_vol_mask]
            low_vol_returns_b = returns_b[-120:][low_vol_mask]

            if len(high_vol_returns_a) >= 20:
                if np.var(high_vol_returns_a) > 0 and np.var(high_vol_returns_b) > 0:
                    corr_high_vol = float(np.corrcoef(high_vol_returns_a, high_vol_returns_b)[0, 1])
                    if np.isnan(corr_high_vol):
                        corr_high_vol = None

            if len(low_vol_returns_a) >= 20:
                if np.var(low_vol_returns_a) > 0 and np.var(low_vol_returns_b) > 0:
                    corr_low_vol = float(np.corrcoef(low_vol_returns_a, low_vol_returns_b)[0, 1])
                    if np.isnan(corr_low_vol):
                        corr_low_vol = None

            if corr_high_vol is not None and corr_low_vol is not None:
                vol_regime_spread = corr_high_vol - corr_low_vol

    # Convexity: correlation asymmetry in tails
    convexity_score = None
    extreme_up_corr = None
    extreme_down_corr = None
    convexity_note = None

    if len(returns_a) >= 120:
        # Split by quartiles of asset_a
        q25 = np.percentile(returns_a, 25)
        q75 = np.percentile(returns_a, 75)

        down_mask = returns_a <= q25
        middle_mask = (returns_a > q25) & (returns_a < q75)
        up_mask = returns_a >= q75

        down_count = np.sum(down_mask)
        middle_count = np.sum(middle_mask)
        up_count = np.sum(up_mask)

        if down_count >= 20 and middle_count >= 20 and up_count >= 20:
            # Extreme down: bottom 25%
            if down_count >= 2:
                down_a = returns_a[down_mask]
                down_b = returns_b[down_mask]
                if np.var(down_a) > 0 and np.var(down_b) > 0:
                    extreme_down_corr = float(np.corrcoef(down_a, down_b)[0, 1])
                    if np.isnan(extreme_down_corr):
                        extreme_down_corr = None

            # Extreme up: top 25%
            if up_count >= 2:
                up_a = returns_a[up_mask]
                up_b = returns_b[up_mask]
                if np.var(up_a) > 0 and np.var(up_b) > 0:
                    extreme_up_corr = float(np.corrcoef(up_a, up_b)[0, 1])
                    if np.isnan(extreme_up_corr):
                        extreme_up_corr = None

            # Middle 50%
            middle_a = returns_a[middle_mask]
            middle_b = returns_b[middle_mask]
            middle_corr = None
            if middle_count >= 2 and np.var(middle_a) > 0 and np.var(middle_b) > 0:
                middle_corr = float(np.corrcoef(middle_a, middle_b)[0, 1])
                if np.isnan(middle_corr):
                    middle_corr = None

            # Convexity score
            if extreme_up_corr is not None and extreme_down_corr is not None and middle_corr is not None:
                mean_extreme = (np.abs(extreme_up_corr) + np.abs(extreme_down_corr)) / 2
                convexity_score = mean_extreme - np.abs(middle_corr)

                # Plain English note
                if convexity_score > 0.10:
                    convexity_note = (
                        f"Strong positive convexity ({_r(convexity_score)}): "
                        f"Tails more correlated than middle. Risk amplification in extreme moves. "
                        f"Down: {_r(extreme_down_corr)}, Up: {_r(extreme_up_corr)}, Middle: {_r(middle_corr)}."
                    )
                elif convexity_score > 0.03:
                    convexity_note = (
                        f"Moderate positive convexity ({_r(convexity_score)}): "
                        f"Slight tail correlation increase. Down: {_r(extreme_down_corr)}, "
                        f"Up: {_r(extreme_up_corr)}, Middle: {_r(middle_corr)}."
                    )
                elif convexity_score < -0.10:
                    convexity_note = (
                        f"Strong negative convexity ({_r(convexity_score)}): "
                        f"Tails decorrelated vs middle. Tail hedge properties. "
                        f"Down: {_r(extreme_down_corr)}, Up: {_r(extreme_up_corr)}, Middle: {_r(middle_corr)}."
                    )
                elif convexity_score < -0.03:
                    convexity_note = (
                        f"Moderate negative convexity ({_r(convexity_score)}): "
                        f"Slight tail decoration. Down: {_r(extreme_down_corr)}, "
                        f"Up: {_r(extreme_up_corr)}, Middle: {_r(middle_corr)}."
                    )
                else:
                    convexity_note = (
                        f"Linear correlation ({_r(convexity_score)}): "
                        f"Tails move in line with middle. Down: {_r(extreme_down_corr)}, "
                        f"Up: {_r(extreme_up_corr)}, Middle: {_r(middle_corr)}."
                    )

                # Asymmetry note
                if extreme_down_corr is not None and extreme_up_corr is not None:
                    asymmetry = np.abs(extreme_down_corr) - np.abs(extreme_up_corr)
                    if asymmetry > 0.15:
                        convexity_note += " Stronger on down moves (crash correlation risk)."
                    elif asymmetry < -0.15:
                        convexity_note += " Stronger on up moves."

    # Regime classification logic
    regime_label = None
    if corr_60d is not None and corr_120d is not None:
        # Check for breakdown (opposite signs, both far from zero)
        if (corr_60d > 0.1 and corr_120d < -0.1) or (corr_60d < -0.1 and corr_120d > 0.1):
            if np.abs(corr_60d) > 0.1:
                regime_label = "breakdown"

        # Check for stress coupling (high vol regime spread AND meaningful correlation)
        # A high vol_regime_spread with low absolute correlation means the assets
        # respond differently to volatility regimes but aren't actually correlated.
        # That's decoupling, not stress coupling. Require |corr_60d| > 0.25 to
        # label as stress_coupling — otherwise it's misleading.
        if regime_label is None and vol_regime_spread is not None:
            if vol_regime_spread > 0.20:
                if corr_60d is not None and np.abs(corr_60d) > 0.25:
                    regime_label = "stress_coupling"
                elif corr_60d is not None and np.abs(corr_60d) <= 0.25:
                    regime_label = "decoupling"

        # Check for decoupling (120d high but 60d low)
        if regime_label is None:
            if np.abs(corr_120d) > 0.25 and np.abs(corr_60d) < 0.10:
                regime_label = "decoupling"

        # Check percentile extremes
        if regime_label is None and corr_percentile_1y is not None:
            if (corr_percentile_1y > 90 or corr_percentile_1y < 10):
                if np.abs(corr_60d) > 0.3:
                    regime_label = "stress_coupling"
                else:
                    regime_label = "decoupling"

        # Default to normal
        if regime_label is None:
            regime_label = "normal"

    return PairCorrelation(
        pair_name=name,
        asset_a=label_a,
        asset_b=label_b,
        corr_20d=corr_20d,
        corr_60d=corr_60d,
        corr_120d=corr_120d,
        corr_trend=corr_trend,
        corr_percentile_1y=corr_percentile_1y,
        beta_60d=beta_60d,
        beta_120d=beta_120d,
        corr_high_vol=corr_high_vol,
        corr_low_vol=corr_low_vol,
        vol_regime_spread=vol_regime_spread,
        convexity_score=convexity_score,
        extreme_up_corr=extreme_up_corr,
        extreme_down_corr=extreme_down_corr,
        convexity_note=convexity_note,
        regime_label=regime_label,
    )


# ============================================================================
# Main Computation Function
# ============================================================================


def compute_correlation_matrix(
    spy: Optional[list[dict]] = None,  # [{"close": float}]
    tlt: Optional[list[dict]] = None,  # [{"close": float}]
    gld: Optional[list[dict]] = None,  # [{"close": float}]
    hyg: Optional[list[dict]] = None,  # [{"close": float}]
    oil: Optional[list[dict]] = None,  # [{"value": float}]
    dxy: Optional[list[dict]] = None,  # [{"value": float}]
    vix: Optional[list[dict]] = None,  # [{"value": float}]
    as_of_date: Optional[str] = None,
) -> CorrelationMatrix:
    """
    Compute complete cross-asset correlation matrix and regime statistics.

    Analyzes seven key asset pairs:
      - Oil-Equity: commodities / risk assets
      - Oil-Bond: commodities / safe havens
      - Dollar-Equity: FX strength / equity beta
      - Gold-Equity: precious metals / risk assets
      - Gold-Bond: precious metals / fixed income
      - Credit-Equity: corporate bonds / equities (systemic risk)
      - Dollar-Gold: FX / commodities (negative correlation common)

    Args:
        spy: SPY daily closes [{"close": float}], ascending
        tlt: TLT daily closes [{"close": float}], ascending
        gld: GLD daily closes [{"close": float}], ascending
        hyg: HYG daily closes [{"close": float}], ascending
        oil: WTI oil FRED series [{"value": float}], ascending
        dxy: Dollar index FRED series [{"value": float}], ascending
        vix: VIX FRED series [{"value": float}], ascending
        as_of_date: Date label for snapshot (e.g., "2025-03-20")

    Returns:
        CorrelationMatrix with all pair analytics and aggregate stats
    """
    # Convert inputs to log returns
    spy_ret = _price_to_log_returns(spy)
    tlt_ret = _price_to_log_returns(tlt)
    gld_ret = _price_to_log_returns(gld)
    hyg_ret = _price_to_log_returns(hyg)
    oil_ret = _fred_to_log_returns(oil)
    dxy_ret = _fred_to_log_returns(dxy)
    vix_lev = _fred_to_levels(vix)

    # Compute seven pairs
    oil_equity = None
    if oil_ret is not None and spy_ret is not None:
        oil_equity = _compute_pair(
            "Oil-Equity", "Oil (WTI)", "Equity (SPY)",
            oil_ret, spy_ret, vix_lev
        )

    oil_bond = None
    if oil_ret is not None and tlt_ret is not None:
        oil_bond = _compute_pair(
            "Oil-Bond", "Oil (WTI)", "Bonds (TLT)",
            oil_ret, tlt_ret, vix_lev
        )

    dollar_equity = None
    if dxy_ret is not None and spy_ret is not None:
        dollar_equity = _compute_pair(
            "Dollar-Equity", "Dollar (DXY)", "Equity (SPY)",
            dxy_ret, spy_ret, vix_lev
        )

    gold_equity = None
    if gld_ret is not None and spy_ret is not None:
        gold_equity = _compute_pair(
            "Gold-Equity", "Gold (GLD)", "Equity (SPY)",
            gld_ret, spy_ret, vix_lev
        )

    gold_bond = None
    if gld_ret is not None and tlt_ret is not None:
        gold_bond = _compute_pair(
            "Gold-Bond", "Gold (GLD)", "Bonds (TLT)",
            gld_ret, tlt_ret, vix_lev
        )

    credit_equity = None
    if hyg_ret is not None and spy_ret is not None:
        credit_equity = _compute_pair(
            "Credit-Equity", "Credit (HYG)", "Equity (SPY)",
            hyg_ret, spy_ret, vix_lev
        )

    dollar_gold = None
    if dxy_ret is not None and gld_ret is not None:
        dollar_gold = _compute_pair(
            "Dollar-Gold", "Dollar (DXY)", "Gold (GLD)",
            dxy_ret, gld_ret, vix_lev
        )

    # Aggregate statistics
    stress_coupling_count = 0
    breakdown_count = 0
    correlations_for_avg = []

    for pair in [oil_equity, oil_bond, dollar_equity, gold_equity, gold_bond, credit_equity, dollar_gold]:
        if pair is not None:
            if pair.regime_label == "stress_coupling":
                stress_coupling_count += 1
            elif pair.regime_label == "breakdown":
                breakdown_count += 1

            if pair.corr_60d is not None:
                correlations_for_avg.append(np.abs(pair.corr_60d))

    avg_cross_asset_corr = None
    if correlations_for_avg:
        avg_cross_asset_corr = np.mean(correlations_for_avg)

    diversification_regime = None
    if avg_cross_asset_corr is not None:
        if avg_cross_asset_corr < 0.25:
            diversification_regime = "diversified"
        elif avg_cross_asset_corr < 0.45:
            diversification_regime = "concentrated"
        else:
            diversification_regime = "correlated"

    return CorrelationMatrix(
        oil_equity=oil_equity,
        oil_bond=oil_bond,
        dollar_equity=dollar_equity,
        gold_equity=gold_equity,
        gold_bond=gold_bond,
        credit_equity=credit_equity,
        dollar_gold=dollar_gold,
        stress_coupling_count=stress_coupling_count,
        breakdown_count=breakdown_count,
        avg_cross_asset_corr=avg_cross_asset_corr,
        diversification_regime=diversification_regime,
        as_of_date=as_of_date,
    )


# ============================================================================
# Notes & Reporting Function
# ============================================================================


def generate_correlation_data_notes(corr_matrix: CorrelationMatrix) -> list[str]:
    """
    Generate plain-English notes for a CorrelationMatrix.

    Returns:
        List of strings describing key findings:
        - Guardrail header
        - One line per pair (name, 60d corr, regime)
        - Aggregate summary
    """
    notes = []

    # Guardrail header
    notes.append(
        "[CORRELATION GUARDRAIL] Cross-asset correlations drive portfolio diversification. "
        "Low correlations = hedge opportunities. High correlations = systemic risk. "
        "Breakdown regimes signal structural shifts."
    )

    # Per-pair summary
    pairs = [
        corr_matrix.oil_equity,
        corr_matrix.oil_bond,
        corr_matrix.dollar_equity,
        corr_matrix.gold_equity,
        corr_matrix.gold_bond,
        corr_matrix.credit_equity,
        corr_matrix.dollar_gold,
    ]

    for pair in pairs:
        if pair is not None:
            note = (
                f"{pair.pair_name}: 60d corr = {_r(pair.corr_60d, 3)}, "
                f"regime = {pair.regime_label}"
            )
            if pair.corr_trend:
                note += f", trend = {pair.corr_trend}"
            notes.append(note)

    # Aggregate summary
    if corr_matrix.avg_cross_asset_corr is not None:
        agg_note = (
            f"Average absolute correlation: {_r(corr_matrix.avg_cross_asset_corr, 3)}. "
            f"Regime: {corr_matrix.diversification_regime}. "
            f"Stress coupling pairs: {corr_matrix.stress_coupling_count}, "
            f"Breakdown pairs: {corr_matrix.breakdown_count}."
        )
        notes.append(agg_note)

    return notes
