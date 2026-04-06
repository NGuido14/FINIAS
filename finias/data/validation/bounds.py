"""
Computation output bounds checking.

Defines known reasonable ranges for every computed value in the system.
If a computation produces a value outside these bounds, it's either
a bug or an unprecedented market condition — either way, worth flagging.

These bounds are intentionally wide. They catch impossible values
(recession_prob = 1.5, VIX = -3) and extreme outliers, not normal
fluctuations. False positives should be very rare.
"""

from __future__ import annotations
from typing import Optional
import logging

logger = logging.getLogger("finias.data.validation.bounds")


# ============================================================================
# Known Reasonable Bounds
# ============================================================================

COMPUTATION_BOUNDS = {
    # Regime assessment
    "recession_prob": (0.0, 1.0, "Recession probability must be 0-1"),
    "sahm_value": (-0.5, 3.0, "Sahm Rule historically -0.2 to 2.5"),
    "composite_score": (-1.5, 1.5, "Composite score should be near -1 to +1"),
    "stress_index": (-0.1, 1.1, "Stress index should be 0-1"),
    "confidence": (0.0, 1.0, "Confidence must be 0-1"),

    # Category scores
    "growth_cycle_score": (-1.5, 1.5, "Category score should be near -1 to +1"),
    "monetary_liquidity_score": (-1.5, 1.5, "Category score should be near -1 to +1"),
    "inflation_score": (-1.5, 1.5, "Category score should be near -1 to +1"),
    "market_signals_score": (-1.5, 1.5, "Category score should be near -1 to +1"),

    # Key levels
    "vix": (3.0, 100.0, "VIX historically 9-80, allowing buffer"),
    "hy_spread": (0.5, 25.0, "HY spread historically 2.5-20%"),
    "core_pce_yoy": (-3.0, 15.0, "Core PCE historically 0.5-10%"),
    "fed_funds": (-1.0, 25.0, "Fed funds historically 0-20%"),
    "net_liquidity_trillion": (1.0, 12.0, "Net liquidity historically 3-7T, allowing buffer"),
    "spread_2s10s": (-5.0, 5.0, "2s10s spread historically -2.5 to +3.0"),

    # Positioning
    "net_spec_percentile": (0.0, 100.0, "Percentile must be 0-100"),
    "positioning_aggregate_score": (-1.1, 1.1, "Aggregate score should be -1 to +1"),

    # Position sizing
    "max_single_position_pct": (0.1, 10.0, "Max position should be 0.5-5%"),
    "portfolio_beta_target": (0.0, 2.0, "Beta target should be 0-1.5"),
    "cash_target_pct": (0.0, 50.0, "Cash target should be 0-25%"),
}


def check_computation_bounds(
    key_levels: dict,
    additional_values: dict = None,
) -> list[str]:
    """
    Check computed values against known reasonable bounds.

    Args:
        key_levels: The key_levels dict from RegimeAssessment.
        additional_values: Any other named values to check.

    Returns:
        List of violation warning strings. Empty = all within bounds.
    """
    violations = []
    values_to_check = dict(key_levels) if key_levels else {}
    if additional_values:
        values_to_check.update(additional_values)

    for field_name, (min_val, max_val, description) in COMPUTATION_BOUNDS.items():
        value = values_to_check.get(field_name)
        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if value < min_val:
            msg = f"BOUNDS VIOLATION: {field_name}={value} below minimum {min_val}. {description}"
            violations.append(msg)
            logger.error(msg)
        elif value > max_val:
            msg = f"BOUNDS VIOLATION: {field_name}={value} above maximum {max_val}. {description}"
            violations.append(msg)
            logger.error(msg)

    return violations


def check_value_change(
    current: float,
    prior: float,
    field_name: str,
    max_change_pct: float = 200.0,
) -> Optional[str]:
    """
    Check if a value changed by more than expected between assessments.

    Catches cases like VIX jumping from 25 to 250 (data error) or
    recession probability jumping from 9% to 90% (computation bug).

    Args:
        current: Current value
        prior: Previous assessment's value
        field_name: For reporting
        max_change_pct: Maximum acceptable percentage change

    Returns:
        Warning string if suspicious, None if okay.
    """
    if prior == 0:
        return None

    change_pct = abs(current - prior) / abs(prior) * 100

    if change_pct > max_change_pct:
        return (
            f"LARGE CHANGE: {field_name} changed {change_pct:.0f}% "
            f"({prior} → {current}). Verify data integrity."
        )

    return None
