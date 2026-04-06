"""
CFTC Positioning Computation Module.

Computes speculative positioning signals from raw COT data:
  1. Net spec percentile (156-week / 3-year lookback) per contract
  2. Crowding flag (extreme positioning >90th or <10th percentile)
  3. Rate of change (4-week net spec change) per contract
  4. Aggregate positioning score across all 5 contracts
  5. Cross-asset positioning divergences

These signals answer: "Does the market already know this?" — the gap
between what conditions ARE and what's ALREADY PRICED.

The S&P 500 positioning signal feeds directly into forward_bias
(replacing the underperforming stress_contrarian signal).
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class ContractPositioning:
    """Positioning analysis for a single futures contract."""

    contract_key: str                           # sp500, treasury_10y, etc.
    report_date: Optional[date] = None          # Date of latest COT report
    net_spec: int = 0                           # Raw net speculative position
    net_spec_percentile: float = 50.0           # 0-100, where current sits in history
    crowding: str = "neutral"                   # crowded_long, crowded_short, neutral
    rate_of_change_4w: int = 0                  # 4-week change in net spec
    lookback_weeks: int = 0                     # How many weeks of history used
    confidence: str = "low"                     # high (156+), moderate (52-155), low (<52)
    open_interest: int = 0                      # Total open interest

    def to_dict(self) -> dict:
        return {
            "contract_key": self.contract_key,
            "report_date": str(self.report_date) if self.report_date else None,
            "net_spec": self.net_spec,
            "net_spec_percentile": round(self.net_spec_percentile, 1),
            "crowding": self.crowding,
            "rate_of_change_4w": self.rate_of_change_4w,
            "lookback_weeks": self.lookback_weeks,
            "confidence": self.confidence,
            "open_interest": self.open_interest,
            "_note": (
                "Percentile: 0=extreme short positioning, 100=extreme long. "
                ">90=crowded_long (contrarian bearish), <10=crowded_short (contrarian bullish). "
                "Rate of change: positive=adding longs, negative=adding shorts."
            ),
        }


@dataclass
class PositioningAnalysis:
    """Aggregate positioning analysis across all 5 futures contracts."""

    contracts: dict = field(default_factory=dict)   # {key: ContractPositioning}
    aggregate_score: float = 0.0                     # -1 to +1 (negative=net short bias)
    crowding_alert_count: int = 0                    # How many contracts at extremes
    divergences: list = field(default_factory=list)  # Cross-asset inconsistencies
    data_staleness_days: int = 999                   # Days since latest report
    sp500_positioning_signal: str = "neutral"         # For forward_bias: constructive/neutral/cautious

    def to_dict(self) -> dict:
        contracts_dict = {}
        for key, cp in self.contracts.items():
            contracts_dict[key] = cp.to_dict()

        return {
            "contracts": contracts_dict,
            "aggregate": {
                "score": round(self.aggregate_score, 3),
                "crowding_alert_count": self.crowding_alert_count,
                "divergences": self.divergences,
                "data_staleness_days": self.data_staleness_days,
                "sp500_positioning_signal": self.sp500_positioning_signal,
                "_note": (
                    "Score: weighted average of contract percentiles mapped to -1/+1. "
                    "Negative=market net short (contrarian bullish). "
                    "Positive=market net long (contrarian bearish). "
                    "sp500_positioning_signal feeds directly into forward_bias."
                ),
            },
        }


# ============================================================================
# Contract Weights for Aggregate Score
# ============================================================================

# S&P 500 gets highest weight — most direct equity market signal
CONTRACT_WEIGHTS = {
    "sp500": 0.40,
    "treasury_10y": 0.20,
    "wti_crude": 0.15,
    "gold": 0.15,
    "dollar_index": 0.10,
}


# ============================================================================
# Core Computation Functions
# ============================================================================

def compute_contract_positioning(
    contract_key: str,
    history: list[dict],
) -> ContractPositioning:
    """
    Compute positioning signals for a single contract.

    Args:
        contract_key: Contract identifier (sp500, treasury_10y, etc.)
        history: List of dicts with report_date, net_spec, open_interest
                 Sorted chronologically (oldest first).

    Returns:
        ContractPositioning with percentile, crowding, and rate of change.
    """
    result = ContractPositioning(contract_key=contract_key)

    if not history:
        return result

    result.lookback_weeks = len(history)

    # Confidence based on history depth
    if result.lookback_weeks >= 156:
        result.confidence = "high"
    elif result.lookback_weeks >= 52:
        result.confidence = "moderate"
    else:
        result.confidence = "low"

    # Latest values
    latest = history[-1]
    result.net_spec = latest.get("net_spec", 0)
    result.open_interest = latest.get("open_interest", 0)
    result.report_date = latest.get("report_date")

    # --- Percentile Calculation ---
    net_specs = [row["net_spec"] for row in history if row.get("net_spec") is not None]
    if len(net_specs) >= 10:
        current = net_specs[-1]
        count_below = sum(1 for v in net_specs if v <= current)
        result.net_spec_percentile = round((count_below / len(net_specs)) * 100, 1)
    else:
        result.net_spec_percentile = 50.0  # Default to neutral if insufficient data

    # --- Crowding Classification ---
    if result.net_spec_percentile >= 90:
        result.crowding = "crowded_long"
    elif result.net_spec_percentile <= 10:
        result.crowding = "crowded_short"
    else:
        result.crowding = "neutral"

    # --- Rate of Change (4 weeks) ---
    if len(net_specs) >= 5:
        result.rate_of_change_4w = net_specs[-1] - net_specs[-5]
    elif len(net_specs) >= 2:
        result.rate_of_change_4w = net_specs[-1] - net_specs[0]

    return result


def compute_positioning_analysis(
    contract_data: dict[str, list[dict]],
    staleness_days: int = 999,
) -> PositioningAnalysis:
    """
    Compute full positioning analysis across all contracts.

    Args:
        contract_data: {contract_key: [list of history dicts]}
                       Each history list sorted chronologically (oldest first).
        staleness_days: Days since latest COT report date.

    Returns:
        PositioningAnalysis with per-contract signals, aggregate score,
        divergences, and the S&P 500 positioning signal for forward_bias.
    """
    result = PositioningAnalysis()
    result.data_staleness_days = staleness_days

    # Compute per-contract positioning
    for contract_key, history in contract_data.items():
        cp = compute_contract_positioning(contract_key, history)
        result.contracts[contract_key] = cp

    if not result.contracts:
        return result

    # --- Aggregate Score ---
    # Map each contract's percentile to -1/+1 scale, then weight
    weighted_sum = 0.0
    weight_total = 0.0

    for key, cp in result.contracts.items():
        weight = CONTRACT_WEIGHTS.get(key, 0.10)
        # Map 0-100 percentile to -1/+1 (50th = 0, 0th = -1, 100th = +1)
        mapped = (cp.net_spec_percentile - 50) / 50.0
        weighted_sum += mapped * weight
        weight_total += weight

    if weight_total > 0:
        result.aggregate_score = round(weighted_sum / weight_total, 3)

    # --- Crowding Alert Count ---
    result.crowding_alert_count = sum(
        1 for cp in result.contracts.values()
        if cp.crowding in ("crowded_long", "crowded_short")
    )

    # --- S&P 500 Positioning Signal (for forward_bias) ---
    sp500 = result.contracts.get("sp500")
    if sp500 and sp500.confidence != "low":
        if sp500.net_spec_percentile <= 10:
            result.sp500_positioning_signal = "constructive"  # Crowded short = contrarian buy
        elif sp500.net_spec_percentile >= 90:
            result.sp500_positioning_signal = "cautious"      # Crowded long = contrarian sell
        else:
            result.sp500_positioning_signal = "neutral"
    else:
        result.sp500_positioning_signal = "neutral"

    # --- Cross-Asset Divergences ---
    result.divergences = _detect_divergences(result.contracts)

    return result


def _detect_divergences(contracts: dict[str, ContractPositioning]) -> list[str]:
    """
    Detect cross-asset positioning inconsistencies.

    These are situations where positioning across different asset classes
    tells a contradictory story — which often precedes sharp reversals.
    """
    divergences = []

    sp500 = contracts.get("sp500")
    gold = contracts.get("gold")
    treasury = contracts.get("treasury_10y")
    wti = contracts.get("wti_crude")
    dollar = contracts.get("dollar_index")

    # Risk-on vs safe-haven contradiction
    if sp500 and gold:
        if sp500.crowding == "crowded_long" and gold.crowding == "crowded_long":
            divergences.append(
                "Specs crowded long BOTH equities AND gold — inconsistent. "
                "Can't be risk-on and hedging simultaneously at extremes. "
                "One side will unwind."
            )
        if sp500.crowding == "crowded_short" and gold.crowding == "crowded_short":
            divergences.append(
                "Specs crowded short BOTH equities AND gold — extreme pessimism "
                "across asset classes. Historically precedes sharp reversal."
            )

    # Equity-Treasury positioning alignment
    if sp500 and treasury:
        if (sp500.crowding == "crowded_short" and
                treasury.net_spec_percentile > 80):
            divergences.append(
                "Specs short equities but long treasuries — classic recession "
                "positioning. If recession doesn't materialize, both unwind violently."
            )

    # Oil-Dollar relationship
    if wti and dollar:
        if (wti.crowding == "crowded_long" and
                dollar.crowding == "crowded_long"):
            divergences.append(
                "Specs long both oil AND dollar — unusual since strong dollar "
                "typically pressures commodity prices. Supply-driven oil thesis."
            )

    return divergences


# ============================================================================
# Data Notes Generator
# ============================================================================

def generate_positioning_data_notes(positioning: PositioningAnalysis) -> list[str]:
    """
    Generate plain-English positioning notes for the interpretation prompt.

    These notes are prepended to Claude's context so it understands
    speculative positioning BEFORE interpreting the regime data.
    """
    if not positioning.contracts:
        return ["- POSITIONING: No CFTC COT data available."]

    notes = []

    # Header
    staleness = positioning.data_staleness_days
    staleness_label = "normal" if staleness <= 10 else "stale" if staleness <= 21 else "VERY STALE"
    notes.append(
        f"- POSITIONING (CFTC COT, {staleness} days since last report, {staleness_label}):"
    )

    # Per-contract summary
    CONTRACT_LABELS = {
        "sp500": "S&P 500",
        "treasury_10y": "10Y Treasury",
        "wti_crude": "WTI Crude",
        "gold": "Gold",
        "dollar_index": "Dollar Index",
    }

    for key in ["sp500", "treasury_10y", "wti_crude", "gold", "dollar_index"]:
        cp = positioning.contracts.get(key)
        if cp is None:
            continue

        label = CONTRACT_LABELS.get(key, key)
        crowding_flag = ""
        if cp.crowding == "crowded_long":
            crowding_flag = " ★CROWDED LONG"
        elif cp.crowding == "crowded_short":
            crowding_flag = " ★CROWDED SHORT"

        # Direction label and percentile context to prevent misinterpretation
        direction = "LONG" if cp.net_spec > 0 else "SHORT"
        if cp.net_spec > 0:
            pctl_context = f"more long than {cp.net_spec_percentile:.0f}% of last 3yr"
        elif cp.net_spec_percentile > 50:
            pctl_context = f"less short than {cp.net_spec_percentile:.0f}% of last 3yr"
        else:
            pctl_context = f"more short than {100 - cp.net_spec_percentile:.0f}% of last 3yr"

        roc_dir = "adding" if cp.rate_of_change_4w > 0 else "reducing"
        notes.append(
            f"    {label}: net spec {cp.net_spec:+,} ({direction}), "
            f"{cp.net_spec_percentile:.0f}th percentile ({pctl_context}){crowding_flag}, "
            f"{roc_dir} {abs(cp.rate_of_change_4w):,}/4wk "
            f"({cp.confidence} confidence, {cp.lookback_weeks}wk history)"
        )

    # Aggregate
    agg_dir = "net short bias" if positioning.aggregate_score < -0.1 else \
              "net long bias" if positioning.aggregate_score > 0.1 else "balanced"
    notes.append(
        f"    AGGREGATE: {positioning.aggregate_score:+.2f} ({agg_dir}), "
        f"{positioning.crowding_alert_count} crowding alerts"
    )

    # Divergences
    for div in positioning.divergences:
        notes.append(f"    DIVERGENCE: {div}")

    # Contrarian context (no fabricated statistics)
    sp500 = positioning.contracts.get("sp500")
    if sp500 and sp500.crowding != "neutral":
        if sp500.crowding == "crowded_short":
            notes.append(
                "    CONTRARIAN NOTE: S&P 500 specs at extreme short positioning. "
                "Extreme positioning historically precedes contrarian reversals. "
                "Use when assessing forward bias — crowded short = asymmetry to upside."
            )
        elif sp500.crowding == "crowded_long":
            notes.append(
                "    CONTRARIAN NOTE: S&P 500 specs at extreme long positioning. "
                "Extreme long positioning means limited marginal buyers remain. "
                "Use when assessing forward bias — crowded long = asymmetry to downside."
            )

    return notes
