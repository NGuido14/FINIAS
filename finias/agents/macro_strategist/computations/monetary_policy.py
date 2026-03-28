"""
Monetary Policy & Liquidity Analysis Module

Tracks Fed policy stance, balance sheet dynamics, and system liquidity.
Net liquidity (Fed assets - TGA - reverse repo) is the single most
underappreciated driver of risk assets. When net liquidity expands,
risk assets rally. When it contracts, they struggle.

This module also tracks financial conditions indices and credit creation
to gauge the transmission of monetary policy to the real economy.

All computation is pure Python. No API calls. No Claude.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class MonetaryPolicyAnalysis:
    """Complete monetary policy and liquidity assessment."""

    # Policy rates
    fed_funds_current: Optional[float] = None
    fed_funds_target_upper: Optional[float] = None
    fed_funds_target_lower: Optional[float] = None

    # Fed balance sheet
    fed_total_assets: Optional[float] = None          # Billions
    fed_treasuries: Optional[float] = None
    fed_mbs: Optional[float] = None
    balance_sheet_change_4w: Optional[float] = None   # 4-week change in billions
    balance_sheet_change_13w: Optional[float] = None  # 13-week change
    qt_monthly_pace: Optional[float] = None           # Estimated monthly runoff

    # Liquidity components
    tga_level: Optional[float] = None
    reverse_repo_level: Optional[float] = None
    bank_reserves: Optional[float] = None

    # Net liquidity
    net_liquidity: Optional[float] = None             # Billions
    net_liquidity_change_4w: Optional[float] = None
    net_liquidity_change_13w: Optional[float] = None
    net_liquidity_change_26w: Optional[float] = None
    net_liquidity_trend: Optional[str] = None         # expanding, stable, contracting

    # Financial conditions
    nfci: Optional[float] = None                      # Negative=loose, positive=tight
    nfci_change_4w: Optional[float] = None
    nfci_trend: Optional[str] = None                  # loosening, stable, tightening
    financial_stress: Optional[float] = None          # St. Louis FSI

    # Credit creation
    bank_credit_yoy: Optional[float] = None
    consumer_credit_yoy: Optional[float] = None
    m2_yoy: Optional[float] = None

    # Classifications
    policy_stance: str = "unknown"                    # hawkish, neutral, dovish, emergency
    policy_direction: str = "unknown"                 # tightening, on_hold, easing
    liquidity_regime: str = "unknown"                 # ample, adequate, tightening, scarce

    # Scores
    policy_score: float = 0.0                         # -1 (very tight) to +1 (very loose)
    liquidity_score: float = 0.0                      # -1 (draining) to +1 (flooding)

    def to_dict(self) -> dict:
        return {
            "policy_rates": {
                "fed_funds": self.fed_funds_current,
                "target_upper": self.fed_funds_target_upper,
                "target_lower": self.fed_funds_target_lower,
            },
            "balance_sheet": {
                "total_assets": self.fed_total_assets,
                "treasuries": self.fed_treasuries,
                "mbs": self.fed_mbs,
                "change_4w": self.balance_sheet_change_4w,
                "change_13w": self.balance_sheet_change_13w,
                "qt_monthly_pace": self.qt_monthly_pace,
            },
            "liquidity": {
                "net_liquidity": self.net_liquidity,
                "tga": self.tga_level,
                "reverse_repo": self.reverse_repo_level,
                "bank_reserves": self.bank_reserves,
                "change_4w": self.net_liquidity_change_4w,
                "change_13w": self.net_liquidity_change_13w,
                "change_26w": self.net_liquidity_change_26w,
                "trend": self.net_liquidity_trend,
            },
            "financial_conditions": {
                "nfci": self.nfci,
                "nfci_change_4w": self.nfci_change_4w,
                "nfci_trend": self.nfci_trend,
                "stress_index": self.financial_stress,
            },
            "credit": {
                "bank_credit_yoy": self.bank_credit_yoy,
                "consumer_credit_yoy": self.consumer_credit_yoy,
                "m2_yoy": self.m2_yoy,
            },
            "policy_stance": self.policy_stance,
            "policy_direction": self.policy_direction,
            "liquidity_regime": self.liquidity_regime,
            "policy_score": self.policy_score,
            "liquidity_score": self.liquidity_score,
        }


def analyze_monetary_policy(
    fed_funds: list[dict],
    fed_target_upper: list[dict],
    fed_target_lower: list[dict],
    fed_total_assets: list[dict],
    fed_treasuries: list[dict],
    fed_mbs: list[dict],
    tga: list[dict],
    reverse_repo: list[dict],
    bank_reserves: list[dict],
    nfci_series: list[dict],
    stress_series: list[dict],
    bank_credit: list[dict],
    consumer_credit: list[dict],
    m2_series: list[dict],
) -> MonetaryPolicyAnalysis:
    """
    Perform complete monetary policy and liquidity analysis.

    All inputs are lists of {"date": str, "value": float} sorted ascending.
    """
    result = MonetaryPolicyAnalysis()

    # --- Policy Rates ---
    result.fed_funds_current = _latest(fed_funds)
    result.fed_funds_target_upper = _latest(fed_target_upper)
    result.fed_funds_target_lower = _latest(fed_target_lower)

    # --- Balance Sheet ---
    result.fed_total_assets = _latest(fed_total_assets)
    result.fed_treasuries = _latest(fed_treasuries)
    result.fed_mbs = _latest(fed_mbs)
    result.balance_sheet_change_4w = _change_over_observations(fed_total_assets, 4)
    result.balance_sheet_change_13w = _change_over_observations(fed_total_assets, 13)

    # Estimate monthly QT pace from 13-week change
    if result.balance_sheet_change_13w is not None:
        result.qt_monthly_pace = result.balance_sheet_change_13w / 3.0

    # --- Net Liquidity ---
    net_liq_current, net_liq_series = _compute_net_liquidity_series(
        fed_total_assets, tga, reverse_repo
    )
    result.net_liquidity = net_liq_current
    result.tga_level = _latest(tga)
    result.reverse_repo_level = _latest(reverse_repo)
    result.bank_reserves = _latest(bank_reserves)

    if net_liq_series is not None and len(net_liq_series) > 0:
        result.net_liquidity_change_4w = _change_over_observations(net_liq_series, 4)
        result.net_liquidity_change_13w = _change_over_observations(net_liq_series, 13)
        result.net_liquidity_change_26w = _change_over_observations(net_liq_series, 26)
        result.net_liquidity_trend = _classify_liquidity_trend(net_liq_series)

    # --- Financial Conditions ---
    result.nfci = _latest(nfci_series)
    result.nfci_change_4w = _change_over_observations(nfci_series, 4)
    result.nfci_trend = _classify_nfci_trend(nfci_series)
    result.financial_stress = _latest(stress_series)

    # --- Credit Creation ---
    result.bank_credit_yoy = _compute_yoy_growth(bank_credit)
    result.consumer_credit_yoy = _compute_yoy_growth(consumer_credit)
    result.m2_yoy = _compute_yoy_growth(m2_series)

    # --- Classifications ---
    result.policy_stance = _classify_policy_stance(
        result.fed_funds_current, result.qt_monthly_pace, result.nfci
    )
    result.policy_direction = _classify_policy_direction(fed_target_upper)
    result.liquidity_regime = _classify_liquidity_regime(
        result.net_liquidity_trend, result.nfci, result.bank_reserves
    )

    # --- Scores ---
    result.policy_score = _compute_policy_score(result)
    result.liquidity_score = _compute_liquidity_score(result)

    return result


def _latest(series: list[dict]) -> Optional[float]:
    if not series:
        return None
    return series[-1]["value"]


def _change_over_observations(series: list[dict], n: int) -> Optional[float]:
    """Change over last N observations (for weekly data, N=4 ≈ 1 month)."""
    if not series or len(series) <= n:
        return None
    return series[-1]["value"] - series[-(n + 1)]["value"]


def _compute_net_liquidity_series(
    fed_assets: list[dict],
    tga: list[dict],
    reverse_repo: list[dict],
) -> tuple[Optional[float], Optional[list[dict]]]:
    """
    Net Liquidity = Fed Total Assets - TGA - Reverse Repo

    Aligns series by date and computes the net liquidity time series.
    Returns (current_value, full_series).
    """
    if not fed_assets or not tga or not reverse_repo:
        return None, None

    # Build date-keyed lookups
    tga_map = {d["date"]: d["value"] for d in tga}
    rrp_map = {d["date"]: d["value"] for d in reverse_repo}

    net_liq_series = []
    for point in fed_assets:
        date_key = point["date"]
        tga_val = tga_map.get(date_key)
        rrp_val = rrp_map.get(date_key)

        if tga_val is not None and rrp_val is not None:
            # All values in millions from FRED — convert to billions
            fa = point["value"]  # Already in millions
            net = fa - tga_val - rrp_val
            net_liq_series.append({"date": date_key, "value": net})

    if not net_liq_series:
        # Fallback: use latest values even if dates don't align perfectly
        fa = fed_assets[-1]["value"]
        tga_v = tga[-1]["value"]
        rrp_v = reverse_repo[-1]["value"]
        current = fa - tga_v - rrp_v
        return current, None

    return net_liq_series[-1]["value"], net_liq_series


def _classify_liquidity_trend(net_liq_series: list[dict]) -> str:
    """Classify net liquidity trend from series."""
    if len(net_liq_series) < 13:
        return "unknown"

    # 13-week change as % of level
    current = net_liq_series[-1]["value"]
    past = net_liq_series[-13]["value"] if len(net_liq_series) >= 13 else net_liq_series[0]["value"]

    if current == 0:
        return "unknown"

    pct_change = (current - past) / abs(past) * 100

    if pct_change > 1.0:
        return "expanding"
    elif pct_change < -1.0:
        return "contracting"
    return "stable"


def _classify_nfci_trend(nfci: list[dict]) -> Optional[str]:
    """Classify NFCI trend. Rising NFCI = tightening conditions."""
    if len(nfci) < 4:
        return None
    change = nfci[-1]["value"] - nfci[-4]["value"]
    if change > 0.05:
        return "tightening"
    elif change < -0.05:
        return "loosening"
    return "stable"


def _compute_yoy_growth(series: list[dict]) -> Optional[float]:
    """Compute year-over-year growth rate."""
    if not series or len(series) < 2:
        return None

    current = series[-1]["value"]

    # Find value approximately 12 months ago
    # For monthly data: 12 observations ago
    # For weekly data: 52 observations ago
    # Heuristic: use the observation closest to 12 months back
    target_idx = None
    if len(series) >= 52:
        target_idx = -52  # Weekly
    elif len(series) >= 12:
        target_idx = -12  # Monthly
    else:
        target_idx = 0

    past = series[target_idx]["value"]

    if past == 0:
        return None

    return (current - past) / abs(past) * 100


def _classify_policy_stance(
    fed_funds: Optional[float],
    qt_pace: Optional[float],
    nfci: Optional[float],
) -> str:
    """
    Classify monetary policy stance.

    Considers rate level, balance sheet direction, and resulting conditions.
    """
    if fed_funds is None:
        return "unknown"

    # Rate-based assessment
    is_high_rates = fed_funds > 4.0
    is_low_rates = fed_funds < 1.5
    is_zero = fed_funds < 0.5

    # Balance sheet
    is_qt = qt_pace is not None and qt_pace < -10  # Shrinking by >$10B/month
    is_qe = qt_pace is not None and qt_pace > 10   # Growing by >$10B/month

    if is_zero and is_qe:
        return "emergency"
    elif is_low_rates or is_qe:
        return "dovish"
    elif is_high_rates and is_qt:
        return "hawkish"
    elif is_high_rates:
        return "hawkish"
    else:
        return "neutral"


def _classify_policy_direction(fed_target_upper: list[dict]) -> str:
    """Determine if policy is tightening, on hold, or easing."""
    if len(fed_target_upper) < 60:
        return "unknown"

    current = fed_target_upper[-1]["value"]
    three_months_ago = fed_target_upper[-60]["value"] if len(fed_target_upper) >= 60 else fed_target_upper[0]["value"]

    diff = current - three_months_ago

    if diff > 0.10:
        return "tightening"
    elif diff < -0.10:
        return "easing"
    return "on_hold"


def _classify_liquidity_regime(
    net_liq_trend: Optional[str],
    nfci: Optional[float],
    bank_reserves: Optional[float],
) -> str:
    """Classify overall liquidity regime."""
    if net_liq_trend == "contracting":
        if nfci is not None and nfci > 0.25:
            return "scarce"
        return "tightening"
    elif net_liq_trend == "expanding":
        if nfci is not None and nfci < -0.25:
            return "ample"
        return "adequate"
    else:
        return "adequate"


def _compute_policy_score(result: MonetaryPolicyAnalysis) -> float:
    """
    Compute -1 (very tight) to +1 (very loose) policy score.
    """
    score = 0.0

    # Rate level
    if result.fed_funds_current is not None:
        if result.fed_funds_current < 1.0:
            score += 0.4
        elif result.fed_funds_current < 2.5:
            score += 0.2
        elif result.fed_funds_current < 4.0:
            score -= 0.1
        elif result.fed_funds_current < 5.0:
            score -= 0.3
        else:
            score -= 0.4

    # Direction
    if result.policy_direction == "easing":
        score += 0.3
    elif result.policy_direction == "tightening":
        score -= 0.3

    # Financial conditions
    if result.nfci is not None:
        score -= result.nfci * 0.3  # NFCI positive = tight = bearish

    return max(-1.0, min(1.0, score))


def _compute_liquidity_score(result: MonetaryPolicyAnalysis) -> float:
    """
    Compute -1 (draining) to +1 (flooding) liquidity score.
    """
    score = 0.0

    # Net liquidity trend
    if result.net_liquidity_trend == "expanding":
        score += 0.4
    elif result.net_liquidity_trend == "contracting":
        score -= 0.4

    # M2 growth
    if result.m2_yoy is not None:
        if result.m2_yoy > 5:
            score += 0.2
        elif result.m2_yoy < 0:
            score -= 0.3

    # Bank credit
    if result.bank_credit_yoy is not None:
        if result.bank_credit_yoy > 5:
            score += 0.15
        elif result.bank_credit_yoy < 0:
            score -= 0.2

    # NFCI as liquidity proxy
    if result.nfci is not None:
        score -= result.nfci * 0.25

    return max(-1.0, min(1.0, score))
