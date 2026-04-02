"""
Cross-Asset Signal Analysis Module

Monitors relationships between asset classes that institutional desks watch.
Each signal tells a different story about the market's expectations for
growth, inflation, risk appetite, and systemic stress.

Signals:
  - Dollar (DXY): Strong dollar = headwind for risk assets and EM
  - Credit spreads (HY OAS): Credit market's view on default risk
  - Copper/Gold ratio: Growth expectations proxy (leads equities)
  - Oil dynamics: Inflation input AND growth indicator
  - Stock-bond correlation: When both fall, diversification breaks down
  - Small cap/Large cap ratio: Risk appetite gauge
  - Credit-equity divergence: Warning when credit and stocks disagree

All computation is pure Python. No API calls. No Claude.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CrossAssetAnalysis:
    """Expanded cross-asset signal assessment."""

    # Dollar
    dxy_level: Optional[float] = None
    dxy_trend: Optional[str] = None           # strengthening, weakening, stable
    dxy_change_30d: Optional[float] = None

    # Credit
    hy_spread: Optional[float] = None         # High yield OAS
    hy_spread_trend: Optional[str] = None     # tightening, widening, stable
    hy_spread_change_30d: Optional[float] = None
    credit_stress: bool = False               # True if HY spread > 500bps

    # Inflation expectations
    breakeven_5y: Optional[float] = None
    breakeven_10y: Optional[float] = None
    inflation_expectations: str = "unknown"   # anchored, rising, falling

    # Copper/Gold ratio (growth proxy)
    copper_gold_ratio: Optional[float] = None
    copper_gold_change_20d: Optional[float] = None
    copper_gold_signal: str = "neutral"       # growth_optimism, growth_pessimism, neutral

    # Oil dynamics
    oil_price: Optional[float] = None
    oil_change_20d_pct: Optional[float] = None
    oil_signal: str = "neutral"               # demand_driven, supply_shock, deflationary, neutral

    # Brent crude (global benchmark — more relevant than WTI during geopolitical shocks)
    brent_price: Optional[float] = None
    brent_change_20d_pct: Optional[float] = None
    wti_brent_spread: Optional[float] = None       # WTI - Brent (negative = Brent premium)
    wti_brent_spread_widening: bool = False         # True if |spread| > $5 (geopolitical signal)

    # Stock-bond correlation
    stock_bond_corr_20d: Optional[float] = None
    stock_bond_corr_60d: Optional[float] = None
    risk_parity_stress: bool = False          # True when stocks and bonds fall together

    # Small cap / Large cap (risk appetite)
    iwm_spy_ratio: Optional[float] = None
    iwm_spy_change_20d: Optional[float] = None
    risk_appetite: str = "neutral"            # strong, moderate, weak, risk_averse

    # Credit-equity divergence
    credit_equity_divergence: bool = False
    divergence_type: Optional[str] = None     # credit_warning, equity_warning

    # EM stress
    em_relative_performance_20d: Optional[float] = None
    em_stress: bool = False

    # Composite
    cross_asset_score: float = 0.0            # -1 to 1 — positive = risk-on

    # Correlation matrix (rolling correlations, betas, convexity)
    correlation_matrix: Optional[dict] = None  # CorrelationMatrix.to_dict() output

    def to_dict(self) -> dict:
        return {
            "dollar": {
                "dxy": self.dxy_level,
                "trend": self.dxy_trend,
                "change_30d": self.dxy_change_30d,
            },
            "credit": {
                "hy_spread": self.hy_spread,
                "trend": self.hy_spread_trend,
                "change_30d": self.hy_spread_change_30d,
                "stress": self.credit_stress,
            },
            "inflation_expectations": {
                "breakeven_5y": self.breakeven_5y,
                "breakeven_10y": self.breakeven_10y,
                "direction": self.inflation_expectations,
            },
            "copper_gold": {
                "ratio": self.copper_gold_ratio,
                "ratio_change_20d_pct": self.copper_gold_change_20d,
                "_note": "Percentage change in copper/gold PRICE RATIO over 20 days. Example: +8.6 means the ratio rose 8.6%, indicating growth optimism.",
                "signal": self.copper_gold_signal,
            },
            "oil": {
                "wti_price": self.oil_price,
                "wti_change_20d_pct": self.oil_change_20d_pct,
                "brent_price": self.brent_price,
                "brent_change_20d_pct": self.brent_change_20d_pct,
                "wti_brent_spread": self.wti_brent_spread,
                "wti_brent_spread_widening": self.wti_brent_spread_widening,
                "_spread_note": "WTI minus Brent. Negative means Brent premium. Spread > $5 signals geopolitical supply disruption affecting global more than domestic.",
                "signal": self.oil_signal,
            },
            "stock_bond_correlation": {
                "corr_20d": self.stock_bond_corr_20d,
                "corr_60d": self.stock_bond_corr_60d,
                "risk_parity_stress": self.risk_parity_stress,
            },
            "risk_appetite": {
                "iwm_spy_price_ratio": self.iwm_spy_ratio,
                "iwm_vs_spy_relative_return_20d_percentage_points": self.iwm_spy_change_20d,
                "_unit": "Relative return in percentage points. Example: -0.25 means IWM returned 0.25 percentage points LESS than SPY. This is NOT -25%.",
                "appetite": self.risk_appetite,
            },
            "credit_equity_divergence": {
                "divergence": self.credit_equity_divergence,
                "type": self.divergence_type,
            },
            "em": {
                "relative_return_vs_spy_20d_percentage_points": self.em_relative_performance_20d,
                "_unit": "EEM 20-day return minus SPY 20-day return, in percentage points. Example: -2.6 means EEM returned 2.6 percentage points less than SPY.",
                "stress": self.em_stress,
            },
            "cross_asset_score": self.cross_asset_score,
            "correlations": self.correlation_matrix,
        }


def analyze_cross_assets(
    dxy_series: list[dict],
    hy_spread_series: list[dict],
    breakeven_5y: list[dict],
    breakeven_10y: list[dict],
    # New parameters
    copper_prices: list[dict] = None,    # CPER or JJC ETF prices
    gold_prices: list[dict] = None,      # GLD prices
    oil_series: list[dict] = None,       # WTI from FRED (DCOILWTICO)
    brent_series: list[dict] = None,     # Brent from FRED (DCOILBRENTEU)
    spy_prices: list[dict] = None,       # SPY for correlations
    tlt_prices: list[dict] = None,       # TLT for stock-bond correlation
    iwm_prices: list[dict] = None,       # IWM for risk appetite
    hyg_prices: list[dict] = None,       # HYG for credit-equity divergence
    eem_prices: list[dict] = None,       # EEM for EM stress
    vix_series: list[dict] = None,       # VIXCLS for correlation regime splits
) -> CrossAssetAnalysis:
    """Analyze cross-asset signals with full intermarket analysis."""

    result = CrossAssetAnalysis()

    # === Dollar ===
    result.dxy_level = _latest(dxy_series)
    result.dxy_trend = _classify_trend_direction(dxy_series, 30)
    result.dxy_change_30d = _change_over_days(dxy_series, 30)

    # === Credit ===
    result.hy_spread = _latest(hy_spread_series)
    result.hy_spread_trend = _classify_spread_trend(hy_spread_series, 30)
    result.hy_spread_change_30d = _change_over_days(hy_spread_series, 30)
    result.credit_stress = result.hy_spread is not None and result.hy_spread > 5.0

    # === Inflation Expectations ===
    result.breakeven_5y = _latest(breakeven_5y)
    result.breakeven_10y = _latest(breakeven_10y)
    result.inflation_expectations = _classify_inflation_expectations(breakeven_5y)

    # === Copper/Gold Ratio ===
    if copper_prices and gold_prices:
        _analyze_copper_gold(result, copper_prices, gold_prices)

    # === Oil ===
    if oil_series:
        _analyze_oil(result, oil_series)

    # Brent crude and WTI-Brent spread
    if brent_series and len(brent_series) > 0:
        result.brent_price = brent_series[-1]["value"]
        if len(brent_series) >= 20:
            old_brent = brent_series[-20]["value"]
            if old_brent > 0:
                result.brent_change_20d_pct = (brent_series[-1]["value"] / old_brent - 1) * 100

        # Compute WTI-Brent spread
        if result.oil_price is not None and result.brent_price is not None:
            result.wti_brent_spread = round(result.oil_price - result.brent_price, 2)
            result.wti_brent_spread_widening = abs(result.wti_brent_spread) > 5.0

    # === Stock-Bond Correlation ===
    if spy_prices and tlt_prices:
        _analyze_stock_bond_correlation(result, spy_prices, tlt_prices)

    # === Small Cap / Large Cap Risk Appetite ===
    if iwm_prices and spy_prices:
        _analyze_risk_appetite(result, iwm_prices, spy_prices)

    # === Credit-Equity Divergence ===
    if hyg_prices and spy_prices:
        _analyze_credit_equity_divergence(result, hyg_prices, spy_prices)

    # === EM Stress ===
    if eem_prices and spy_prices:
        _analyze_em_stress(result, eem_prices, spy_prices)

    # === Composite Score ===
    result.cross_asset_score = _compute_cross_asset_score(result)

    # === Correlation Matrix (rolling correlations, betas, convexity) ===
    try:
        from finias.agents.macro_strategist.computations.correlation import compute_correlation_matrix
        from datetime import date as _date
        corr_matrix = compute_correlation_matrix(
            spy=spy_prices,
            tlt=tlt_prices,
            gld=gold_prices,
            hyg=hyg_prices,
            oil=oil_series,
            dxy=dxy_series,
            vix=vix_series,
            as_of_date=str(_date.today()),
        )
        result.correlation_matrix = corr_matrix.to_dict()
    except Exception as e:
        import logging
        logging.getLogger("finias.agent.macro_strategist").warning(
            f"Correlation matrix computation failed: {e}"
        )
        result.correlation_matrix = None

    return result


# === Signal Analysis Functions ===

def _analyze_copper_gold(result: CrossAssetAnalysis, copper: list[dict], gold: list[dict]):
    """
    Copper/Gold ratio: growth expectations proxy.
    Copper is growth-sensitive, gold is fear-sensitive.
    Rising ratio = growth optimism. Falling = growth pessimism.
    Often leads equity markets by 2-4 weeks.
    """
    min_len = min(len(copper), len(gold))
    if min_len < 20:
        return

    cu = np.array([p["close"] for p in copper[-min_len:]])
    au = np.array([p["close"] for p in gold[-min_len:]])

    ratio = cu / au
    result.copper_gold_ratio = float(ratio[-1])

    if len(ratio) >= 20:
        result.copper_gold_change_20d = float(
            (ratio[-1] / ratio[-20] - 1) * 100
        )

        if result.copper_gold_change_20d > 2.0:
            result.copper_gold_signal = "growth_optimism"
        elif result.copper_gold_change_20d < -2.0:
            result.copper_gold_signal = "growth_pessimism"
        else:
            result.copper_gold_signal = "neutral"


def _analyze_oil(result: CrossAssetAnalysis, oil: list[dict]):
    """
    Oil is simultaneously an inflation input and growth indicator.
    Rising with strong growth = healthy demand.
    Rising with weak growth = supply shock (stagflationary).
    """
    result.oil_price = _latest(oil)

    if len(oil) >= 20:
        old = oil[-20]["value"]
        if old > 0:
            result.oil_change_20d_pct = (oil[-1]["value"] / old - 1) * 100

            if result.oil_change_20d_pct > 10:
                result.oil_signal = "supply_shock"  # Agent will cross-ref with growth
            elif result.oil_change_20d_pct > 3:
                result.oil_signal = "demand_driven"
            elif result.oil_change_20d_pct < -10:
                result.oil_signal = "deflationary"
            else:
                result.oil_signal = "neutral"


def _analyze_stock_bond_correlation(
    result: CrossAssetAnalysis,
    spy: list[dict],
    tlt: list[dict],
):
    """
    Stock-bond correlation. Normally negative (bonds hedge stocks).
    When positive (both falling together), risk parity portfolios unwind,
    creating self-reinforcing selling pressure.
    """
    min_len = min(len(spy), len(tlt))
    if min_len < 61:
        return

    spy_closes = np.array([p["close"] for p in spy[-min_len:]])
    tlt_closes = np.array([p["close"] for p in tlt[-min_len:]])

    spy_ret = np.diff(np.log(spy_closes))
    tlt_ret = np.diff(np.log(tlt_closes))

    if len(spy_ret) >= 20:
        result.stock_bond_corr_20d = float(
            np.corrcoef(spy_ret[-20:], tlt_ret[-20:])[0, 1]
        )

    if len(spy_ret) >= 60:
        result.stock_bond_corr_60d = float(
            np.corrcoef(spy_ret[-60:], tlt_ret[-60:])[0, 1]
        )

    # Risk parity stress: positive correlation with both declining
    if result.stock_bond_corr_20d is not None and result.stock_bond_corr_20d > 0.3:
        # Check if both are declining
        spy_20d_ret = (spy_closes[-1] / spy_closes[-20] - 1) * 100
        tlt_20d_ret = (tlt_closes[-1] / tlt_closes[-20] - 1) * 100
        if spy_20d_ret < -2 and tlt_20d_ret < -2:
            result.risk_parity_stress = True


def _analyze_risk_appetite(
    result: CrossAssetAnalysis,
    iwm: list[dict],
    spy: list[dict],
):
    """
    IWM vs SPY relative return: risk appetite gauge.
    Small caps outperforming = strong risk appetite.
    Large caps outperforming = defensive / flight to quality.

    Uses relative return difference (IWM return minus SPY return)
    instead of ratio change, because ratio change amplifies small
    moves when price levels differ significantly.
    """
    min_len = min(len(iwm), len(spy))
    if min_len < 20:
        return

    iwm_closes = np.array([p["close"] for p in iwm[-min_len:]])
    spy_closes = np.array([p["close"] for p in spy[-min_len:]])

    # Price ratio (for reference only)
    result.iwm_spy_ratio = float(iwm_closes[-1] / spy_closes[-1])

    if len(iwm_closes) >= 20:
        # Relative return: IWM 20-day return minus SPY 20-day return
        # This gives the actual performance gap in percentage points
        iwm_ret_20d = (iwm_closes[-1] / iwm_closes[-20] - 1) * 100
        spy_ret_20d = (spy_closes[-1] / spy_closes[-20] - 1) * 100
        result.iwm_spy_change_20d = float(iwm_ret_20d - spy_ret_20d)

        # Thresholds based on relative return (smaller than ratio change)
        if result.iwm_spy_change_20d > 3.0:
            result.risk_appetite = "strong"
        elif result.iwm_spy_change_20d > 0.5:
            result.risk_appetite = "moderate"
        elif result.iwm_spy_change_20d > -3.0:
            result.risk_appetite = "weak"
        else:
            result.risk_appetite = "risk_averse"


def _analyze_credit_equity_divergence(
    result: CrossAssetAnalysis,
    hyg: list[dict],
    spy: list[dict],
):
    """
    Credit-equity divergence. When HYG (credit) and SPY (equity) disagree,
    one of them is wrong. Historically, credit is more often right.
    """
    min_len = min(len(hyg), len(spy))
    if min_len < 20:
        return

    hyg_ret = (hyg[-1]["close"] / hyg[-20]["close"] - 1) * 100 if min_len >= 20 else None
    spy_ret = (spy[-1]["close"] / spy[-20]["close"] - 1) * 100 if min_len >= 20 else None

    if hyg_ret is not None and spy_ret is not None:
        # Significant divergence: one up >2% and other down >2% over 20 days
        if spy_ret > 2 and hyg_ret < -1:
            result.credit_equity_divergence = True
            result.divergence_type = "credit_warning"  # Credit says risk, equity ignoring
        elif hyg_ret > 1 and spy_ret < -2:
            result.credit_equity_divergence = True
            result.divergence_type = "equity_warning"  # Equity says risk, credit calm


def _analyze_em_stress(
    result: CrossAssetAnalysis,
    eem: list[dict],
    spy: list[dict],
):
    """EM stress: EEM underperformance + dollar strength = EM pressure."""
    min_len = min(len(eem), len(spy))
    if min_len < 20:
        return

    eem_ret = (eem[-1]["close"] / eem[-20]["close"] - 1) * 100
    spy_ret = (spy[-1]["close"] / spy[-20]["close"] - 1) * 100

    result.em_relative_performance_20d = eem_ret - spy_ret

    # EM stress: underperforming by >5% over 20 days while dollar strengthening
    if result.em_relative_performance_20d < -5:
        result.em_stress = True


# === Helper Functions ===

def _latest(series: list[dict]) -> Optional[float]:
    if not series:
        return None
    return series[-1]["value"]


def _change_over_days(series: list[dict], days: int) -> Optional[float]:
    if len(series) <= days:
        return None
    return series[-1]["value"] - series[-(days + 1)]["value"]


def _classify_trend_direction(series: list[dict], window: int) -> Optional[str]:
    """Classify trend direction."""
    if len(series) < window:
        return None
    values = [s["value"] for s in series[-window:]]
    if values[0] == 0:
        return "stable"
    change_pct = (values[-1] - values[0]) / abs(values[0]) * 100
    if change_pct > 2:
        return "strengthening"
    elif change_pct < -2:
        return "weakening"
    return "stable"


def _classify_spread_trend(series: list[dict], window: int) -> Optional[str]:
    if len(series) < window:
        return None
    change = series[-1]["value"] - series[-window]["value"]
    if change > 0.15:
        return "widening"
    elif change < -0.15:
        return "tightening"
    return "stable"


def _classify_inflation_expectations(be_5y: list[dict]) -> str:
    if not be_5y or len(be_5y) < 20:
        return "unknown"
    recent = np.mean([s["value"] for s in be_5y[-5:]])
    earlier = np.mean([s["value"] for s in be_5y[-25:-20]]) if len(be_5y) >= 25 else be_5y[0]["value"]
    if recent - earlier > 0.15:
        return "rising"
    elif earlier - recent > 0.15:
        return "falling"
    return "anchored"


def _compute_cross_asset_score(result: CrossAssetAnalysis) -> float:
    """
    Composite cross-asset score: -1 (risk-off) to +1 (risk-on).
    Now uses 8 signal sources instead of 2.
    """
    score = 0.0
    signal_count = 0

    # Dollar (weight ~15%)
    if result.dxy_trend == "weakening":
        score += 0.15
    elif result.dxy_trend == "strengthening":
        score -= 0.15
    signal_count += 1

    # Credit spreads (weight ~20%)
    if result.credit_stress:
        score -= 0.30
    elif result.hy_spread is not None:
        if result.hy_spread < 3.0:
            score += 0.15
        elif result.hy_spread < 4.0:
            score += 0.05
        elif result.hy_spread > 4.5:
            score -= 0.10
    if result.hy_spread_trend == "tightening":
        score += 0.05
    elif result.hy_spread_trend == "widening":
        score -= 0.10
    signal_count += 1

    # Copper/Gold (weight ~15%)
    if result.copper_gold_signal == "growth_optimism":
        score += 0.15
    elif result.copper_gold_signal == "growth_pessimism":
        score -= 0.15
    signal_count += 1

    # Oil (weight ~10%)
    if result.oil_signal == "demand_driven":
        score += 0.05
    elif result.oil_signal == "supply_shock":
        score -= 0.10
    elif result.oil_signal == "deflationary":
        score -= 0.05
    signal_count += 1

    # Stock-bond correlation (weight ~15%)
    if result.risk_parity_stress:
        score -= 0.20
    elif result.stock_bond_corr_20d is not None and result.stock_bond_corr_20d > 0.2:
        score -= 0.10  # Positive correlation is concerning even without both declining
    signal_count += 1

    # Risk appetite (weight ~10%)
    appetite_scores = {"strong": 0.10, "moderate": 0.05, "weak": -0.05, "risk_averse": -0.15}
    score += appetite_scores.get(result.risk_appetite, 0)
    signal_count += 1

    # Credit-equity divergence (weight ~10%)
    if result.credit_equity_divergence:
        if result.divergence_type == "credit_warning":
            score -= 0.15  # Credit is usually right
        elif result.divergence_type == "equity_warning":
            score -= 0.05
    signal_count += 1

    # EM stress (weight ~5%)
    if result.em_stress:
        score -= 0.10
    signal_count += 1

    return max(-1.0, min(1.0, score))
