"""
FINIAS Historical Data Layer

Layer 1: compute_historical_trajectory() — pre-computed during morning refresh,
         stored in Redis, gives Director always-available trend context.

Layer 2: query_history() — on-demand deep query tool. Claude picks metrics,
         date ranges, and options. Returns formatted data for synthesis.

Both layers query existing PostgreSQL tables. No new tables needed.
"""

from __future__ import annotations
from datetime import date, timedelta
from typing import Optional
import json
import logging
import math

logger = logging.getLogger("finias.history")


# ============================================================================
# ALLOWED METRICS — Claude sees this list in the tool description
# Maps clean names to (table, column_or_series) tuples
# ============================================================================

# Metrics from macro_data_matrix (date-aligned, preferred)
MATRIX_METRICS = {
    # Inflation
    "core_pce": "core_pce",
    "cpi": "cpi_all",
    "cpi_core": "cpi_core",
    "sticky_cpi": "sticky_cpi",
    "pce": "pce",
    "ppi": "ppi_all",
    "trimmed_mean_pce": "trimmed_mean_pce",
    # Rates & Yields
    "fed_funds": "fedfunds",
    "dgs2": "dgs2",
    "dgs5": "dgs5",
    "dgs10": "dgs10",
    "dgs30": "dgs30",
    "spread_2s10s": "t10y2y",
    "spread_3m10y": "t10y3m",
    "hy_spread": "hy_oas",
    "breakeven_5y": "breakeven_5y",
    "breakeven_10y": "breakeven_10y",
    "tips_5y": "tips_5y",
    "tips_10y": "tips_10y",
    # Volatility
    "vix": "vix",
    # Liquidity
    "net_liquidity": "net_liquidity",
    "fed_total_assets": "fed_total_assets",
    "reverse_repo": "reverse_repo",
    "tga_balance": "tga_balance",
    "m2": "m2",
    "nfci": "nfci",
    # Labor
    "unemployment": "unemployment",
    "initial_claims": "initial_claims",
    "nonfarm_payrolls": "nonfarm_payrolls",
    "continuing_claims": "continuing_claims",
    "avg_hourly_earnings": "avg_hourly_earnings",
    # Activity & Sentiment
    "consumer_sentiment": "consumer_sentiment",
    "industrial_production": "industrial_production",
    "building_permits": "building_permits",
    "housing_starts": "housing_starts",
    "retail_sales": "retail_sales",
    "capacity_utilization": "capacity_utilization",
    "durable_goods": "durable_goods",
    # Commodities & Dollar
    "oil_wti": "oil_wti",
    "oil_brent": "oil_brent",
    "dxy": "dxy",
}

# Market data (from market_data_daily, requires different query)
MARKET_METRICS = {
    "spy": "SPY", "qqq": "QQQ", "iwm": "IWM",
    "tlt": "TLT", "hyg": "HYG", "gld": "GLD",
    "eem": "EEM", "rsp": "RSP",
    "xle": "XLE", "xlf": "XLF", "xlk": "XLK",
    "xlv": "XLV", "xli": "XLI", "xlc": "XLC",
    "xlu": "XLU", "xlb": "XLB", "xly": "XLY",
    "xlre": "XLRE", "xlp": "XLP",
}

ALL_METRIC_NAMES = sorted(list(MATRIX_METRICS.keys()) + list(MARKET_METRICS.keys()))


# ============================================================================
# AUTO-SAMPLING — keeps token budget manageable
# ============================================================================

def auto_sample(rows: list[dict], date_key: str = "obs_date", max_points: int = 24) -> list[dict]:
    """
    Downsample a time series to at most max_points.
    Always keeps the first and last observation.
    Selects evenly spaced points between them.
    """
    if len(rows) <= max_points:
        return rows
    if len(rows) < 2:
        return rows

    indices = [0]
    step = (len(rows) - 1) / (max_points - 1)
    for i in range(1, max_points - 1):
        idx = int(round(i * step))
        if idx not in indices:
            indices.append(idx)
    indices.append(len(rows) - 1)

    return [rows[i] for i in sorted(set(indices))]


# ============================================================================
# CROSS-CORRELATIONS
# ============================================================================

def compute_correlation(series_a: list[float], series_b: list[float]) -> Optional[float]:
    """Pearson correlation between two aligned series."""
    if len(series_a) != len(series_b) or len(series_a) < 5:
        return None

    n = len(series_a)
    mean_a = sum(series_a) / n
    mean_b = sum(series_b) / n

    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(series_a, series_b)) / n
    std_a = math.sqrt(sum((a - mean_a) ** 2 for a in series_a) / n)
    std_b = math.sqrt(sum((b - mean_b) ** 2 for b in series_b) / n)

    if std_a < 1e-10 or std_b < 1e-10:
        return None

    return round(cov / (std_a * std_b), 3)


def compute_cross_correlations(
    metric_series: dict[str, list[dict]],
) -> dict[str, float]:
    """
    Compute pairwise correlations between all requested metrics.
    Returns dict like {"core_pce vs oil_wti": 0.82, ...}
    """
    results = {}
    metric_names = list(metric_series.keys())

    for i in range(len(metric_names)):
        for j in range(i + 1, len(metric_names)):
            name_a = metric_names[i]
            name_b = metric_names[j]
            series_a = metric_series[name_a]
            series_b = metric_series[name_b]

            # Align by date
            dates_a = {d["date"]: d["value"] for d in series_a}
            dates_b = {d["date"]: d["value"] for d in series_b}
            common_dates = sorted(set(dates_a.keys()) & set(dates_b.keys()))

            if len(common_dates) < 10:
                continue

            vals_a = [dates_a[d] for d in common_dates]
            vals_b = [dates_b[d] for d in common_dates]

            corr = compute_correlation(vals_a, vals_b)
            if corr is not None:
                results[f"{name_a} vs {name_b}"] = corr

    return results


# ============================================================================
# INFLECTION DETECTION
# ============================================================================

def detect_inflections(
    series: list[dict],
    metric_name: str,
    thresholds: dict[str, float] = None,
) -> list[dict]:
    """
    Detect trend reversals and threshold crossings in a time series.
    Returns list of inflection events.
    """
    if len(series) < 2:
        return []

    inflections = []

    # Known thresholds for key metrics
    default_thresholds = {
        "vix": [20, 25, 30, 35],
        "core_pce": [2.0, 2.5, 3.0, 3.5, 4.0],
        "fed_funds": [3.0, 3.5, 4.0, 4.5, 5.0],
        "unemployment": [4.0, 4.5, 5.0],
        "hy_spread": [3.0, 4.0, 5.0, 6.0],
        "oil_wti": [80, 90, 100, 110, 120],
        "spread_2s10s": [-0.5, 0.0, 0.5],
        "dxy": [100, 105, 110, 115, 120],
    }

    check_thresholds = (thresholds or {}).get(metric_name) or default_thresholds.get(metric_name, [])

    # Threshold crossings
    for i in range(1, len(series)):
        prev_val = series[i - 1]["value"]
        curr_val = series[i]["value"]
        for threshold in check_thresholds:
            if prev_val < threshold <= curr_val:
                inflections.append({
                    "date": series[i]["date"],
                    "type": "crossed_above",
                    "threshold": threshold,
                    "value": round(curr_val, 4),
                })
            elif prev_val > threshold >= curr_val:
                inflections.append({
                    "date": series[i]["date"],
                    "type": "crossed_below",
                    "threshold": threshold,
                    "value": round(curr_val, 4),
                })

    # Trend reversals (sustained direction change over 5+ observations)
    if len(series) >= 10:
        window = max(3, len(series) // 8)
        for i in range(window, len(series) - window):
            before = [s["value"] for s in series[i - window:i]]
            after = [s["value"] for s in series[i:i + window]]
            trend_before = before[-1] - before[0]
            trend_after = after[-1] - after[0]
            # Sign change in trend direction
            if trend_before > 0 and trend_after < 0 and abs(trend_before) > abs(before[0]) * 0.01:
                inflections.append({
                    "date": series[i]["date"],
                    "type": "peak_reversal",
                    "value": round(series[i]["value"], 4),
                })
            elif trend_before < 0 and trend_after > 0 and abs(trend_before) > abs(before[0]) * 0.01:
                inflections.append({
                    "date": series[i]["date"],
                    "type": "trough_reversal",
                    "value": round(series[i]["value"], 4),
                })

    return inflections


# ============================================================================
# LAYER 1: Historical Trajectory (cached, free)
# ============================================================================

TRAJECTORY_METRICS = [
    ("core_pce", "Core PCE"),
    ("vix", "VIX"),
    ("fed_funds", "Fed Funds"),
    ("unemployment", "Unemployment"),
    ("oil_wti", "Oil WTI"),
    ("hy_spread", "HY Spread"),
    ("net_liquidity", "Net Liquidity"),
    ("dxy", "Dollar (DXY)"),
    ("consumer_sentiment", "Consumer Sentiment"),
    ("spread_2s10s", "2s10s Spread"),
    ("breakeven_5y", "5Y Breakeven"),
    ("initial_claims", "Initial Claims"),
]


async def compute_historical_trajectory(db) -> dict:
    """
    Compute Layer 1 trajectory summary for cached context.
    Called during morning refresh, stored in Redis.

    Returns dict with:
      - metric_snapshots: key metrics at 1mo/3mo/6mo/12mo lookback
      - regime_history: timeline of FINIAS assessments
      - inflection_points: threshold crossings in the last 12 months
    """
    today = date.today()
    lookbacks = {
        "1mo": today - timedelta(days=30),
        "3mo": today - timedelta(days=90),
        "6mo": today - timedelta(days=180),
        "12mo": today - timedelta(days=365),
    }

    trajectory = {
        "computed_at": today.isoformat(),
        "metric_snapshots": {},
        "regime_history": [],
        "inflection_points": [],
    }

    # === Metric snapshots at lookback points ===
    for metric_name, label in TRAJECTORY_METRICS:
        column = MATRIX_METRICS.get(metric_name)
        if not column:
            continue

        snapshots = {"label": label, "current": None}

        # Get current (most recent non-null)
        try:
            row = await db.fetchrow(
                f"SELECT obs_date, {column} FROM macro_data_matrix "
                f"WHERE {column} IS NOT NULL ORDER BY obs_date DESC LIMIT 1"
            )
            if row and row[column] is not None:
                snapshots["current"] = round(float(row[column]), 4)
                snapshots["current_date"] = str(row["obs_date"])
        except Exception as e:
            logger.warning(f"Could not fetch current {metric_name}: {e}")
            continue

        if snapshots["current"] is None:
            continue

        # Get lookback values
        for period, lookback_date in lookbacks.items():
            try:
                row = await db.fetchrow(
                    f"SELECT {column} FROM macro_data_matrix "
                    f"WHERE {column} IS NOT NULL AND obs_date <= $1 "
                    f"ORDER BY obs_date DESC LIMIT 1",
                    lookback_date
                )
                if row and row[column] is not None:
                    snapshots[period] = round(float(row[column]), 4)
            except Exception:
                pass

        trajectory["metric_snapshots"][metric_name] = snapshots

    # === Regime assessment history ===
    try:
        rows = await db.fetch(
            """
            SELECT id, assessed_at, primary_regime, binding_constraint,
                   composite_score, interpretation_json
            FROM regime_assessments
            ORDER BY id DESC LIMIT 30
            """
        )
        for row in reversed(rows):
            summary = ""
            try:
                interp = json.loads(row["interpretation_json"]) if isinstance(row["interpretation_json"], str) else (row["interpretation_json"] or {})
                summary = (interp.get("summary") or "")[:120]
            except Exception:
                pass

            trajectory["regime_history"].append({
                "date": str(row["assessed_at"].date()) if hasattr(row["assessed_at"], "date") else str(row["assessed_at"])[:10],
                "regime": row["primary_regime"],
                "binding": row["binding_constraint"],
                "composite": round(float(row["composite_score"]), 3) if row["composite_score"] else None,
                "summary": summary,
            })
    except Exception as e:
        logger.warning(f"Could not fetch regime history: {e}")

    # === Inflection points from last 12 months ===
    for metric_name, label in TRAJECTORY_METRICS:
        column = MATRIX_METRICS.get(metric_name)
        if not column:
            continue
        try:
            rows = await db.fetch(
                f"SELECT obs_date, {column} AS value FROM macro_data_matrix "
                f"WHERE {column} IS NOT NULL AND obs_date >= $1 "
                f"ORDER BY obs_date ASC",
                lookbacks["12mo"]
            )
            series = [{"date": str(r["obs_date"]), "value": float(r["value"])} for r in rows]
            inflections = detect_inflections(series, metric_name)
            for inf in inflections:
                inf["metric"] = label
                trajectory["inflection_points"].append(inf)
        except Exception:
            pass

    # Sort inflections by date
    trajectory["inflection_points"].sort(key=lambda x: x.get("date", ""))

    return trajectory


def format_trajectory_for_context(trajectory: dict) -> str:
    """Format Layer 1 trajectory as a string for Director cached context."""
    if not trajectory:
        return ""

    parts = []
    parts.append("HISTORICAL TRAJECTORY (auto-computed, last 12 months):")

    # Metric snapshots
    snapshots = trajectory.get("metric_snapshots", {})
    if snapshots:
        parts.append("  Key Metrics Over Time:")
        for metric_name, data in snapshots.items():
            label = data.get("label", metric_name)
            current = data.get("current")
            if current is None:
                continue

            # Build trend line
            points = []
            for period in ["12mo", "6mo", "3mo", "1mo"]:
                val = data.get(period)
                if val is not None:
                    points.append(f"{period}: {val}")
            points.append(f"now: {current}")

            # Compute change from 12mo ago
            val_12mo = data.get("12mo")
            change_str = ""
            if val_12mo is not None and val_12mo != 0:
                change = current - val_12mo
                change_pct = (change / abs(val_12mo)) * 100
                direction = "+" if change > 0 else ""
                change_str = f" [{direction}{change:.2f}, {direction}{change_pct:.1f}%]"

            parts.append(f"    {label}: {' → '.join(points)}{change_str}")

    # Regime history (last 10)
    regime_history = trajectory.get("regime_history", [])
    if regime_history:
        parts.append("  FINIAS Assessment History (recent):")
        for entry in regime_history[-10:]:
            d = entry.get("date", "?")
            r = entry.get("regime", "?")
            b = entry.get("binding", "?")
            s = entry.get("summary", "")
            if s:
                s = f' — "{s}"'
            parts.append(f"    {d}: {r} (binding: {b}){s}")

    # Inflection points (last 8)
    inflections = trajectory.get("inflection_points", [])
    if inflections:
        parts.append("  Notable Inflection Points:")
        for inf in inflections[-8:]:
            m = inf.get("metric", "?")
            d = inf.get("date", "?")
            t = inf.get("type", "?").replace("_", " ")
            v = inf.get("value", "?")
            th = inf.get("threshold")
            th_str = f" (threshold: {th})" if th else ""
            parts.append(f"    {d}: {m} {t} at {v}{th_str}")

    return "\n".join(parts)


# ============================================================================
# LAYER 2: Deep History Query (on-demand tool)
# ============================================================================

async def query_history(db, params: dict) -> str:
    """
    Execute a historical data query. Called by the Director when Claude
    invokes the query_macro_history tool.

    Args:
        db: DatabasePool instance
        params: Tool input from Claude containing:
            - metrics: list[str] — metric names from ALLOWED list
            - start_date: str — ISO date or relative ("12mo", "6mo", "3mo", "1mo")
            - end_date: str — ISO date or "now" (default)
            - include_assessments: bool — include regime assessment history
            - include_interpretations: bool — include FINIAS findings/risks
            - include_positioning: bool — include CFTC COT history
            - compute_correlations: bool — cross-metric correlations
            - detect_inflections: bool — threshold crossings
            - regime_filter: str|None — only data during this regime
            - sampling: str — "daily", "weekly", "monthly", "auto" (default)

    Returns:
        Formatted string for Claude to synthesize.
    """
    # Parse parameters
    metrics = params.get("metrics", [])
    start_str = params.get("start_date", "12mo")
    end_str = params.get("end_date", "now")
    include_assessments = params.get("include_assessments", False)
    include_interpretations = params.get("include_interpretations", False)
    include_positioning = params.get("include_positioning", False)
    do_correlations = params.get("compute_correlations", False)
    do_inflections = params.get("detect_inflections", False)
    regime_filter = params.get("regime_filter")
    sampling = params.get("sampling", "auto")

    # Validate metrics
    valid_metrics = []
    invalid_metrics = []
    for m in metrics:
        m_lower = m.lower().strip()
        if m_lower in MATRIX_METRICS or m_lower in MARKET_METRICS:
            valid_metrics.append(m_lower)
        else:
            invalid_metrics.append(m)

    if not valid_metrics and not include_assessments and not include_positioning:
        return f"ERROR: No valid metrics requested. Invalid: {invalid_metrics}. Available: {', '.join(ALL_METRIC_NAMES)}"

    # Parse dates
    today = date.today()
    end_date = today if end_str == "now" else _parse_date(end_str, today)

    relative_map = {"1mo": 30, "3mo": 90, "6mo": 180, "12mo": 365, "2y": 730, "5y": 1825, "max": 3650}
    if start_str in relative_map:
        start_date = today - timedelta(days=relative_map[start_str])
    else:
        start_date = _parse_date(start_str, today - timedelta(days=365))

    # Build response
    output_parts = []
    output_parts.append(f"HISTORICAL DATA: {start_date} to {end_date}")
    if invalid_metrics:
        output_parts.append(f"NOTE: Unrecognized metrics skipped: {', '.join(invalid_metrics)}")

    # === Query matrix metrics ===
    matrix_metrics = [m for m in valid_metrics if m in MATRIX_METRICS]
    market_metrics = [m for m in valid_metrics if m in MARKET_METRICS]
    all_series = {}  # For correlation computation

    if matrix_metrics:
        columns = [MATRIX_METRICS[m] for m in matrix_metrics]
        col_str = ", ".join(columns)

        try:
            rows = await db.fetch(
                f"SELECT obs_date, {col_str} FROM macro_data_matrix "
                f"WHERE obs_date BETWEEN $1 AND $2 ORDER BY obs_date ASC",
                start_date, end_date
            )

            for metric_name in matrix_metrics:
                col = MATRIX_METRICS[metric_name]
                series = [
                    {"date": str(r["obs_date"]), "value": round(float(r[col]), 4)}
                    for r in rows if r[col] is not None
                ]

                if not series:
                    output_parts.append(f"\n{metric_name.upper()}: No data for this period")
                    continue

                all_series[metric_name] = series

                # Apply sampling
                if sampling == "auto":
                    display_series = auto_sample(series)
                elif sampling == "monthly":
                    display_series = _sample_monthly(series)
                elif sampling == "weekly":
                    display_series = _sample_weekly(series)
                else:
                    display_series = auto_sample(series, max_points=30)

                # Compute summary stats
                values = [s["value"] for s in series]
                start_val = values[0]
                end_val = values[-1]
                high_val = max(values)
                low_val = min(values)
                change = end_val - start_val
                change_pct = (change / abs(start_val) * 100) if start_val != 0 else 0

                output_parts.append(f"\n{metric_name.upper()} ({series[0]['date']} to {series[-1]['date']}, {len(series)} observations):")
                output_parts.append(f"  Start: {start_val} → End: {end_val} | Change: {change:+.4f} ({change_pct:+.1f}%)")
                output_parts.append(f"  High: {high_val} | Low: {low_val} | Range: {high_val - low_val:.4f}")

                # Trend direction
                if len(values) >= 6:
                    first_third = sum(values[:len(values)//3]) / (len(values)//3)
                    last_third = sum(values[-len(values)//3:]) / (len(values)//3)
                    if last_third > first_third * 1.02:
                        output_parts.append(f"  Trend: RISING")
                    elif last_third < first_third * 0.98:
                        output_parts.append(f"  Trend: FALLING")
                    else:
                        output_parts.append(f"  Trend: FLAT")

                # Data points
                output_parts.append(f"  Series: " + " → ".join(
                    f"{s['date']}: {s['value']}" for s in display_series
                ))

                # Inflections
                if do_inflections:
                    inflections = detect_inflections(series, metric_name)
                    if inflections:
                        output_parts.append(f"  Inflections:")
                        for inf in inflections[-5:]:
                            th_str = f" (threshold: {inf.get('threshold')})" if inf.get("threshold") else ""
                            output_parts.append(f"    {inf['date']}: {inf['type'].replace('_', ' ')} at {inf['value']}{th_str}")

        except Exception as e:
            output_parts.append(f"\nERROR querying matrix metrics: {e}")

    # === Query market metrics (ETF prices) ===
    if market_metrics:
        for metric_name in market_metrics:
            symbol = MARKET_METRICS[metric_name]
            try:
                rows = await db.fetch(
                    "SELECT trade_date, close FROM market_data_daily "
                    "WHERE symbol = $1 AND trade_date BETWEEN $2 AND $3 "
                    "ORDER BY trade_date ASC",
                    symbol, start_date, end_date
                )

                series = [
                    {"date": str(r["trade_date"]), "value": round(float(r["close"]), 2)}
                    for r in rows
                ]

                if not series:
                    output_parts.append(f"\n{symbol}: No price data for this period")
                    continue

                all_series[metric_name] = series
                display_series = auto_sample(series)

                values = [s["value"] for s in series]
                start_val = values[0]
                end_val = values[-1]
                change_pct = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0

                output_parts.append(f"\n{symbol} PRICE ({series[0]['date']} to {series[-1]['date']}, {len(series)} trading days):")
                output_parts.append(f"  Start: ${start_val} → End: ${end_val} | Return: {change_pct:+.1f}%")
                output_parts.append(f"  High: ${max(values)} | Low: ${min(values)}")
                output_parts.append(f"  Series: " + " → ".join(
                    f"{s['date']}: ${s['value']}" for s in display_series
                ))

            except Exception as e:
                output_parts.append(f"\n{symbol}: Error — {e}")

    # === Cross-correlations ===
    if do_correlations and len(all_series) >= 2:
        correlations = compute_cross_correlations(all_series)
        if correlations:
            output_parts.append(f"\nCROSS-CORRELATIONS ({start_date} to {end_date}):")
            for pair, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
                output_parts.append(f"  {pair}: {corr:+.3f} ({strength})")

    # === Regime assessments ===
    if include_assessments:
        try:
            rows = await db.fetch(
                """
                SELECT id, assessed_at, primary_regime, binding_constraint,
                       composite_score, interpretation_json
                FROM regime_assessments
                WHERE assessed_at >= $1 AND assessed_at <= $2
                ORDER BY assessed_at ASC
                """,
                start_date, end_date + timedelta(days=1)
            )

            if rows:
                output_parts.append(f"\nFINIAS REGIME ASSESSMENTS ({len(rows)} in period):")
                for row in rows:
                    d = str(row["assessed_at"])[:10]
                    r = row["primary_regime"]
                    b = row["binding_constraint"]
                    cs = round(float(row["composite_score"]), 3) if row["composite_score"] else "?"

                    # Extract summary and key findings from interpretation
                    findings_str = ""
                    risks_str = ""
                    if include_interpretations:
                        try:
                            interp = json.loads(row["interpretation_json"]) if isinstance(row["interpretation_json"], str) else (row["interpretation_json"] or {})
                            summary = (interp.get("summary") or "")[:150]
                            findings = interp.get("key_findings", [])[:3]
                            risks = interp.get("risks", [])[:2]
                            if summary:
                                findings_str = f'\n      Summary: "{summary}"'
                            if findings:
                                findings_str += "\n      Findings: " + " | ".join(f[:80] for f in findings)
                            if risks:
                                risks_str = "\n      Risks: " + " | ".join(r[:80] for r in risks)
                        except Exception:
                            pass

                    output_parts.append(f"  [{d}] regime={r}, binding={b}, composite={cs}{findings_str}{risks_str}")

                # Regime filter
                if regime_filter:
                    filtered = [r for r in rows if r["primary_regime"] == regime_filter]
                    output_parts.append(f"\n  Filtered to '{regime_filter}': {len(filtered)} of {len(rows)} assessments")
            else:
                output_parts.append(f"\nNo FINIAS assessments found for {start_date} to {end_date}")

        except Exception as e:
            output_parts.append(f"\nERROR querying assessments: {e}")

    # === CFTC Positioning history ===
    if include_positioning:
        try:
            rows = await db.fetch(
                """
                SELECT report_date, contract_key, net_spec, open_interest
                FROM cot_positioning
                WHERE report_date BETWEEN $1 AND $2
                ORDER BY report_date ASC, contract_key
                """,
                start_date, end_date
            )

            if rows:
                # Group by contract
                contracts = {}
                for r in rows:
                    key = r["contract_key"]
                    if key not in contracts:
                        contracts[key] = []
                    contracts[key].append({
                        "date": str(r["report_date"]),
                        "net_spec": int(r["net_spec"]),
                        "open_interest": int(r["open_interest"]) if r["open_interest"] else None,
                    })

                output_parts.append(f"\nCFTC POSITIONING HISTORY:")
                for contract, data in contracts.items():
                    sampled = auto_sample(data, date_key="date", max_points=12)
                    start_pos = data[0]["net_spec"]
                    end_pos = data[-1]["net_spec"]
                    output_parts.append(f"  {contract} ({len(data)} weeks): {start_pos:+,} → {end_pos:+,}")
                    output_parts.append(f"    Series: " + " → ".join(
                        f"{s['date']}: {s['net_spec']:+,}" for s in sampled
                    ))
            else:
                output_parts.append(f"\nNo CFTC positioning data for {start_date} to {end_date}")

        except Exception as e:
            output_parts.append(f"\nERROR querying positioning: {e}")

    return "\n".join(output_parts)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _parse_date(date_str: str, fallback: date) -> date:
    """Parse an ISO date string, return fallback on failure."""
    try:
        parts = date_str.strip().split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return fallback


def _sample_monthly(series: list[dict]) -> list[dict]:
    """Keep only the last observation per month."""
    monthly = {}
    for s in series:
        month_key = s["date"][:7]  # "YYYY-MM"
        monthly[month_key] = s
    return list(monthly.values())


def _sample_weekly(series: list[dict]) -> list[dict]:
    """Keep approximately one observation per week."""
    if len(series) <= 52:
        return series
    step = max(1, len(series) // 52)
    result = [series[i] for i in range(0, len(series), step)]
    if series[-1] not in result:
        result.append(series[-1])
    return result


# ============================================================================
# TOOL DEFINITION — included in Director's tool list
# ============================================================================

def get_macro_history_tool_definition() -> dict:
    """
    Claude tool_use definition for the history query tool.
    The metrics list is embedded in the description so Claude knows what's available.
    """
    metric_list = ", ".join(ALL_METRIC_NAMES)

    return {
        "name": "query_macro_history",
        "description": (
            "Query historical macro-economic and market data from FINIAS databases. "
            "Use this when the user asks about trends, changes over time, historical comparisons, "
            "how metrics have evolved, or references a specific past date range. "
            "This tool queries stored time series data and FINIAS assessment history. "
            "It returns formatted data including values over time, summary statistics, "
            "cross-correlations between metrics, and threshold crossings. "
            f"Available metrics: {metric_list}. "
            "Market metrics (spy, qqq, xle, xlf, etc.) return daily closing prices. "
            "Macro metrics (core_pce, vix, fed_funds, etc.) return values from the macro data matrix. "
            "Use include_assessments=true to see how FINIAS classified each period. "
            "Use include_interpretations=true to see what FINIAS said (findings, risks). "
            "Use include_positioning=true for CFTC COT speculative positioning history. "
            "Use compute_correlations=true to see how metrics move relative to each other. "
            "Use detect_inflections=true to find threshold crossings and trend reversals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"List of metric names to query. Available: {metric_list}",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date as ISO (2025-01-20) or relative (1mo, 3mo, 6mo, 12mo, 2y, 5y, max). Default: 12mo.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date as ISO or 'now'. Default: now.",
                },
                "include_assessments": {
                    "type": "boolean",
                    "description": "Include FINIAS regime assessment history for the period.",
                },
                "include_interpretations": {
                    "type": "boolean",
                    "description": "Include FINIAS interpretation details (summary, findings, risks) per assessment.",
                },
                "include_positioning": {
                    "type": "boolean",
                    "description": "Include CFTC COT speculative positioning history.",
                },
                "compute_correlations": {
                    "type": "boolean",
                    "description": "Compute pairwise correlations between all requested metrics.",
                },
                "detect_inflections": {
                    "type": "boolean",
                    "description": "Detect threshold crossings and trend reversals.",
                },
                "regime_filter": {
                    "type": "string",
                    "description": "Filter assessments to only this regime (risk_off, transition, risk_on, crisis). Optional.",
                },
                "sampling": {
                    "type": "string",
                    "description": "Data sampling: auto (default, ~24 points), daily, weekly, monthly.",
                },
            },
            "required": ["metrics"],
        },
    }
