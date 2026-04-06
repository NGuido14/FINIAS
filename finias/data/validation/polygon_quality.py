"""
Polygon price data quality validation.

Validates daily price bars for gaps, zero-volume days,
suspicious price moves, and potential corporate actions.
"""

from __future__ import annotations
import logging
from datetime import date, timedelta
from typing import Optional

from finias.data.validation.quality import (
    SeriesQualityReport,
    check_series_gaps,
    check_staleness,
)

logger = logging.getLogger("finias.data.validation.polygon")


# Expected Polygon symbols
EXPECTED_SYMBOLS = [
    "SPY", "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU",
    "XLC", "XLY", "XLRE", "XLB", "RSP", "IWM", "TLT", "GLD",
    "HYG", "EEM", "CPER",
]


async def validate_price_bars(
    db,
    symbol: str,
    lookback_days: int = 365,
) -> SeriesQualityReport:
    """
    Validate daily price bars for a single symbol.

    Checks:
    1. Are there gaps in trading days? (>4 days between consecutive bars)
    2. Are there zero-volume days? (possible data error or halt)
    3. Are there suspicious single-day moves? (>15% without context)
    4. Is the data stale? (last bar older than expected)

    Args:
        db: DatabasePool instance
        symbol: Ticker symbol (e.g., "SPY")
        lookback_days: How far back to check

    Returns:
        SeriesQualityReport with findings.
    """
    from_date = date.today() - timedelta(days=lookback_days)

    rows = await db.fetch(
        """
        SELECT trade_date as date, close as value, volume, high, low
        FROM market_data_daily
        WHERE symbol = $1 AND trade_date >= $2
        ORDER BY trade_date ASC
        """,
        symbol, from_date,
    )

    observations = [{"date": r["date"], "value": float(r["value"])} for r in rows]

    report = SeriesQualityReport(
        series_id=symbol,
        expected_frequency="daily",
    )

    if not rows:
        report.status = "critical"
        report.warnings.append(f"{symbol}: no price data available")
        return report

    report.observation_count = len(rows)
    report.latest_date = rows[-1]["date"]
    report.staleness_days = (date.today() - rows[-1]["date"]).days

    # 1. Gap detection (using core function)
    gaps = check_series_gaps(observations, "daily", symbol)
    report.gaps = gaps
    report.is_consecutive = len(gaps) == 0
    for gap in gaps:
        report.warnings.append(
            f"{symbol}: {gap['gap_periods']}-day gap between "
            f"{gap['from_date']} and {gap['to_date']}"
        )

    # 2. Zero volume detection
    zero_vol_dates = []
    for r in rows[-60:]:  # Check last 60 trading days
        if r["volume"] is not None and r["volume"] == 0:
            zero_vol_dates.append(str(r["date"]))
    if zero_vol_dates:
        report.warnings.append(
            f"{symbol}: {len(zero_vol_dates)} zero-volume day(s) in last 60 bars: "
            f"{', '.join(zero_vol_dates[:3])}{'...' if len(zero_vol_dates) > 3 else ''}"
        )

    # 3. Suspicious single-day moves (>15%)
    for i in range(1, len(rows)):
        prev_close = float(rows[i-1]["value"])
        curr_close = float(rows[i]["value"])
        if prev_close > 0:
            pct_change = abs(curr_close / prev_close - 1) * 100
            if pct_change > 15:
                report.warnings.append(
                    f"{symbol}: {pct_change:.1f}% single-day move on "
                    f"{rows[i]['date']} ({prev_close:.2f} → {curr_close:.2f}). "
                    f"Verify: possible split, halt, or data error."
                )
                report.outliers.append({
                    "date": str(rows[i]["date"]),
                    "value": curr_close,
                    "prev_value": prev_close,
                    "pct_change": round(pct_change, 1),
                })

    # 4. Staleness
    stale = check_staleness(observations, "daily", symbol)
    if stale:
        report.warnings.append(stale["message"])

    # Status
    if report.staleness_days > 10:
        report.status = "warning"
    elif report.warnings:
        report.status = "warning"
    else:
        report.status = "healthy"

    return report


async def validate_all_polygon(
    db,
    symbols: list[str] = None,
    lookback_days: int = 365,
) -> dict[str, SeriesQualityReport]:
    """
    Validate all Polygon symbols.

    Returns dict of symbol → SeriesQualityReport.
    """
    if symbols is None:
        symbols = EXPECTED_SYMBOLS

    reports = {}
    for symbol in symbols:
        try:
            reports[symbol] = await validate_price_bars(db, symbol, lookback_days)
        except Exception as e:
            logger.error(f"Failed to validate {symbol}: {e}")
            report = SeriesQualityReport(series_id=symbol)
            report.status = "critical"
            report.warnings.append(f"{symbol}: validation failed ({e})")
            reports[symbol] = report

    return reports
