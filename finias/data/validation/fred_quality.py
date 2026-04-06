"""
FRED-specific data quality validation.

Validates all 68 MACRO_SERIES for gaps, staleness, and population.
Uses the frequency mapping from data_loader.py (already defined in the codebase).

Key feature: identifies which series require consecutive observations
for accurate computation (e.g., UNRATE for Sahm Rule).
"""

from __future__ import annotations
import logging
from datetime import date
from typing import Optional

from finias.data.validation.quality import (
    DataQualityReport,
    SeriesQualityReport,
    validate_series,
    CONSECUTIVE_CRITICAL,
)

logger = logging.getLogger("finias.data.validation.fred")


# ============================================================================
# FRED Series Frequency Mapping
# Reuses the categorization from finias/backtesting/data_loader.py
# ============================================================================

FRED_FREQUENCY = {
    # Daily (0-day lag)
    "DGS2": "daily", "DGS5": "daily", "DGS10": "daily",
    "DGS30": "daily", "DTB3": "daily", "VIXCLS": "daily",
    "VXVCLS": "daily", "T10Y2Y": "daily", "T10Y3M": "daily",
    "BAMLH0A0HYM2": "daily", "DFEDTARU": "daily",
    "DFEDTARL": "daily", "DTWEXBGS": "daily",
    "T5YIE": "daily", "T10YIE": "daily",
    "DFII5": "daily", "DFII10": "daily",
    "THREEFYTP10": "daily", "DCOILWTICO": "daily",
    "DCOILBRENTEU": "daily", "T5YIFR": "daily",
    "RRPONTSYD": "daily",

    # Weekly
    "WALCL": "weekly", "TREAST": "weekly", "WSHOMCB": "weekly",
    "WTREGEN": "weekly", "WRESBAL": "weekly", "NFCI": "weekly",
    "ANFCI": "weekly", "STLFSI4": "weekly", "ICSA": "weekly",
    "CCSA": "weekly", "TOTBKCR": "weekly",

    # Monthly
    "FEDFUNDS": "monthly", "CPIAUCSL": "monthly", "CPILFESL": "monthly",
    "CUSR0000SEHC": "monthly", "CUSR0000SAS": "monthly",
    "PCEPI": "monthly", "PCEPILFE": "monthly",
    "STICKCPIM159SFRBATL": "monthly", "FLEXCPIM159SFRBATL": "monthly",
    "PCETRIM12M159SFRBDAL": "monthly", "PPIACO": "monthly",
    "CES0500000003": "monthly", "UNRATE": "monthly", "PAYEMS": "monthly",
    "U6RATE": "monthly", "CIVPART": "monthly", "LNS11300060": "monthly",
    "TEMPHELPS": "monthly", "AWHAETP": "monthly",
    "UMCSENT": "monthly", "INDPRO": "monthly",
    "TOTALSL": "monthly", "M2SL": "monthly",
    "TCU": "monthly", "PERMIT": "monthly", "HOUST": "monthly",
    "RSAFS": "monthly", "PI": "monthly", "PCEC96": "monthly",
    "DGORDER": "monthly", "CFNAI": "monthly",
    "GACDFSA066MSFRBPHI": "monthly",
    "JTSJOL": "monthly", "JTSQUR": "monthly",

    # Quarterly
    "GDPNOW": "quarterly",

    # Special cases
    "SAHMREALTIME": "monthly",
}


async def validate_fred_series(
    db,
    series_id: str,
    lookback_days: int = 730,
) -> SeriesQualityReport:
    """
    Validate a single FRED series from the database.

    Fetches stored observations and runs all quality checks.

    Args:
        db: DatabasePool instance
        series_id: FRED series identifier
        lookback_days: How far back to check (default 2 years)

    Returns:
        SeriesQualityReport with gaps, staleness, outliers.
    """
    from_date = date.today() - __import__("datetime").timedelta(days=lookback_days)

    rows = await db.fetch(
        """
        SELECT obs_date as date, value
        FROM economic_indicators
        WHERE series_id = $1 AND obs_date >= $2
        ORDER BY obs_date ASC
        """,
        series_id, from_date,
    )

    observations = [{"date": r["date"], "value": float(r["value"])} for r in rows]
    frequency = FRED_FREQUENCY.get(series_id, "monthly")
    is_consecutive_required = series_id in CONSECUTIVE_CRITICAL

    return validate_series(
        observations=observations,
        series_id=series_id,
        expected_frequency=frequency,
        consecutive_required=is_consecutive_required,
    )


async def validate_all_fred(
    db,
    series_ids: list[str] = None,
    lookback_days: int = 730,
) -> DataQualityReport:
    """
    Validate all FRED series in the database.

    Checks every series in FRED_FREQUENCY (or a subset if provided)
    for gaps, staleness, and outliers.

    Args:
        db: DatabasePool instance
        series_ids: Optional subset to validate (default: all in FRED_FREQUENCY)
        lookback_days: How far back to check

    Returns:
        DataQualityReport with per-series reports and aggregate status.
    """
    if series_ids is None:
        series_ids = list(FRED_FREQUENCY.keys())

    report = DataQualityReport()

    for series_id in series_ids:
        try:
            series_report = await validate_fred_series(db, series_id, lookback_days)
            report.series_reports[series_id] = series_report

            # Escalate critical issues
            if series_report.status == "critical":
                for warning in series_report.warnings:
                    if "consecutive" in warning.lower() or "no data" in warning.lower():
                        report.critical_issues.append(warning)

            # Collect warnings
            for warning in series_report.warnings:
                if warning not in report.critical_issues:
                    report.warnings.append(warning)

        except Exception as e:
            logger.error(f"Failed to validate {series_id}: {e}")
            report.warnings.append(f"{series_id}: validation failed ({e})")

    # Determine overall status
    critical_count = sum(
        1 for r in report.series_reports.values() if r.status == "critical"
    )
    warning_count = sum(
        1 for r in report.series_reports.values() if r.status == "warning"
    )

    if critical_count > 0:
        report.overall_status = "critical"
    elif warning_count > 3:
        report.overall_status = "degraded"
    else:
        report.overall_status = "healthy"

    return report


async def detect_fred_gaps(
    db,
    critical_only: bool = True,
    lookback_days: int = 730,
) -> list[dict]:
    """
    Quick scan for gaps in FRED data — focused on series that matter.

    If critical_only=True, only checks the 7 CONSECUTIVE_CRITICAL series.
    Returns a flat list of gap dicts for easy consumption.

    This is the function that would have caught the UNRATE October 2025 gap.
    """
    series_to_check = (
        list(CONSECUTIVE_CRITICAL.keys()) if critical_only
        else list(FRED_FREQUENCY.keys())
    )

    all_gaps = []
    for series_id in series_to_check:
        report = await validate_fred_series(db, series_id, lookback_days)
        for gap in report.gaps:
            all_gaps.append({
                "series_id": series_id,
                "from_date": gap["from_date"],
                "to_date": gap["to_date"],
                "gap_periods": gap["gap_periods"],
                "computation_affected": CONSECUTIVE_CRITICAL.get(series_id, "unknown"),
                "severity": "critical" if series_id in CONSECUTIVE_CRITICAL else "warning",
            })

    return all_gaps
