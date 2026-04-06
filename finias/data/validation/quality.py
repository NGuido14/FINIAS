"""
Core data quality validation functions — data-source agnostic.

Any agent can import these to validate time series data before computation.
Functions are pure Python, no database or API calls, zero cost.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional
import logging

logger = logging.getLogger("finias.data.validation")


# ============================================================================
# Quality Report Dataclasses
# ============================================================================

@dataclass
class SeriesQualityReport:
    """Quality assessment for a single time series."""
    series_id: str
    status: str = "unknown"             # healthy, warning, critical
    observation_count: int = 0
    latest_date: Optional[date] = None
    staleness_days: int = 999
    expected_frequency: str = "unknown"  # daily, weekly, monthly, quarterly
    gaps: list = field(default_factory=list)         # [{from_date, to_date, gap_periods}]
    outliers: list = field(default_factory=list)     # [{date, value, z_score}]
    warnings: list = field(default_factory=list)     # Human-readable warning strings
    is_consecutive: bool = True          # No gaps in expected frequency

    def to_dict(self) -> dict:
        return {
            "series_id": self.series_id,
            "status": self.status,
            "observation_count": self.observation_count,
            "latest_date": str(self.latest_date) if self.latest_date else None,
            "staleness_days": self.staleness_days,
            "expected_frequency": self.expected_frequency,
            "gaps": self.gaps,
            "outlier_count": len(self.outliers),
            "warnings": self.warnings,
            "is_consecutive": self.is_consecutive,
        }


@dataclass
class DataQualityReport:
    """Aggregate quality assessment across all data sources."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    series_reports: dict = field(default_factory=dict)    # {series_id: SeriesQualityReport}
    critical_issues: list = field(default_factory=list)   # Issues that degrade computation accuracy
    warnings: list = field(default_factory=list)          # Notable but non-critical issues
    overall_status: str = "unknown"                       # healthy, degraded, critical

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status,
            "critical_count": len(self.critical_issues),
            "warning_count": len(self.warnings),
            "critical_issues": self.critical_issues,
            "warnings": self.warnings[:10],  # Cap at 10 for storage
            "series_count": len(self.series_reports),
            "healthy_count": sum(
                1 for r in self.series_reports.values() if r.status == "healthy"
            ),
            "degraded_count": sum(
                1 for r in self.series_reports.values() if r.status in ("warning", "critical")
            ),
        }

    def get_quality_warnings_for_notes(self) -> list[str]:
        """
        Generate warnings suitable for Claude's data notes.
        Only includes actionable issues that affect interpretation quality.
        """
        notes = []
        if self.critical_issues:
            notes.append(
                f"DATA QUALITY CRITICAL: {len(self.critical_issues)} issue(s) detected. "
                f"Reduce confidence in affected computations: "
                + "; ".join(self.critical_issues[:3])
            )
        if self.warnings:
            notable = [w for w in self.warnings if "gap" in w.lower() or "consecutive" in w.lower()]
            if notable:
                notes.append(
                    f"DATA QUALITY WARNING: {'; '.join(notable[:3])}"
                )
        return notes

    def get_freshness_warnings(self) -> list[str]:
        """
        Generate warnings for the MacroContext data_freshness_warnings field.
        These are consumed by downstream agents for confidence calibration.
        """
        warnings = []
        for issue in self.critical_issues:
            warnings.append(f"CRITICAL: {issue}")
        for report in self.series_reports.values():
            if not report.is_consecutive and report.series_id in CONSECUTIVE_CRITICAL:
                warnings.append(
                    f"{report.series_id} has gaps — {CONSECUTIVE_CRITICAL[report.series_id]} "
                    f"computation may be inaccurate"
                )
        return warnings


# ============================================================================
# Series where gaps corrupt downstream computations
# ============================================================================

CONSECUTIVE_CRITICAL = {
    "UNRATE": "Sahm Rule",
    "ICSA": "Initial claims YoY / recession model",
    "UMCSENT": "Consumer sentiment YoY / recession model",
    "INDPRO": "Industrial production YoY / recession model",
    "PERMIT": "Building permits YoY / recession model",
    "PCEPILFE": "Core PCE YoY / inflation module",
    "CPIAUCSL": "CPI YoY / inflation module",
}


# ============================================================================
# Core Validation Functions
# ============================================================================

def check_series_gaps(
    observations: list[dict],
    expected_frequency: str,
    series_id: str = "unknown",
) -> list[dict]:
    """
    Detect gaps in a time series based on expected publication frequency.

    For monthly data: checks that consecutive observations are ~1 month apart.
    For weekly data: checks that consecutive observations are ~6-8 days apart.
    For daily data: checks that gaps don't exceed 4 days (weekends + holidays).

    Args:
        observations: List of dicts with 'date' key (str or date object).
                      Must be sorted chronologically (oldest first).
        expected_frequency: "daily", "weekly", "monthly", "quarterly"
        series_id: For logging/reporting purposes.

    Returns:
        List of gap dicts: [{from_date, to_date, gap_periods, expected_periods}]
        Empty list = no gaps detected.
    """
    if len(observations) < 2:
        return []

    # Parse dates
    dates = []
    for obs in observations:
        d = obs.get("date") or obs.get("obs_date")
        if d is None:
            continue
        if isinstance(d, str):
            try:
                d = date.fromisoformat(d)
            except (ValueError, TypeError):
                continue
        if isinstance(d, datetime):
            d = d.date()
        if isinstance(d, date):
            dates.append(d)

    if len(dates) < 2:
        return []

    dates.sort()

    # Frequency-specific gap thresholds
    # max_gap_days: anything beyond this between consecutive observations is a gap
    thresholds = {
        "daily": {"max_gap_days": 5, "label": "trading days"},      # Mon-Fri + holidays
        "weekly": {"max_gap_days": 10, "label": "weeks"},            # 7 days + buffer
        "monthly": {"max_gap_days": 45, "label": "months"},          # ~30 days + buffer
        "quarterly": {"max_gap_days": 120, "label": "quarters"},     # ~90 days + buffer
    }

    threshold = thresholds.get(expected_frequency, thresholds["monthly"])
    max_gap = threshold["max_gap_days"]

    gaps = []
    for i in range(1, len(dates)):
        delta_days = (dates[i] - dates[i-1]).days

        if delta_days > max_gap:
            # Calculate how many periods are missing
            if expected_frequency == "monthly":
                gap_periods = (dates[i].year - dates[i-1].year) * 12 + \
                              (dates[i].month - dates[i-1].month) - 1
            elif expected_frequency == "weekly":
                gap_periods = (delta_days // 7) - 1
            elif expected_frequency == "daily":
                gap_periods = delta_days - 1  # Approximate (includes weekends)
            else:
                gap_periods = 1

            if gap_periods > 0:
                gaps.append({
                    "from_date": str(dates[i-1]),
                    "to_date": str(dates[i]),
                    "delta_days": delta_days,
                    "gap_periods": gap_periods,
                    "series_id": series_id,
                })

    if gaps:
        total_missing = sum(g["gap_periods"] for g in gaps)
        logger.warning(
            f"Data quality: {series_id} has {len(gaps)} gap(s) "
            f"({total_missing} missing {threshold['label']})"
        )

    return gaps


def check_staleness(
    observations: list[dict],
    expected_frequency: str,
    series_id: str = "unknown",
    as_of: Optional[date] = None,
) -> Optional[dict]:
    """
    Check if a series is staler than expected for its publication frequency.

    Returns a warning dict if stale, None if fresh enough.
    """
    if not observations:
        return {
            "series_id": series_id,
            "status": "missing",
            "message": f"{series_id}: no observations available",
        }

    if as_of is None:
        as_of = date.today()

    # Parse latest date
    latest_obs = observations[-1]
    latest_date = latest_obs.get("date") or latest_obs.get("obs_date")
    if isinstance(latest_date, str):
        try:
            latest_date = date.fromisoformat(latest_date)
        except (ValueError, TypeError):
            return None
    if isinstance(latest_date, datetime):
        latest_date = latest_date.date()

    days_old = (as_of - latest_date).days

    # Expected staleness by frequency
    max_acceptable = {
        "daily": 5,        # Weekends + holidays
        "weekly": 14,      # 2 weeks max
        "monthly": 60,     # 2 months max (publication lag)
        "quarterly": 120,  # 4 months max
    }

    max_days = max_acceptable.get(expected_frequency, 60)

    if days_old > max_days:
        return {
            "series_id": series_id,
            "status": "stale",
            "days_old": days_old,
            "max_acceptable": max_days,
            "message": f"{series_id}: {days_old} days old (max acceptable: {max_days} for {expected_frequency})",
        }

    return None


def check_value_bounds(
    value: float,
    min_val: float,
    max_val: float,
    label: str = "unknown",
) -> Optional[str]:
    """
    Check if a computed value is within reasonable bounds.

    Returns a warning string if out of bounds, None if okay.
    """
    if value < min_val:
        return f"{label}: {value} is below minimum bound {min_val}"
    if value > max_val:
        return f"{label}: {value} is above maximum bound {max_val}"
    return None


def check_outliers(
    observations: list[dict],
    z_threshold: float = 4.0,
    series_id: str = "unknown",
    lookback: int = 52,
) -> list[dict]:
    """
    Detect statistical outliers in a series using z-score.

    Only checks the most recent observation against the trailing window.
    A z-score > 4.0 means the value is 4 standard deviations from the
    trailing mean — extremely unusual and worth flagging.

    Returns list of outlier dicts (usually 0 or 1 elements).
    """
    if len(observations) < lookback + 1:
        return []

    values = [obs.get("value") or obs.get("net_spec") for obs in observations]
    values = [v for v in values if v is not None]

    if len(values) < lookback + 1:
        return []

    window = values[-(lookback + 1):-1]  # Trailing window excluding current
    current = values[-1]

    import statistics
    mean = statistics.mean(window)
    stdev = statistics.stdev(window) if len(window) > 1 else 1.0

    if stdev == 0:
        return []

    z_score = abs(current - mean) / stdev

    if z_score > z_threshold:
        latest_date = observations[-1].get("date") or observations[-1].get("obs_date")
        return [{
            "series_id": series_id,
            "date": str(latest_date),
            "value": current,
            "mean": round(mean, 4),
            "stdev": round(stdev, 4),
            "z_score": round(z_score, 2),
            "message": f"{series_id}: latest value {current} is {z_score:.1f} std devs from trailing mean",
        }]

    return []


def validate_series(
    observations: list[dict],
    series_id: str,
    expected_frequency: str,
    consecutive_required: bool = False,
    as_of: Optional[date] = None,
) -> SeriesQualityReport:
    """
    Run all quality checks on a single series.

    Args:
        observations: Sorted chronologically (oldest first).
        series_id: Identifier for logging.
        expected_frequency: "daily", "weekly", "monthly", "quarterly"
        consecutive_required: If True, gaps produce critical status.
        as_of: Date to check staleness against (default: today).

    Returns:
        SeriesQualityReport with all findings.
    """
    report = SeriesQualityReport(
        series_id=series_id,
        expected_frequency=expected_frequency,
    )

    if not observations:
        report.status = "critical"
        report.warnings.append(f"{series_id}: no data available")
        return report

    report.observation_count = len(observations)

    # Parse latest date
    latest = observations[-1]
    latest_date = latest.get("date") or latest.get("obs_date")
    if isinstance(latest_date, str):
        try:
            latest_date = date.fromisoformat(latest_date)
        except (ValueError, TypeError):
            latest_date = None
    if isinstance(latest_date, datetime):
        latest_date = latest_date.date()
    report.latest_date = latest_date

    if latest_date and as_of:
        report.staleness_days = (as_of - latest_date).days
    elif latest_date:
        report.staleness_days = (date.today() - latest_date).days

    # Check gaps
    gaps = check_series_gaps(observations, expected_frequency, series_id)
    report.gaps = gaps
    report.is_consecutive = len(gaps) == 0

    if gaps:
        total_missing = sum(g["gap_periods"] for g in gaps)
        gap_warning = (
            f"{series_id}: {len(gaps)} gap(s) detected "
            f"({total_missing} missing {expected_frequency} observations)"
        )
        report.warnings.append(gap_warning)

        if consecutive_required:
            report.status = "critical"
            report.warnings.append(
                f"{series_id} requires consecutive data for accurate computation. "
                f"Downstream values (e.g., {CONSECUTIVE_CRITICAL.get(series_id, 'related computations')}) "
                f"may be inaccurate."
            )

    # Check staleness
    stale = check_staleness(observations, expected_frequency, series_id, as_of)
    if stale:
        report.warnings.append(stale["message"])

    # Check outliers (only for series with enough data)
    if len(observations) > 52:
        outliers = check_outliers(observations, series_id=series_id)
        report.outliers = outliers
        for o in outliers:
            report.warnings.append(o["message"])

    # Determine overall status
    if report.status == "critical":
        pass  # Already set by gap + consecutive_required
    elif not report.is_consecutive and consecutive_required:
        report.status = "critical"
    elif report.warnings:
        report.status = "warning"
    else:
        report.status = "healthy"

    return report
