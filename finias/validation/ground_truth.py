"""
Ground-Truth Validation for FINIAS Computations

Compares FINIAS's internally computed values against FRED's published
reference series to confirm the financial math is correct.

Validations:
  1. Sahm Rule: computed value vs FRED SAHMREALTIME
  2. 2s10s Spread: computed value vs FRED T10Y2Y

If values match within tolerance, we have confidence the computations
are correct. If they diverge, there's a formula bug that unit tests
can't catch because they test logic, not real-data alignment.

Usage:
    python -m finias.validation.ground_truth
"""

import asyncio
import logging
import sys
from datetime import date, timedelta

from finias.core.config.settings import get_settings
from finias.core.database.connection import DatabasePool
from finias.core.database.migrations import run_migrations
from finias.data.providers.fred_client import FredClient

logger = logging.getLogger("finias.validation")


async def validate_sahm_rule(db: DatabasePool, fred: FredClient) -> dict:
    """
    Compare computed Sahm Rule values against FRED SAHMREALTIME.

    The Sahm Rule is the 3-month moving average of unemployment minus
    the 12-month low of 3-month averages. FRED publishes this as
    SAHMREALTIME. If our computation matches, the formula is correct.

    Returns dict with: matched, max_error, mean_error, divergent_dates
    """
    # Fetch FRED reference series
    reference = await fred.get_series(
        "SAHMREALTIME",
        observation_start=date.today() - timedelta(days=365 * 3),
    )
    if not reference:
        return {"status": "SKIP", "reason": "SAHMREALTIME not available from FRED"}

    # Fetch our stored Sahm values from regime_assessments
    rows = await db.fetch(
        """
        SELECT assessed_at::date as assess_date, sahm_value
        FROM regime_assessments
        WHERE sahm_value IS NOT NULL
        ORDER BY assessed_at ASC
        """
    )
    if not rows:
        return {"status": "SKIP", "reason": "No regime assessments with sahm_value"}

    # Build lookup from FRED reference (date -> value)
    ref_lookup = {}
    for obs in reference:
        ref_lookup[obs["date"]] = obs["value"]

    # Compare at each overlapping date
    # FRED publishes monthly, our assessments are ~weekly
    # Find the nearest FRED date for each assessment
    ref_dates = sorted(ref_lookup.keys())

    errors = []
    divergent = []
    compared = 0

    for row in rows:
        our_date = str(row["assess_date"])
        our_value = float(row["sahm_value"])

        # Find nearest FRED reference date (within 7 days)
        nearest_ref = None
        for rd in ref_dates:
            if abs((date.fromisoformat(rd) - date.fromisoformat(our_date)).days) <= 7:
                nearest_ref = rd

        if nearest_ref is None:
            continue

        ref_value = ref_lookup[nearest_ref]
        error = abs(our_value - ref_value)
        errors.append(error)
        compared += 1

        if error > 0.02:
            divergent.append({
                "our_date": our_date,
                "ref_date": nearest_ref,
                "our_value": round(our_value, 4),
                "ref_value": round(ref_value, 4),
                "error": round(error, 4),
            })

    if not errors:
        return {"status": "SKIP", "reason": "No overlapping dates between assessments and FRED"}

    max_error = max(errors)
    mean_error = sum(errors) / len(errors)

    return {
        "status": "PASS" if max_error <= 0.02 else "WARN",
        "compared": compared,
        "max_error": round(max_error, 4),
        "mean_error": round(mean_error, 4),
        "tolerance": 0.02,
        "divergent_dates": divergent,
    }


async def validate_2s10s_spread(db: DatabasePool, fred: FredClient) -> dict:
    """
    Compare computed 2s10s spread against FRED T10Y2Y.

    Our 2s10s spread is DGS10 - DGS2. FRED publishes T10Y2Y which is
    the same computation. If they match, our yield curve spread is correct.

    Returns dict with: matched, max_error, mean_error, divergent_dates
    """
    # Fetch FRED reference
    reference = await fred.get_series(
        "T10Y2Y",
        observation_start=date.today() - timedelta(days=365),
    )
    if not reference:
        return {"status": "SKIP", "reason": "T10Y2Y not available from FRED"}

    # Fetch our stored spread values
    rows = await db.fetch(
        """
        SELECT assessed_at::date as assess_date, spread_2s10s
        FROM regime_assessments
        WHERE spread_2s10s IS NOT NULL
        ORDER BY assessed_at ASC
        """
    )
    if not rows:
        return {"status": "SKIP", "reason": "No regime assessments with spread_2s10s"}

    # Build FRED lookup
    ref_lookup = {obs["date"]: obs["value"] for obs in reference}
    ref_dates = sorted(ref_lookup.keys())

    errors = []
    divergent = []
    compared = 0

    for row in rows:
        our_date = str(row["assess_date"])
        our_value = float(row["spread_2s10s"])

        nearest_ref = None
        for rd in ref_dates:
            if abs((date.fromisoformat(rd) - date.fromisoformat(our_date)).days) <= 3:
                nearest_ref = rd

        if nearest_ref is None:
            continue

        ref_value = ref_lookup[nearest_ref]
        error = abs(our_value - ref_value)
        errors.append(error)
        compared += 1

        if error > 0.05:
            divergent.append({
                "our_date": our_date,
                "ref_date": nearest_ref,
                "our_value": round(our_value, 4),
                "ref_value": round(ref_value, 4),
                "error": round(error, 4),
            })

    if not errors:
        return {"status": "SKIP", "reason": "No overlapping dates"}

    max_error = max(errors)
    mean_error = sum(errors) / len(errors)

    return {
        "status": "PASS" if max_error <= 0.05 else "WARN",
        "compared": compared,
        "max_error": round(max_error, 4),
        "mean_error": round(mean_error, 4),
        "tolerance": 0.05,
        "divergent_dates": divergent,
    }


async def run_all_validations(db: DatabasePool, fred: FredClient) -> dict:
    """Run all ground-truth validations and return results."""
    results = {}

    print("\n  Validating Sahm Rule vs SAHMREALTIME...")
    results["sahm_rule"] = await validate_sahm_rule(db, fred)
    sahm = results["sahm_rule"]
    if sahm["status"] == "PASS":
        print(f"  ✓ Sahm Rule: VALIDATED ({sahm['compared']} dates, "
              f"max error {sahm['max_error']}, tolerance {sahm['tolerance']})")
    elif sahm["status"] == "WARN":
        print(f"  ⚠ Sahm Rule: DIVERGENCE DETECTED ({sahm['compared']} dates, "
              f"max error {sahm['max_error']}, tolerance {sahm['tolerance']})")
        for d in sahm.get("divergent_dates", []):
            print(f"      {d['our_date']}: ours={d['our_value']}, FRED={d['ref_value']}, error={d['error']}")
    else:
        print(f"  - Sahm Rule: {sahm.get('reason', 'skipped')}")

    print("\n  Validating 2s10s spread vs T10Y2Y...")
    results["spread_2s10s"] = await validate_2s10s_spread(db, fred)
    spread = results["spread_2s10s"]
    if spread["status"] == "PASS":
        print(f"  ✓ 2s10s Spread: VALIDATED ({spread['compared']} dates, "
              f"max error {spread['max_error']}, tolerance {spread['tolerance']})")
    elif spread["status"] == "WARN":
        print(f"  ⚠ 2s10s Spread: DIVERGENCE DETECTED ({spread['compared']} dates, "
              f"max error {spread['max_error']}, tolerance {spread['tolerance']})")
        for d in spread.get("divergent_dates", []):
            print(f"      {d['our_date']}: ours={d['our_value']}, FRED={d['ref_value']}, error={d['error']}")
    else:
        print(f"  - 2s10s Spread: {spread.get('reason', 'skipped')}")

    return results


async def main():
    """Run ground-truth validation as a standalone script."""
    logging.basicConfig(level=logging.WARNING)

    print("\n" + "=" * 60)
    print("  FINIAS GROUND-TRUTH VALIDATION")
    print("=" * 60)

    db = DatabasePool()
    await db.initialize()
    await run_migrations(db)

    fred = FredClient()

    try:
        results = await run_all_validations(db, fred)

        print("\n" + "=" * 60)
        all_pass = all(r["status"] == "PASS" for r in results.values() if r["status"] != "SKIP")
        skipped = sum(1 for r in results.values() if r["status"] == "SKIP")
        if all_pass and skipped == 0:
            print("  ALL VALIDATIONS PASSED")
        elif all_pass:
            print(f"  VALIDATIONS PASSED ({skipped} skipped — need more data)")
        else:
            print("  SOME VALIDATIONS HAVE DIVERGENCES — investigate")
        print("=" * 60)
    finally:
        await fred.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
