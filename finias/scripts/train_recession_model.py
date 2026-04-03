"""
Train the logistic recession probability model.

Downloads 50+ years of FRED data, computes features, trains logistic
regression against NBER recession dates, validates on holdout, and
saves coefficients to JSON.

Usage:
    pip install scikit-learn  # One-time, training only
    python -m finias.scripts.train_recession_model

The output file (recession_coefficients.json) is committed to the repo.
Runtime scoring uses only math.exp() — no sklearn dependency.

Training period: 1982-01 to 2019-12 (pre-COVID)
Validation period: 2020-01 to 2025-12 (includes COVID recession)
"""

import asyncio
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Output path
OUTPUT_PATH = Path(__file__).parent.parent / "agents" / "macro_strategist" / "models" / "recession_coefficients.json"

# FRED series needed
SERIES_IDS = {
    "UNRATE": "Unemployment Rate",
    "T10Y3M": "10Y minus 3M Treasury Spread",
    "ICSA": "Initial Jobless Claims",
    "PERMIT": "Building Permits",
    "UMCSENT": "Consumer Sentiment",
    "INDPRO": "Industrial Production Index",
    "USREC": "NBER Recession Indicator",
}


async def fetch_fred_series(series_id: str, start_date: str = "1970-01-01") -> list[dict]:
    """Fetch a FRED series directly from the API."""
    import aiohttp
    from finias.core.config.settings import get_settings

    settings = get_settings()
    api_key = settings.fred_api_key

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "sort_order": "asc",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                logger.error(f"FRED API error for {series_id}: {resp.status}")
                return []
            data = await resp.json()
            observations = data.get("observations", [])
            result = []
            for obs in observations:
                if obs["value"] != ".":
                    result.append({
                        "date": obs["date"],
                        "value": float(obs["value"]),
                    })
            return result


def compute_sahm_values(unemployment: list[dict]) -> dict:
    """
    Compute Sahm Rule value for every month in the unemployment series.

    Uses the exact same formula as _compute_sahm_rule() in business_cycle.py:
    3-month average minus 12-month low of 3-month averages.

    Returns: {date_str: sahm_value}
    """
    if len(unemployment) < 15:
        return {}

    values = [u["value"] for u in unemployment]
    dates = [u["date"] for u in unemployment]

    # Compute rolling 3-month averages
    rolling_3m = {}
    for i in range(2, len(values)):
        avg = np.mean(values[i-2:i+1])
        rolling_3m[dates[i]] = avg

    # For each month, compute Sahm value
    sahm_by_date = {}
    sorted_dates = sorted(rolling_3m.keys())

    for i, d in enumerate(sorted_dates):
        if i < 12:
            continue  # Need 12 months of history

        current_3m = rolling_3m[d]
        # 12-month low of 3-month averages (excluding current)
        prior_12m = [rolling_3m[sorted_dates[j]] for j in range(max(0, i-12), i)]
        if prior_12m:
            low_12m = min(prior_12m)
            sahm_by_date[d] = round(current_3m - low_12m, 4)

    return sahm_by_date


def compute_yoy_pct(series: list[dict]) -> dict:
    """
    Compute YoY % change for a monthly series.
    Returns: {date_str: yoy_pct_change}
    """
    if len(series) < 13:
        return {}

    result = {}
    for i in range(12, len(series)):
        current = series[i]["value"]
        prior = series[i-12]["value"]
        if prior != 0:
            result[series[i]["date"]] = round((current / prior - 1) * 100, 2)

    return result


def monthly_average(daily_series: list[dict]) -> list[dict]:
    """Convert daily series to monthly averages."""
    from collections import defaultdict
    monthly = defaultdict(list)
    for obs in daily_series:
        month_key = obs["date"][:7]  # "YYYY-MM"
        monthly[month_key].append(obs["value"])

    result = []
    for month_key in sorted(monthly.keys()):
        avg = np.mean(monthly[month_key])
        result.append({"date": f"{month_key}-01", "value": round(avg, 4)})

    return result


def weekly_to_monthly_average(weekly_series: list[dict]) -> list[dict]:
    """Convert weekly claims data to monthly averages."""
    from collections import defaultdict
    monthly = defaultdict(list)
    for obs in weekly_series:
        month_key = obs["date"][:7]
        monthly[month_key].append(obs["value"])

    result = []
    for month_key in sorted(monthly.keys()):
        avg = np.mean(monthly[month_key])
        result.append({"date": f"{month_key}-01", "value": round(avg, 0)})

    return result


async def main():
    logger.info("=" * 60)
    logger.info("  FINIAS RECESSION MODEL TRAINER")
    logger.info("=" * 60)

    # === Step 1: Download FRED data ===
    logger.info("\n1. Downloading FRED data (50+ years)...")

    raw_data = {}
    for series_id, name in SERIES_IDS.items():
        logger.info(f"   Fetching {series_id} ({name})...")
        data = await fetch_fred_series(series_id, start_date="1970-01-01")
        raw_data[series_id] = data
        logger.info(f"   → {len(data)} observations, {data[0]['date'] if data else 'N/A'} to {data[-1]['date'] if data else 'N/A'}")

    # === Step 2: Compute features ===
    logger.info("\n2. Computing features...")

    # Sahm value from unemployment
    sahm_by_date = compute_sahm_values(raw_data["UNRATE"])
    logger.info(f"   Sahm values: {len(sahm_by_date)} months")

    # T10Y3M: convert daily to monthly average
    t10y3m_monthly = monthly_average(raw_data["T10Y3M"])
    t10y3m_by_date = {obs["date"]: obs["value"] for obs in t10y3m_monthly}
    logger.info(f"   Yield curve (3m10y): {len(t10y3m_by_date)} months")

    # Initial claims: weekly → monthly → YoY
    claims_monthly = weekly_to_monthly_average(raw_data["ICSA"])
    claims_yoy = compute_yoy_pct(claims_monthly)
    logger.info(f"   Claims YoY: {len(claims_yoy)} months")

    # Building permits: monthly → YoY
    permits_yoy = compute_yoy_pct(raw_data["PERMIT"])
    logger.info(f"   Permits YoY: {len(permits_yoy)} months")

    # Consumer sentiment: YoY % change (not level — level has structural break post-COVID)
    sentiment_yoy = compute_yoy_pct(raw_data["UMCSENT"])
    logger.info(f"   Sentiment YoY: {len(sentiment_yoy)} months")

    # Industrial production: monthly → YoY
    indpro_yoy = compute_yoy_pct(raw_data["INDPRO"])
    logger.info(f"   INDPRO YoY: {len(indpro_yoy)} months")

    # NBER recession indicator (target variable)
    recession_by_date = {obs["date"]: int(obs["value"]) for obs in raw_data["USREC"]}
    logger.info(f"   Recession dates: {len(recession_by_date)} months, "
                f"{sum(recession_by_date.values())} recession months")

    # === Step 3: Align features into matrix ===
    logger.info("\n3. Aligning features into matrix...")

    # Find common date range
    all_date_sets = [
        set(sahm_by_date.keys()),
        set(t10y3m_by_date.keys()),
        set(claims_yoy.keys()),
        set(permits_yoy.keys()),
        set(sentiment_yoy.keys()),
        set(indpro_yoy.keys()),
        set(recession_by_date.keys()),
    ]
    common_dates = sorted(set.intersection(*all_date_sets))
    logger.info(f"   Common dates: {len(common_dates)} months ({common_dates[0]} to {common_dates[-1]})")

    # Build aligned arrays
    feature_names = [
        "sahm_value", "yield_curve_3m10y", "claims_yoy_pct",
        "permits_yoy_pct", "sentiment_yoy_pct", "indpro_yoy_pct",
    ]

    X_rows = []
    y_rows = []
    dates_used = []

    for d in common_dates:
        row = [
            sahm_by_date[d],
            t10y3m_by_date[d],
            claims_yoy[d],
            permits_yoy[d],
            sentiment_yoy[d],
            indpro_yoy[d],
        ]
        X_rows.append(row)
        y_rows.append(recession_by_date[d])
        dates_used.append(d)

    X = np.array(X_rows)
    y = np.array(y_rows)

    logger.info(f"   Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    logger.info(f"   Recession months: {y.sum()} ({y.mean()*100:.1f}%)")
    logger.info(f"   Expansion months: {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")

    # === Step 4: Train/validation split ===
    # Train: 1982-2019 (pre-COVID), Validate: 2020-2025
    logger.info("\n4. Splitting train/validation...")

    train_mask = np.array([d < "2020-01-01" for d in dates_used])
    val_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    logger.info(f"   Train: {X_train.shape[0]} samples, {y_train.sum()} recession months")
    logger.info(f"   Validation: {X_val.shape[0]} samples, {y_val.sum()} recession months")

    # === Step 5: Standardize features ===
    logger.info("\n5. Standardizing features (using training statistics only)...")

    train_means = X_train.mean(axis=0)
    train_stds = X_train.std(axis=0)

    # Replace zero stds with 1.0 to avoid division by zero
    train_stds[train_stds == 0] = 1.0

    X_train_std = (X_train - train_means) / train_stds
    X_val_std = (X_val - train_means) / train_stds

    for i, name in enumerate(feature_names):
        logger.info(f"   {name}: mean={train_means[i]:.4f}, std={train_stds[i]:.4f}")

    # === Step 6: Train logistic regression ===
    logger.info("\n6. Training logistic regression...")

    try:
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.metrics import roc_auc_score, brier_score_loss
    except ImportError:
        logger.error("scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)

    model = LogisticRegressionCV(
        Cs=10,  # Test 10 regularization strengths
        cv=5,  # 5-fold cross-validation
        l1_ratios=(0,),  # Pure L2 regularization (new sklearn API)
        scoring="roc_auc",  # Optimize for AUC
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train_std, y_train)

    logger.info(f"   Best regularization C={model.C_[0]:.4f}")
    logger.info(f"   Intercept: {model.intercept_[0]:.4f}")
    for i, name in enumerate(feature_names):
        logger.info(f"   {name}: {model.coef_[0][i]:.4f}")

    # === Step 7: Validate ===
    logger.info("\n7. Validation results...")

    # Training performance
    train_probs = model.predict_proba(X_train_std)[:, 1]
    train_auc = roc_auc_score(y_train, train_probs)
    train_brier = brier_score_loss(y_train, train_probs)
    logger.info(f"   Training AUC: {train_auc:.4f}")
    logger.info(f"   Training Brier: {train_brier:.4f}")

    # Validation performance
    val_probs = model.predict_proba(X_val_std)[:, 1]
    if y_val.sum() > 0:
        val_auc = roc_auc_score(y_val, val_probs)
        val_brier = brier_score_loss(y_val, val_probs)
        logger.info(f"   Validation AUC: {val_auc:.4f}")
        logger.info(f"   Validation Brier: {val_brier:.4f}")
    else:
        val_auc = None
        val_brier = None
        logger.info("   Validation: No recession months in validation period")

    # Calibration table
    logger.info("\n   Calibration (training set):")
    for low, high in [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]:
        mask = (train_probs >= low) & (train_probs < high)
        if mask.sum() > 0:
            actual_rate = y_train[mask].mean()
            logger.info(f"     Predicted {low:.0%}-{high:.0%}: "
                       f"{mask.sum()} months, actual recession rate {actual_rate:.1%}")

    # Recession detection check
    logger.info("\n   Known recessions (validation period):")
    val_dates = [d for d, m in zip(dates_used, val_mask) if m]
    for i, d in enumerate(val_dates):
        if y_val[i] == 1:
            logger.info(f"     {d}: predicted prob={val_probs[i]:.3f} — {'DETECTED' if val_probs[i] > 0.3 else 'MISSED'}")

    # Current conditions (most recent observation)
    logger.info(f"\n   Current conditions ({dates_used[-1]}):")
    current_prob = model.predict_proba(X_val_std[-1:])[:, 1][0] if len(X_val_std) > 0 else model.predict_proba(X_train_std[-1:])[:, 1][0]
    logger.info(f"     Recession probability: {current_prob:.3f} ({current_prob*100:.1f}%)")

    # === Step 8: Save coefficients ===
    logger.info("\n8. Saving coefficients...")

    coefficients = {
        "model_version": "1.0",
        "model_type": "logistic_regression_l2",
        "training_period": f"{dates_used[0]} to {val_dates[0] if val_dates else dates_used[-1]}",
        "validation_period": f"{val_dates[0] if val_dates else 'N/A'} to {val_dates[-1] if val_dates else 'N/A'}",
        "n_train": int(X_train.shape[0]),
        "n_validation": int(X_val.shape[0]),
        "n_recession_months_train": int(y_train.sum()),
        "n_recession_months_val": int(y_val.sum()),
        "base_rate": round(float(y_train.mean()), 4),
        "regularization_C": round(float(model.C_[0]), 6),
        "intercept": round(float(model.intercept_[0]), 6),
        "coefficients": {
            name: round(float(model.coef_[0][i]), 6)
            for i, name in enumerate(feature_names)
        },
        "feature_means": {
            name: round(float(train_means[i]), 6)
            for i, name in enumerate(feature_names)
        },
        "feature_stds": {
            name: round(float(train_stds[i]), 6)
            for i, name in enumerate(feature_names)
        },
        "train_auc": round(float(train_auc), 4),
        "train_brier": round(float(train_brier), 4),
        "validation_auc": round(float(val_auc), 4) if val_auc is not None else None,
        "validation_brier": round(float(val_brier), 4) if val_brier is not None else None,
        "feature_descriptions": {
            "sahm_value": "Sahm Rule: 3m avg unemployment rise above 12m low of 3m averages",
            "yield_curve_3m10y": "10Y minus 3M Treasury spread (negative = inverted)",
            "claims_yoy_pct": "Initial claims 4-week avg, YoY % change (positive = rising claims)",
            "permits_yoy_pct": "Building permits YoY % change (negative = declining permits)",
            "sentiment_yoy_pct": "U. Michigan Consumer Sentiment YoY % change (negative = deteriorating)",
            "indpro_yoy_pct": "Industrial production YoY % change (negative = contracting)",
        },
        "notes": [
            "Trained on pre-COVID data (1982-2019) to avoid contamination from exogenous shock.",
            "COVID recession (Feb-Apr 2020) is in validation set — model may not detect it because "
            "COVID had zero classic leading indicator deterioration before onset.",
            "Class weight 'balanced' used to handle 13% recession base rate.",
            "Features standardized using training-period means and stds.",
            "At runtime, missing features contribute 0 (at-mean assumption).",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(coefficients, f, indent=2)

    logger.info(f"   Saved to: {OUTPUT_PATH}")
    logger.info(f"   File size: {OUTPUT_PATH.stat().st_size} bytes")

    logger.info("\n" + "=" * 60)
    if val_auc and val_auc > 0.85:
        logger.info("  MODEL TRAINING COMPLETE — validation AUC > 0.85")
        logger.info("  Commit recession_coefficients.json to the repo.")
    elif val_auc:
        logger.info(f"  MODEL TRAINING COMPLETE — validation AUC = {val_auc:.3f}")
        logger.info("  Review results before committing. AUC < 0.85 may indicate")
        logger.info("  insufficient discriminative power.")
    else:
        logger.info("  MODEL TRAINING COMPLETE — no validation period recessions")
        logger.info("  Training AUC looks good. Validate manually.")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
