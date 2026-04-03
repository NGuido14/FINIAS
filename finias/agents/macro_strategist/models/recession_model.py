"""
Calibrated logistic recession probability model.

Trained on NBER recession dates (1982-2019) using 6 features from FRED.
Validated out-of-sample on 2020-2025 (includes COVID recession).

At runtime, this module:
  1. Loads pre-trained coefficients from JSON (cached on first call)
  2. Standardizes current indicator values using training-period statistics
  3. Applies the logistic function: P = 1 / (1 + exp(-(b0 + b1*x1 + ... + b6*x6)))
  4. Returns a genuinely calibrated probability

If a feature is missing (None), it contributes zero to the score —
equivalent to being at the historical mean. This is mathematically
correct for standardized features and degrades gracefully.

If the coefficients file doesn't exist (training hasn't been run),
returns None so the caller can fall back to the heuristic.

Features (6):
  1. sahm_value — Sahm Rule (unemployment 3m avg rise above 12m low)
  2. yield_curve_3m10y — 10Y minus 3M Treasury spread
  3. claims_yoy_pct — Initial claims YoY % change
  4. permits_yoy_pct — Building permits YoY % change
  5. sentiment_yoy_pct — University of Michigan sentiment YoY % change
  6. indpro_yoy_pct — Industrial production YoY % change

No sklearn dependency. No numpy dependency. Just math.exp().
"""

import json
import math
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("finias.models.recession")

# Cached model coefficients (loaded once)
_MODEL_CACHE: Optional[dict] = None

# Path to coefficients file (relative to this module)
_COEFFICIENTS_PATH = Path(__file__).parent / "recession_coefficients.json"


def _load_model() -> Optional[dict]:
    """Load model coefficients from JSON. Returns None if file doesn't exist."""
    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if not _COEFFICIENTS_PATH.exists():
        logger.warning(
            f"Recession model coefficients not found at {_COEFFICIENTS_PATH}. "
            "Run 'python -m finias.scripts.train_recession_model' to train. "
            "Falling back to heuristic."
        )
        return None

    try:
        with open(_COEFFICIENTS_PATH) as f:
            _MODEL_CACHE = json.load(f)
        logger.info(
            f"Recession model loaded: trained on {_MODEL_CACHE.get('training_period', 'unknown')}, "
            f"validation AUC={_MODEL_CACHE.get('validation_auc', 'N/A')}"
        )
        return _MODEL_CACHE
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load recession model: {e}")
        return None


def predict_recession_probability(
    sahm_value: Optional[float] = None,
    yield_curve_3m10y: Optional[float] = None,
    claims_yoy_pct: Optional[float] = None,
    permits_yoy_pct: Optional[float] = None,
    sentiment_yoy_pct: Optional[float] = None,
    indpro_yoy_pct: Optional[float] = None,
) -> Optional[float]:
    """
    Compute calibrated recession probability from current indicators.

    Returns a float between 0.0 and 1.0 representing the probability
    that the US economy is currently in a recession, based on the
    historical relationship between these indicators and NBER recession dates.

    Returns None if the model coefficients haven't been trained yet.

    Args:
        sahm_value: Sahm Rule value (3m avg unemployment rise above 12m low)
        yield_curve_3m10y: 10Y minus 3M Treasury spread (negative = inverted)
        claims_yoy_pct: Initial claims 4-week average, YoY % change
        permits_yoy_pct: Building permits YoY % change
        sentiment_yoy_pct: University of Michigan Consumer Sentiment YoY % change
        indpro_yoy_pct: Industrial production YoY % change

    Returns:
        Calibrated probability (0.0 to 1.0), or None if model not available
    """
    model = _load_model()
    if model is None:
        return None

    coefficients = model["coefficients"]
    means = model["feature_means"]
    stds = model["feature_stds"]
    intercept = model["intercept"]

    # Feature name → value mapping
    features = {
        "sahm_value": sahm_value,
        "yield_curve_3m10y": yield_curve_3m10y,
        "claims_yoy_pct": claims_yoy_pct,
        "permits_yoy_pct": permits_yoy_pct,
        "sentiment_yoy_pct": sentiment_yoy_pct,
        "indpro_yoy_pct": indpro_yoy_pct,
    }

    # Compute logistic score
    z = intercept
    features_used = 0

    for name, value in features.items():
        if value is not None and name in coefficients:
            mean = means.get(name, 0.0)
            std = stds.get(name, 1.0)
            if std > 0:
                z += coefficients[name] * (value - mean) / std
                features_used += 1
            # If std is 0, feature is constant → skip (contributes 0)

    if features_used == 0:
        # No features available — return base rate from training data
        return model.get("base_rate", 0.13)

    # Logistic function
    # Clip z to prevent overflow in exp()
    z = max(-20.0, min(20.0, z))
    probability = 1.0 / (1.0 + math.exp(-z))

    return round(probability, 4)
