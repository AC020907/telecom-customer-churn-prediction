"""
Inference utilities for the Interconnect churn model.

Given raw attributes for a single customer (the kind of information an
agent or a self-service form would collect), this module:

1. builds a one-row dataframe in the exact raw shape the training
   pipeline expects right before feature engineering;
2. derives the same engineered features used at training time
   (``src.feature_engineering``);
3. loads the trained pipeline (preprocessing + LightGBM) and returns
   the predicted class and the estimated churn probability.

The trained pipeline itself performs one-hot encoding, so this module
never needs to duplicate that logic - it only has to reproduce the raw
-> engineered-feature transformation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src import data_processing, feature_engineering

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

DEFAULT_THRESHOLD = 0.5

# Raw fields a caller must provide for a single customer. These mirror
# the columns available right after merging contract/personal/internet
# /phone and before feature engineering.
REQUIRED_FIELDS = [
    "BeginDate",
    "Type",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
]


class ChurnPredictionError(ValueError):
    """Raised when the supplied customer data is invalid or incomplete."""


def _model_cache() -> dict[str, Any]:
    if not hasattr(_model_cache, "_cache"):
        _model_cache._cache = {}
    return _model_cache._cache


def load_model(model_path: str | Path = MODEL_PATH):
    """Load the trained pipeline, caching it across calls in the same process.

    The model is never retrained here - it must already exist at
    ``model_path`` (produced by ``python -m src.train``).
    """
    model_path = Path(model_path)
    cache = _model_cache()

    cache_key = str(model_path)
    if cache_key not in cache:
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run 'python -m src.train' first to generate it."
            )
        cache[cache_key] = joblib.load(model_path)

    return cache[cache_key]


def prepare_customer_dataframe(customer_data: dict) -> pd.DataFrame:
    """Validate and convert a raw customer dictionary into a one-row
    dataframe with the same engineered features used at training time.
    """
    missing = [field for field in REQUIRED_FIELDS if field not in customer_data]
    if missing:
        raise ChurnPredictionError(f"Missing required field(s): {missing}")

    row = {field: customer_data[field] for field in REQUIRED_FIELDS}
    df = pd.DataFrame([row])

    try:
        df["BeginDate"] = pd.to_datetime(df["BeginDate"], format="mixed")
    except (ValueError, TypeError) as exc:
        raise ChurnPredictionError(f"Invalid BeginDate value: {exc}") from exc

    for numeric_field in ("MonthlyCharges", "TotalCharges"):
        df[numeric_field] = pd.to_numeric(df[numeric_field], errors="coerce")
        if df[numeric_field].isna().any():
            raise ChurnPredictionError(f"Invalid numeric value for {numeric_field}")

    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # Reuse the exact same helpers used to build the training data so
    # NumServices / HasInternet / HasPhone are computed identically.
    df = data_processing.fill_missing_service_values(df)
    df = data_processing.add_service_usage_features(df)

    return df


def predict_churn(
    customer_data: dict,
    model=None,
    threshold: float = DEFAULT_THRESHOLD,
    reference_date: pd.Timestamp | None = None,
) -> tuple[int, float]:
    """Predict churn for a single customer.

    Parameters
    ----------
    customer_data:
        Dictionary with the raw fields listed in ``REQUIRED_FIELDS``.
    model:
        Optional pre-loaded pipeline. If not provided, the model is
        loaded (and cached) from ``models/churn_model.pkl``.
    threshold:
        Probability threshold used to turn the predicted probability
        into a 0/1 class. The notebook did not optimize a custom
        threshold (AUC-ROC is threshold-independent), so the standard
        0.5 is used by default.
    reference_date:
        Date used to compute the customer's age in months. Defaults to
        the dataset's fixed reference date (2020-02-01), the same one
        used at training time, so predictions do not suffer from
        distribution shift (see ``src.feature_engineering`` module
        docstring for details).

    Returns
    -------
    (prediction, probability):
        ``prediction`` is 0 (stays) or 1 (churns) using ``threshold``.
        ``probability`` is the model's estimated churn probability
        (float in [0, 1]).
    """
    if model is None:
        model = load_model()

    raw_df = prepare_customer_dataframe(customer_data)
    X = feature_engineering.build_inference_features(raw_df, reference_date=reference_date)

    probability = float(model.predict_proba(X)[:, 1][0])
    prediction = int(probability >= threshold)

    return prediction, probability
