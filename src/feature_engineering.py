"""
Feature engineering used to train and to serve the churn model.

This module reproduces notebook section "9.1 Ingeniería de características
y prevención de fuga de información" exactly, and defines the final list
of features used to train the LightGBM model (section 9.4).

Leakage prevention
-------------------
The notebook explicitly drops the following columns before training
because they either identify the customer or leak the outcome:

- ``customerID``  -> pure identifier, no predictive meaning.
- ``EndDate`` / ``EndDateDate`` -> directly encode whether/when the
  customer cancelled; using them would leak the target.
- ``TenureMonths`` -> the exploratory version of "customer tenure" used
  ``EndDateDate`` for churned customers, so it indirectly leaks the
  target too. It is dropped in favor of ``CustomerAgeMonths`` below,
  which only ever uses ``BeginDate`` and a reference date that is known
  independently of the outcome.
- ``Churn`` -> the target itself.

Reference date - training AND inference
------------------------------------------
The notebook computes ``CustomerAgeMonths`` (and ``BeginYear`` /
``BeginMonth``) relative to a fixed reference date,
``pd.Timestamp('2020-02-01')``, because that is the date at which the
historical dataset snapshot was taken.

The model was trained on the distribution of customer ages that exist
relative to that snapshot date. Computing ``CustomerAgeMonths`` at
inference time relative to *today* instead would introduce a
distribution shift - the model would be scoring feature values it
never saw during training/validation/test. To keep training and
inference consistent, ``REFERENCE_DATE`` (2020-02-01) is defined once
below and reused everywhere a customer's age needs to be computed:

- ``train.py`` (via ``build_training_features``) uses it to reproduce
  the notebook's historical training data exactly.
- ``predict.py`` / ``app.py`` (via ``build_inference_features``) use
  the same fixed date by default, so every prediction reflects the
  same temporal context the model was trained and evaluated on.

This means the application effectively answers "what would this
customer's churn risk have been as of the dataset's reference
snapshot (2020-02-01)?" rather than "as of today" - which is the
correct question to ask of this model as currently trained. A real
production deployment would retrain the model periodically on recent
data and advance the reference date accordingly; that retraining
pipeline is outside the scope of this portfolio project.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Single, reusable reference date used for BOTH training and inference,
# matching the moment the historical dataset snapshot was taken. Using
# the same fixed date everywhere avoids distribution shift between
# training and serving - see the module docstring above.
REFERENCE_DATE = pd.Timestamp("2020-02-01")

# Columns dropped before training (identifiers + leakage-prone columns).
COLUMNS_TO_DROP = [
    "customerID",
    "EndDate",
    "EndDateDate",
    "BeginDate",
    "TenureMonths",
]

TARGET_COLUMN = "Churn"

# Final feature lists exactly as identified in the notebook
# (section 9.4, cell 99 output).
NUMERIC_FEATURES = [
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "NumServices",
    "HasInternet",
    "HasPhone",
    "CustomerAgeMonths",
    "BeginYear",
    "BeginMonth",
    "IsMonthToMonth",
    "AutomaticPayment",
    "SecuritySupportCount",
]

CATEGORICAL_FEATURES = [
    "Type",
    "PaperlessBilling",
    "PaymentMethod",
    "gender",
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

# Column order expected by the trained pipeline's ColumnTransformer.
MODEL_FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES

AUTOMATIC_PAYMENT_METHODS = [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


def add_temporal_and_contract_features(
    df: pd.DataFrame,
    reference_date: pd.Timestamp = REFERENCE_DATE,
) -> pd.DataFrame:
    """Add CustomerAgeMonths, BeginYear, BeginMonth, IsMonthToMonth,
    AutomaticPayment and SecuritySupportCount, exactly as in the notebook.

    Parameters
    ----------
    df:
        Dataframe containing at least ``BeginDate`` (datetime), ``Type``,
        ``PaymentMethod``, ``OnlineSecurity`` and ``TechSupport``.
    reference_date:
        Date used to compute customer age in months. Defaults to the
        dataset's fixed reference date (2020-02-01), used both for
        training and inference to avoid distribution shift.
    """
    df = df.copy()

    df["CustomerAgeMonths"] = (
        (reference_date.year - df["BeginDate"].dt.year) * 12
        + (reference_date.month - df["BeginDate"].dt.month)
    )

    df["BeginYear"] = df["BeginDate"].dt.year
    df["BeginMonth"] = df["BeginDate"].dt.month

    df["IsMonthToMonth"] = (df["Type"] == "Month-to-month").astype(int)

    df["AutomaticPayment"] = df["PaymentMethod"].isin(
        AUTOMATIC_PAYMENT_METHODS
    ).astype(int)

    df["SecuritySupportCount"] = (
        (df["OnlineSecurity"] == "Yes").astype(int)
        + (df["TechSupport"] == "Yes").astype(int)
    )

    return df


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier/leakage columns and the target, keeping the model's
    input features in the exact order used during training.
    """
    columns_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    columns_to_drop = columns_to_drop + (
        [TARGET_COLUMN] if TARGET_COLUMN in df.columns else []
    )

    X = df.drop(columns=columns_to_drop)

    missing = [c for c in MODEL_FEATURE_ORDER if c not in X.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    return X[MODEL_FEATURE_ORDER]


def build_training_features(
    clean_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the (X, y) matrices used to train the model, from the clean,
    merged dataset produced by ``data_processing.build_clean_dataset``.

    Uses the fixed dataset reference date so the resulting training data
    matches the notebook exactly.
    """
    model_df = add_temporal_and_contract_features(
        clean_df, reference_date=REFERENCE_DATE
    )

    X = select_model_features(model_df)
    y = model_df[TARGET_COLUMN]

    return X, y


def build_inference_features(
    raw_df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build the model-ready feature row(s) for one or more new customers.

    ``raw_df`` must already contain the service-usage columns
    (``NumServices``, ``HasInternet``, ``HasPhone``) - see
    ``src.data_processing.add_service_usage_features`` - plus
    ``BeginDate`` and all raw contract/personal/service columns.

    Parameters
    ----------
    reference_date:
        Date used to compute ``CustomerAgeMonths``/``BeginYear``/
        ``BeginMonth``. Defaults to the same fixed dataset reference
        date (2020-02-01) used at training time, so inference does not
        introduce a distribution shift - see the module docstring.
    """
    if reference_date is None:
        reference_date = REFERENCE_DATE

    model_df = add_temporal_and_contract_features(
        raw_df, reference_date=reference_date
    )

    return select_model_features(model_df)


def build_preprocessor() -> ColumnTransformer:
    """Build the tree-model preprocessor used by the winning LightGBM
    pipeline: numeric features pass through untouched, categorical
    features are one-hot encoded (unknown categories at inference time
    are safely ignored rather than raising an error).

    This mirrors ``preprocessor_tree`` from the notebook (section 9.6).
    """
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
