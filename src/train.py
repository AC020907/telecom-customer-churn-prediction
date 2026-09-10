"""
Train the final Interconnect churn-prediction model.

Reproduces the winning model from ``notebooks/interconnect_churn_analysis.ipynb``:

    LightGBM (LGBMClassifier) inside a scikit-learn Pipeline
    (ColumnTransformer preprocessing + LGBMClassifier), tuned with
    RandomizedSearchCV(scoring="roc_auc") and re-fit on the combined
    train + validation data before the final test evaluation.

To avoid re-running a ~35-iteration randomized hyperparameter search
(which is stochastic and slow) every time someone reproduces the
project, this script uses the exact best hyperparameters found in the
notebook's search (see ``LIGHTGBM_BEST_PARAMS`` below) and re-fits the
model with them. This is the standard, correct way to make a tuned
notebook result reproducible in a training script, and it yields the
same architecture, preprocessing and hyperparameters as the notebook -
nothing about the winning model is changed.

Usage
-----
    python -m src.train

Produces:
    models/churn_model.pkl           (joblib-serialized sklearn Pipeline)
    models/model_metadata.json       (metrics + feature lists for the app)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import data_processing, feature_engineering

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "final_provider"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 12345

# Best hyperparameters found by RandomizedSearchCV(scoring="roc_auc")
# in notebook section 11.5 ("Ajuste de LightGBM").
LIGHTGBM_BEST_PARAMS = {
    "n_estimators": 997,
    "learning_rate": 0.0790260231986339,
    "num_leaves": 13,
    "max_depth": 8,
    "min_child_samples": 34,
    "subsample": 0.9455050034169743,
    "colsample_bytree": 0.9698397677207304,
    "reg_alpha": 1.6069369739074215,
    "reg_lambda": 0.43785559304509875,
    "class_weight": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


def evaluate(model, X, y) -> dict:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", feature_engineering.build_preprocessor()),
            ("model", LGBMClassifier(**LIGHTGBM_BEST_PARAMS)),
        ]
    )


def main() -> None:
    print(f"Loading raw data from: {DATA_DIR}")
    clean_df = data_processing.build_clean_dataset(DATA_DIR)

    X, y = feature_engineering.build_training_features(clean_df)
    print(f"Feature matrix: {X.shape}, target distribution:\n{y.value_counts(normalize=True)}")

    # Same 60/20/20 stratified split as the notebook.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape} | Valid: {X_valid.shape} | Test: {X_test.shape}")

    # Sanity check: a dummy classifier should score ~0.5 AUC-ROC.
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    dummy_metrics = evaluate(dummy, X_valid, y_valid)
    print(f"Sanity check - DummyClassifier validation AUC-ROC: {dummy_metrics['roc_auc']:.4f}")

    # Fit on train, check validation performance (informational).
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    valid_metrics = evaluate(pipeline, X_valid, y_valid)
    print("Validation metrics:", {k: round(v, 4) for k, v in valid_metrics.items()})

    # Re-fit on train + validation combined, exactly like the notebook,
    # before the single final evaluation on the untouched test set.
    X_train_full = pd.concat([X_train, X_valid], axis=0)
    y_train_full = pd.concat([y_train, y_valid], axis=0)

    final_pipeline = build_pipeline()
    final_pipeline.fit(X_train_full, y_train_full)

    test_metrics = evaluate(final_pipeline, X_test, y_test)
    print("Final test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})

    # Feature importance (native LightGBM feature_importances_) for the
    # "Model Insights" section of the Streamlit app.
    fitted_preprocessor = final_pipeline.named_steps["preprocessor"]
    fitted_model = final_pipeline.named_steps["model"]

    categorical_names = list(
        fitted_preprocessor.named_transformers_["cat"].get_feature_names_out(
            feature_engineering.CATEGORICAL_FEATURES
        )
    )
    feature_names = feature_engineering.NUMERIC_FEATURES + categorical_names

    # LGBMClassifier.feature_importances_ uses LightGBM's default
    # importance_type="split": the number of times each feature is used
    # to split a tree, NOT a gain-based or SHAP importance.
    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": fitted_model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(20)
        .to_dict(orient="records")
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    metadata = {
        "model_name": "LightGBM (LGBMClassifier)",
        "hyperparameters": LIGHTGBM_BEST_PARAMS,
        "numeric_features": feature_engineering.NUMERIC_FEATURES,
        "categorical_features": feature_engineering.CATEGORICAL_FEATURES,
        "decision_threshold": 0.5,
        "threshold_note": (
            "No explicit threshold optimization was performed in the "
            "original analysis - AUC-ROC was the primary evaluation "
            "metric and is computed directly from predicted "
            "probabilities. The standard 0.5 threshold is used to turn "
            "probabilities into a High/Low risk classification."
        ),
        "validation_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "reference_test_roc_auc_from_notebook": 0.9453,
        "feature_importance_type": "split",
        "feature_importance_note": (
            "LightGBM split-based feature importance: the number of "
            "times each feature was used to split a tree "
            "(LGBMClassifier's default importance_type='split'), not a "
            "gain-based or SHAP importance."
        ),
        "top_features": feature_importance,
        "random_state": RANDOM_STATE,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
