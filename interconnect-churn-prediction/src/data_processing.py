"""
Data loading and cleaning utilities.

This module reproduces exactly the data-loading and data-cleaning steps
performed in the original analysis notebook
(``notebooks/interconnect_churn_analysis.ipynb``, sections 2 to 4):

1. Load the four raw sources (``contract``, ``personal``, ``internet``,
   ``phone``) and merge them on ``customerID``.
2. Convert ``TotalCharges`` to numeric and fill missing values with 0
   (these correspond to brand-new customers with no accumulated charges
   yet, as verified in the notebook).
3. Convert ``BeginDate`` / ``EndDate`` to datetime, keeping the original
   ``EndDate`` (text, "No" for active customers) and creating
   ``EndDateDate`` (datetime, ``NaT`` for active customers).
4. Build the binary target ``Churn`` (1 = customer cancelled).
5. Fill the missing values created by the left joins with explicit
   "no service" categories.
6. Create the simple service-usage features ``NumServices``,
   ``HasInternet`` and ``HasPhone``.

The functions here are intentionally free of any modelling decisions
(feature selection, leakage prevention, final feature engineering) -
those live in ``src/feature_engineering.py`` so that ``train.py`` and
``predict.py`` can reuse the exact same logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Columns that describe internet-related services. A customer missing
# from internet.csv simply has no internet service.
INTERNET_SERVICE_COLUMNS = [
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Columns used to count how many "add-on" services a customer has.
SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
]

RAW_FILES = ("contract.csv", "personal.csv", "internet.csv", "phone.csv")


def load_raw_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the four raw CSV files from ``data_dir``.

    Parameters
    ----------
    data_dir:
        Path to the folder containing ``contract.csv``, ``personal.csv``,
        ``internet.csv`` and ``phone.csv`` (i.e. ``data/final_provider``).
    """
    data_dir = Path(data_dir)

    missing = [f for f in RAW_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw data file(s) in {data_dir}: {missing}. "
            "Expected contract.csv, personal.csv, internet.csv and phone.csv."
        )

    return {
        "contract": pd.read_csv(data_dir / "contract.csv"),
        "personal": pd.read_csv(data_dir / "personal.csv"),
        "internet": pd.read_csv(data_dir / "internet.csv"),
        "phone": pd.read_csv(data_dir / "phone.csv"),
    }


def merge_sources(
    contract: pd.DataFrame,
    personal: pd.DataFrame,
    internet: pd.DataFrame,
    phone: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join the four sources on ``customerID`` (contract as base)."""
    return (
        contract.merge(personal, on="customerID", how="left")
        .merge(internet, on="customerID", how="left")
        .merge(phone, on="customerID", how="left")
    )


def clean_contract(contract: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's cleaning steps to the ``contract`` table.

    - ``BeginDate`` -> datetime
    - ``EndDate`` -> kept as text ("No" for active customers)
    - ``EndDateDate`` -> datetime version of ``EndDate`` (``NaT`` if active)
    - ``TotalCharges`` -> numeric, missing values filled with 0
    - ``Churn`` -> 1 if the customer cancelled, else 0
    """
    contract = contract.copy()

    contract["BeginDate"] = pd.to_datetime(contract["BeginDate"], format="mixed")
    contract["EndDateDate"] = pd.to_datetime(
        contract["EndDate"], errors="coerce", format="mixed"
    )

    contract["TotalCharges"] = pd.to_numeric(contract["TotalCharges"], errors="coerce")
    contract["TotalCharges"] = contract["TotalCharges"].fillna(0)

    contract["Churn"] = (contract["EndDate"] != "No").astype(int)

    return contract


def fill_missing_service_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs created by the left joins with explicit "no service" labels."""
    df = df.copy()

    df[INTERNET_SERVICE_COLUMNS] = df[INTERNET_SERVICE_COLUMNS].fillna(
        "No internet service"
    )
    df["MultipleLines"] = df["MultipleLines"].fillna("No phone service")

    return df


def add_service_usage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``NumServices``, ``HasInternet`` and ``HasPhone``."""
    df = df.copy()

    df["NumServices"] = sum(
        (df[column] == "Yes").astype(int) for column in SERVICE_COLUMNS
    )

    df["HasInternet"] = (df["InternetService"] != "No internet service").astype(int)
    df["HasPhone"] = (df["MultipleLines"] != "No phone service").astype(int)

    return df


def build_clean_dataset(data_dir: str | Path) -> pd.DataFrame:
    """Run the full loading + cleaning pipeline and return one clean dataframe.

    This reproduces notebook sections 2-4 end to end and is the single
    entry point used by both ``train.py`` (on the full historical data)
    and, conceptually, by the feature-engineering step used for a single
    new observation.
    """
    raw = load_raw_data(data_dir)

    contract = clean_contract(raw["contract"])

    df = merge_sources(contract, raw["personal"], raw["internet"], raw["phone"])
    df = fill_missing_service_values(df)
    df = add_service_usage_features(df)

    return df
