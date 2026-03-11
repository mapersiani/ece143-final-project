"""
Data loading and validation utilities for the churn prevention pipeline.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

EXPECTED_COLUMNS = [
    "age", "gender", "security_no", "region_category", "membership_category",
    "joined_through_referral", "preferred_offer_types", "medium_of_operation",
    "internet_option", "days_since_last_login", "avg_time_spent",
    "avg_transaction_value", "avg_frequency_login_days", "points_in_wallet",
    "used_special_discount", "offer_application_preference", "past_complaint",
    "complaint_status", "feedback", "churn_risk_score", "tenure_days",
    "value_segment", "engagement_segment", "price_sensitivity", "support_risk",
    "profile_tags",
]

TARGET_COL = "churn_risk_score"


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the customer churn features dataset.

    Parameters
    ----------
    path : str or Path, optional
        Path to CSV file.  Falls back to data/customer_churn_features.csv.

    Returns
    -------
    pd.DataFrame
    """
    if path is None:
        path = DATA_DIR / "customer_churn_features.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """Run basic validation checks and return a summary dict."""
    summary = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "has_target": TARGET_COL in df.columns,
        "null_counts": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    if summary["has_target"]:
        vc = df[TARGET_COL].value_counts()
        summary["churn_distribution"] = vc.to_dict()
        summary["churn_rate"] = float(df[TARGET_COL].mean())
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    summary["missing_columns"] = list(missing) if missing else []
    return summary


def get_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Categorize columns into logical groups."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return {
        "numeric": [c for c in numeric if c != TARGET_COL],
        "categorical": categorical,
        "target": TARGET_COL,
        "id_cols": ["security_no"],
        "tag_cols": ["profile_tags"],
    }
