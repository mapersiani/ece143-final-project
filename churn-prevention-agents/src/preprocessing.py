"""
Data preprocessing and feature engineering for churn prediction.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


DROP_COLS = ["security_no", "profile_tags"]

REDUNDANT_COLS = ["support_risk"]


def preprocess(df: pd.DataFrame, target_col: str = "churn_risk_score") -> tuple[pd.DataFrame, pd.Series]:
    """Clean, encode, and split features from target.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    target_col : str
        Name of the binary target column.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (numeric, one-hot encoded).
    y : pd.Series
        Binary target.
    """
    assert target_col in df.columns, f"Target '{target_col}' not in columns"

    y = df[target_col].astype(int)

    drop = [c for c in DROP_COLS + REDUNDANT_COLS + [target_col] if c in df.columns]
    X = df.drop(columns=drop).copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X = X.fillna(0)
    return X, y


def get_churn_stats_by_column(df: pd.DataFrame, col: str,
                               target_col: str = "churn_risk_score") -> pd.DataFrame:
    """Compute churn rate and count per group in a given column."""
    grouped = df.groupby(col)[target_col].agg(["mean", "count"])
    grouped.columns = ["churn_rate", "count"]
    return grouped.sort_values("churn_rate", ascending=False)


def compute_correlations(df: pd.DataFrame, target_col: str = "churn_risk_score") -> pd.Series:
    """Return absolute correlations of numeric features with the target, sorted descending."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
    return corr


def get_numeric_stats_by_churn(df: pd.DataFrame, numeric_cols: list[str],
                                target_col: str = "churn_risk_score") -> pd.DataFrame:
    """Compute median of numeric features grouped by churn status."""
    rows = []
    for col in numeric_cols:
        m0 = df.loc[df[target_col] == 0, col].median()
        m1 = df.loc[df[target_col] == 1, col].median()
        rows.append({"feature": col, "retained_median": m0, "churned_median": m1, "delta": m1 - m0})
    return pd.DataFrame(rows)
