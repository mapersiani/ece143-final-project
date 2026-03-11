"""
XGBoost model training, evaluation, and feature importance extraction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

from src.preprocessing import preprocess


DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
}


def train_xgboost(
    df: pd.DataFrame,
    target_col: str = "churn_risk_score",
    params: dict | None = None,
    test_size: float = 0.2,
) -> dict:
    """Train an XGBoost classifier and return model, metrics, and feature importances.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with target column.
    target_col : str
        Binary target column name.
    params : dict, optional
        XGBoost hyperparameters (merged with defaults).
    test_size : float
        Fraction held out for testing.

    Returns
    -------
    dict with keys: model, metrics, feature_importances, X_test, y_test, y_pred, y_proba, columns
    """
    X, y = preprocess(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    scale_pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)

    cfg = {**DEFAULT_PARAMS, "scale_pos_weight": round(scale_pos_weight, 3), **(params or {})}

    model = XGBClassifier(**cfg)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return {
        "model": model,
        "metrics": metrics,
        "feature_importances": importances,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "columns": list(X.columns),
        "params": cfg,
    }


def get_top_features(importances: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """Return a DataFrame of the top-N most important features."""
    top = importances.head(top_n).reset_index()
    top.columns = ["feature", "importance"]
    top["importance_pct"] = (top["importance"] * 100).round(1)
    return top
