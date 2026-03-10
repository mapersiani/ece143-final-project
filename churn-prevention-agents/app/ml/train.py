import os
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_PATH = Path("/app/models/churn_model.joblib")


def train(df: pd.DataFrame, target_col: str = "churn_risk_score", params: dict | None = None) -> dict:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

    y = df[target_col].astype(int)
    drop_cols = [target_col, "security_no", "profile_tags"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    scale_pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)

    cfg = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": round(scale_pos_weight, 3),
        "eval_metric": "logloss",
        "random_state": 42,
        **(params or {}),
    }

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
    logger.info(f"XGBoost metrics: {metrics}")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = importances.head(10).to_dict()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "trained_columns": list(X.columns),
        "feature_columns": feature_cols,
        "target_col": target_col,
    }
    joblib.dump(bundle, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("churn-prediction")
    with mlflow.start_run(run_name="xgb-churn-train") as run:
        mlflow.log_params({"model": "xgboost", "target_col": target_col, **cfg})
        mlflow.log_metrics(metrics)
        mlflow.log_dict(importances.head(20).to_dict(), "feature_importances.json")
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    return {
        "status": "ok",
        "run_id": run_id,
        "metrics": metrics,
        "top_features": top_features,
        "model_path": str(MODEL_PATH),
        "n_rows": int(len(df)),
        "n_features": int(len(X.columns)),
    }
