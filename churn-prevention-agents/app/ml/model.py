import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH = Path("/app/models/churn_model.joblib")

_bundle_cache: dict | None = None


def _get_model_bundle() -> dict | None:
    global _bundle_cache
    if _bundle_cache is not None:
        return _bundle_cache
    if MODEL_PATH.exists():
        try:
            _bundle_cache = joblib.load(MODEL_PATH)
            logger.info("Loaded trained model from %s", MODEL_PATH)
            return _bundle_cache
        except Exception as e:
            logger.warning("Failed to load model: %s", e)
    return None


def predict_churn(df: pd.DataFrame) -> pd.DataFrame:
    bundle = _get_model_bundle()
    if bundle is None:
        logger.warning("No trained model found — using mock data")
        return _mock_predict()

    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    trained_columns = bundle["trained_columns"]

    X = df[feature_cols].copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X = X.fillna(0)

    for col in trained_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[trained_columns]

    proba = model.predict_proba(X)[:, 1]

    importances = pd.Series(model.feature_importances_, index=trained_columns)
    top_global = importances.sort_values(ascending=False).head(5).index.tolist()

    out = df.copy()
    out["churn_probability"] = proba
    out["churn_risk"] = pd.cut(
        proba, bins=[0, 0.4, 0.65, 1.0], labels=["low", "medium", "high"],
    )
    out["top_driver"] = _assign_top_driver(X, importances)
    return out


def get_top_churn_drivers(df: pd.DataFrame, top_n: int = 5) -> list[str]:
    bundle = _get_model_bundle()
    if bundle is None:
        return ["MonthlyCharges", "tenure", "Contract", "TechSupport", "InternetService"][:top_n]

    importances = pd.Series(
        bundle["model"].feature_importances_, index=bundle["trained_columns"],
    )
    raw_top = importances.sort_values(ascending=False).head(top_n * 2)
    clean_names = []
    for feat in raw_top.index:
        base = feat.split("_")[0] if "_" in feat else feat
        if base not in clean_names:
            clean_names.append(base)
        if len(clean_names) >= top_n:
            break
    return clean_names


def segment_customers(df: pd.DataFrame) -> dict[str, list]:
    if "top_driver" not in df.columns:
        return _mock_segment(df)

    segments: dict[str, list] = {}
    for idx, row in df.iterrows():
        driver = row.get("top_driver", "unknown")
        seg = f"driver_{driver}"
        segments.setdefault(seg, []).append(idx)
    return segments


def _assign_top_driver(X: pd.DataFrame, importances: pd.Series) -> list[str]:
    """For each row, pick the feature with the highest (importance * |value|)."""
    imp = importances.values
    vals = X.values.astype(float)
    scores = np.abs(vals) * imp[np.newaxis, :]
    top_idx = scores.argmax(axis=1)
    col_names = X.columns.tolist()
    drivers = []
    for i in top_idx:
        raw = col_names[i]
        base = raw.split("_")[0] if "_" in raw else raw
        drivers.append(base)
    return drivers


# ---- Fallback mock data (used when no trained model) ----

MOCK_CUSTOMERS = [
    {"customerID": "C001", "MonthlyCharges": 89.5, "tenure": 3,  "Contract": "Month-to-month", "TechSupport": "No",  "InternetService": "Fiber optic"},
    {"customerID": "C002", "MonthlyCharges": 45.0, "tenure": 24, "Contract": "One year",        "TechSupport": "Yes", "InternetService": "DSL"},
    {"customerID": "C003", "MonthlyCharges": 110.2,"tenure": 2,  "Contract": "Month-to-month", "TechSupport": "No",  "InternetService": "Fiber optic"},
    {"customerID": "C004", "MonthlyCharges": 29.0, "tenure": 60, "Contract": "Two year",        "TechSupport": "Yes", "InternetService": "DSL"},
    {"customerID": "C005", "MonthlyCharges": 95.0, "tenure": 5,  "Contract": "Month-to-month", "TechSupport": "No",  "InternetService": "Fiber optic"},
    {"customerID": "C006", "MonthlyCharges": 70.0, "tenure": 8,  "Contract": "Month-to-month", "TechSupport": "No",  "InternetService": "Fiber optic"},
    {"customerID": "C007", "MonthlyCharges": 55.0, "tenure": 15, "Contract": "One year",        "TechSupport": "Yes", "InternetService": "DSL"},
    {"customerID": "C008", "MonthlyCharges": 120.0,"tenure": 1,  "Contract": "Month-to-month", "TechSupport": "No",  "InternetService": "Fiber optic"},
    {"customerID": "C009", "MonthlyCharges": 35.0, "tenure": 48, "Contract": "Two year",        "TechSupport": "Yes", "InternetService": "DSL"},
    {"customerID": "C010", "MonthlyCharges": 82.0, "tenure": 6,  "Contract": "Month-to-month", "TechSupport": "No",  "InternetService": "Fiber optic"},
]

MOCK_CHURN_PROBABILITIES = {
    "C001": 0.82, "C002": 0.15, "C003": 0.91, "C004": 0.08,
    "C005": 0.78, "C006": 0.65, "C007": 0.22, "C008": 0.95,
    "C009": 0.11, "C010": 0.71,
}

MOCK_TOP_DRIVERS = {
    "C001": "MonthlyCharges", "C003": "tenure", "C005": "MonthlyCharges",
    "C006": "TechSupport", "C008": "Contract", "C010": "MonthlyCharges",
}


def _mock_predict() -> pd.DataFrame:
    mock_df = pd.DataFrame(MOCK_CUSTOMERS)
    mock_df["churn_probability"] = mock_df["customerID"].map(MOCK_CHURN_PROBABILITIES).fillna(0.5)
    mock_df["churn_risk"] = pd.cut(
        mock_df["churn_probability"],
        bins=[0, 0.4, 0.65, 1.0],
        labels=["low", "medium", "high"],
    )
    mock_df["top_driver"] = mock_df["customerID"].map(MOCK_TOP_DRIVERS).fillna("unknown")
    return mock_df


def _mock_segment(df: pd.DataFrame) -> dict[str, list]:
    segments: dict[str, list] = {}
    for customer_id, driver in MOCK_TOP_DRIVERS.items():
        seg = f"driver_{driver}"
        segments.setdefault(seg, []).append(customer_id)
    if "customerID" in df.columns:
        id_to_idx = {row["customerID"]: idx for idx, row in df.iterrows()}
        return {
            seg: [id_to_idx[cid] for cid in cids if cid in id_to_idx]
            for seg, cids in segments.items()
        }
    return segments
