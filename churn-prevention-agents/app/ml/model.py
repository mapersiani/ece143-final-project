import pandas as pd
import numpy as np

# Hardcoded mock churn data — replace with real model inference later
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

# Precomputed churn probabilities (simulating model output)
MOCK_CHURN_PROBABILITIES = {
    "C001": 0.82, "C002": 0.15, "C003": 0.91, "C004": 0.08,
    "C005": 0.78, "C006": 0.65, "C007": 0.22, "C008": 0.95,
    "C009": 0.11, "C010": 0.71,
}

# Simulated SHAP-style top drivers per customer
MOCK_TOP_DRIVERS = {
    "C001": "MonthlyCharges", "C003": "tenure",          "C005": "MonthlyCharges",
    "C006": "TechSupport",    "C008": "Contract",        "C010": "MonthlyCharges",
}

FEATURE_NAMES = ["MonthlyCharges", "tenure", "Contract", "TechSupport", "InternetService"]


def predict_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return mock churn predictions for a fixed set of example customers.

    :param df: Input DataFrame; currently ignored in favor of hardcoded mock data.
    :return: A DataFrame with customer attributes, churn probability, and churn risk bucket.
    """
    # Use mock data regardless of input — placeholder for real model
    mock_df = pd.DataFrame(MOCK_CUSTOMERS)
    mock_df["churn_probability"] = mock_df["customerID"].map(MOCK_CHURN_PROBABILITIES).fillna(0.5)
    mock_df["churn_risk"] = pd.cut(
        mock_df["churn_probability"],
        bins=[0, 0.4, 0.65, 1.0],
        labels=["low", "medium", "high"],
    )
    return mock_df


def get_top_churn_drivers(df: pd.DataFrame, top_n: int = 5) -> list[str]:
    """
    Compute the most frequent churn drivers across customers using mock SHAP-style data.

    :param df: DataFrame of customers; unused but kept for future real model integration.
    :param top_n: Maximum number of top driver names to return.
    :return: A list of feature names ranked by frequency as top churn drivers.
    """
    # Return hardcoded top drivers across the population
    driver_counts: dict[str, int] = {}
    for driver in MOCK_TOP_DRIVERS.values():
        driver_counts[driver] = driver_counts.get(driver, 0) + 1
    sorted_drivers = sorted(driver_counts, key=driver_counts.get, reverse=True)
    return sorted_drivers[:top_n]


def segment_customers(df: pd.DataFrame) -> dict[str, list]:
    """
    Group customers into segments keyed by their top churn driver.

    :param df: DataFrame of customers that must include a `customerID` column.
    :return: A mapping from driver-based segment IDs to lists of row indices or IDs.
    """
    # Group customers by their top churn driver
    segments: dict[str, list] = {}
    for customer_id, driver in MOCK_TOP_DRIVERS.items():
        segment = f"driver_{driver}"
        segments.setdefault(segment, []).append(customer_id)

    # Map customer IDs back to DataFrame index positions
    if "customerID" in df.columns:
        id_to_idx = {row["customerID"]: idx for idx, row in df.iterrows()}
        return {
            seg: [id_to_idx[cid] for cid in cids if cid in id_to_idx]
            for seg, cids in segments.items()
        }
    return segments
