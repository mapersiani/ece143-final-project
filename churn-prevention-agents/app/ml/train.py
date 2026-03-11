import mlflow
import os

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Placeholder training script — logs a mock experiment to MLFlow
# Replace this with real XGBoost training once model is ready
def train(params: dict | None = None):
    """
    Log a placeholder churn model training run and feature importances to MLflow.

    :param params: Optional extra parameters to log alongside mock defaults.
    :return: A tuple of (None, list of feature names) mirroring a future train API.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("churn-prediction")

    mock_params = {"model": "placeholder", "note": "hardcoded mock data", **(params or {})}
    mock_metrics = {"auc": 0.88, "precision_churn": 0.74, "recall_churn": 0.69, "f1_churn": 0.71}

    with mlflow.start_run(run_name="placeholder-run"):
        mlflow.log_params(mock_params)
        mlflow.log_metrics(mock_metrics)
        mlflow.log_dict(
            {"MonthlyCharges": 0.38, "tenure": 0.31, "Contract": 0.18, "TechSupport": 0.13},
            "feature_importances.json",
        )
        print("Placeholder MLFlow run logged. AUC: 0.88 (mock)")

    return None, ["MonthlyCharges", "tenure", "Contract", "TechSupport", "InternetService"]


if __name__ == "__main__":
    train()
