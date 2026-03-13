import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from preprocess import preprocess_hf_dataset
from ml_models import train_churn_models, generate_predictions, evaluate_performance
from segments import generate_strategic_segments

SEED = 42
RAW_DATA_PATH = Path("../../dataset/customer_churn.csv")
OUTPUT_DIR = Path("../../outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    """
    Executes the churn prediction lifecycle:
    1. Preprocessing (Data Pipeline)
    2. Training 
    3. Evaluation 
    4. Actionable Intelligence 
    """
    
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing raw data at {RAW_DATA_PATH}. Please provide the dataset.")
    
    df_raw = pd.read_csv(RAW_DATA_PATH)
    
    df_clean = preprocess_hf_dataset(df_raw)
    
    X = df_clean.drop(columns=['churn_risk_score', 'security_no'], errors='ignore')
    y = df_clean['churn_risk_score']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    models = train_churn_models(X_train, y_train, seed=SEED)
    
    y_pred_histo, _ = generate_predictions(models, X_test, model_type='Histo')
    evaluate_performance(y_test, y_pred_histo, model_name="HistGradientBoosting (Histo)")
    
    y_pred_rf, _ = generate_predictions(models, X_test, model_type='RF')
    evaluate_performance(y_test, y_pred_rf, model_name="Random Forest (RF)")

    X_full = df_clean.drop(columns=['churn_risk_score', 'security_no'], errors='ignore')
    _, full_probs = generate_predictions(models, X_full, model_type='RF')
    
    final_results = generate_strategic_segments(df_clean, full_probs)

    final_results.to_csv(OUTPUT_DIR / "final_customer_intelligence.csv", index=False)
    
    campaign_cols = ['security_no', 'segment', 'suggested_action', 'churn_probability']
    final_results[campaign_cols].to_csv(OUTPUT_DIR / "marketing_campaign_list.csv", index=False)

if __name__ == "__main__":
    main()