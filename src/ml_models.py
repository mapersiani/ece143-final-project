import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_churn_models(X_train, y_train, seed=42):
    """
    Trains three ensemble models to provide diverse perspectives on churn risk.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        seed (int): Random state for reproducibility.

    Validation:
        - Ensures X and y have matching dimensions.
        - Verifies that no NaN values remain in the training set.

    Returns:
        models : A dictionary containing the three trained model objects.
    """
    assert len(X_train) == len(y_train), "Features and Target length mismatch"
    assert not np.isnan(X_train.values).any(), "Training features contain NaNs"

    models = {
        "Histo": HistGradientBoostingClassifier(random_state=seed).fit(X_train, y_train),
        "RF": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed).fit(X_train, y_train),
        "GB": GradientBoostingClassifier(random_state=seed).fit(X_train, y_train)
    }
    return models

def generate_predictions(models, X_input, model_type='RF'):
    """
    Generates class predictions and probabilities using a specified model.

    Parameters:
        models (dict): Dictionary of trained models from train_council_of_experts.
        X_input (pd.DataFrame): Data to predict on.
        model_type (str): Key of the model to use ('Histo', 'RF', or 'GB').

    Returns:
        tuple: (y_pred, y_prob) containing hard labels and churn probabilities.
    """
    assert model_type in models, f"Model type '{model_type}' not found in the provided model dictionary."
    assert hasattr(models[model_type], "predict_proba"), "Selected model is not capable of probability output."

    selected_model = models[model_type]
    
    y_pred = selected_model.predict(X_input)
    y_prob = selected_model.predict_proba(X_input)[:, 1]

    return y_pred, y_prob

def evaluate_performance(y_true, y_pred, model_name="Model"):
    """
    Validates and evaluates model performance using standard metrics.

    Parameters:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        model_name (str): Name of the model for display purposes.
    """
    assert isinstance(y_true, (pd.Series, np.ndarray, list)), "y_true must be array"
    assert isinstance(y_pred, (pd.Series, np.ndarray, list)), "y_pred must be array"

    assert len(y_true) == len(y_pred), f"Length mismatch: y_true({len(y_true)}) != y_pred({len(y_pred)})"

    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    assert np.all(np.isin(unique_pred, unique_true)), "y_pred contains classes not found in y_true"

    print(f"\n{'='*20} {model_name} Performance Report {'='*20}")
    print(classification_report(y_true, y_pred))
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Calculated Accuracy: {accuracy:.4f}")
    
    return {"accuracy": accuracy}

def get_feature_importance(model, feature_names):
    """
    Extracts, validates, and sorts feature importance for visualization.

    Parameters:
        model: A trained model object (e.g., RandomForestClassifier).
        feature_names (list): The list of feature names used during training.

    Returns:
        pd.DataFrame: A sorted DataFrame of features and their importance scores.
    """
    # -Attribute Check: Not all models have .feature_importances_ 
    # (e.g., LogisticRegression uses .coef_, HistGradientBoosting uses permutation importance)
    assert hasattr(model, 'feature_importances_'), \
        f"The provided model ({type(model).__name__}) does not support the 'feature_importances_' attribute."

    assert isinstance(feature_names, (list, np.ndarray, pd.Index)), "feature_names must be a list or array-like."

    importances = model.feature_importances_
    assert len(importances) == len(feature_names), \
        f"Dimension mismatch: Model has {len(importances)} features, but {len(feature_names)} names were provided."
    
    importance_df = pd.DataFrame({
        'Feature': list(feature_names), 
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return importance_df