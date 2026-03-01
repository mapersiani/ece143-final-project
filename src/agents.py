import numpy as np
import pandas as pd

def generate_strategic_segments(df, churn_probs):
    """
    Categorizes customers into a 2x2 matrix of Risk vs. Value.
    
    Parameters:
        df (pd.DataFrame): The customer data containing 'points_in_wallet'.
        churn_probs (np.ndarray): Probability array from the model.
        
    Returns:
        pd.DataFrame: Original dataframe with 'segment' and 'action' columns.
    """
    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
    assert isinstance(churn_probs, np.ndarray), "churn_probs must be a numpy array"
    assert len(df) == len(churn_probs), "Mismatch: DataFrame and Probabilities must have same length"
    
    assert np.all((churn_probs >= 0) & (churn_probs <= 1)), "churn_probs must be between 0 and 1"
    
    assert 'points_in_wallet' in df.columns, "Critical column 'points_in_wallet' missing for segmentation"
    assert not df['points_in_wallet'].isnull().any(), "Found NaNs in points_in_wallet. Clean data first."

    df = df.copy()
    df['churn_probability'] = churn_probs
    

    point_median = df['points_in_wallet'].median()
    
    # 2x2 Matrix Conditions
    # 1. High Risk (>0.7), High Value (>Median) -> VIP Recovery
    # 2. High Risk (>0.7), Low Value (<=Median) -> Standard Retarget
    # 3. Low Risk (<=0.7), High Value (>Median) -> Loyalty Rewards
    # 4. Low Risk (<=0.7), Low Value (<=Median) -> Upsell Opportunity
    
    conditions = [
        (df['churn_probability'] > 0.7) & (df['points_in_wallet'] > point_median),
        (df['churn_probability'] > 0.7) & (df['points_in_wallet'] <= point_median),
        (df['churn_probability'] <= 0.7) & (df['points_in_wallet'] > point_median),
        (df['churn_probability'] <= 0.7) & (df['points_in_wallet'] <= point_median)
    ]
    
    choices = [
        'High Value - At Risk (VIP Recovery)', 
        'Low Value - At Risk (Standard Retarget)', 
        'High Value - Stable (Loyalty Rewards)', 
        'Low Value - Stable (Upsell Opportunity)'
    ]
    
    df['segment'] = np.select(conditions, choices, default='Unclassified')

    # Assign specific business actions based on the segment
    action_map = {
        'High Value - At Risk (VIP Recovery)': 'Personalized Phone Call & Premium Voucher',
        'Low Value - At Risk (Standard Retarget)': 'Automated Email Discount',
        'High Value - Stable (Loyalty Rewards)': 'Membership Anniversary Gift',
        'Low Value - Stable (Upsell Opportunity)': 'Upgrade Offer Invitation'
    }
    
    df['suggested_action'] = df['segment'].map(action_map)
    
    return df