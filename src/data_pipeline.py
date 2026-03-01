import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_hf_dataset(df):
    '''
    Prepares the Hugging Face churn dataset with strict type validation.
    
    Parameters:
        df (pd.DataFrame): Raw dataframe from the HF source.
        
    Returns:
        pd.DataFrame: Processed features and target variable.
    '''
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert 'churn_risk_score' in df.columns, "Target column 'churn_risk_score' is missing"

    df = df.copy()
    df = df[df['churn_risk_score'] >= 0] 
    
    noise = [ 'referral_id', 'last_visit_time']
    df = df.drop(columns=[c for c in noise if c in df.columns], errors='ignore')

    # Data Cleaning
    df.replace(['?', 'xxxxxxxx'], np.nan, inplace=True)
    df['days_since_last_login'] = pd.to_numeric(df['days_since_last_login'], errors='coerce')
    # Treat -999 as NaN, then fill with median
    df.loc[df['days_since_last_login'] < 0, 'days_since_last_login'] = np.nan
    df['days_since_last_login'] = df['days_since_last_login'].fillna(df['days_since_last_login'].median())

    # Time Stuff
    df['joining_date'] = pd.to_datetime(df['joining_date'], format='mixed', errors='coerce')
    # Fill missing dates with the most common date so tenure isn't NaN
    df['joining_date'] = df['joining_date'].fillna(df['joining_date'].mode()[0])
    reference_date = df['joining_date'].max()
    df['tenure_days'] = (reference_date - df['joining_date']).dt.days

    binary_map = {'Yes': 1, 'No': 0}
    binary_cols = ['used_special_discount', 'offer_application_preference', 'past_complaint', 'joined_through_referral']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0)

    # Number Values
    numeric_cols = ['avg_time_spent', 'avg_transaction_value', 'points_in_wallet', 'age', 'avg_frequency_login_days']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].clip(lower=0)  # Neutralize negative value bugs
        df[col] = df[col].fillna(df[col].median())

    # Categorical Encoding
    encoder = LabelEncoder()
    categorical_cols = ['membership_category', 'feedback', 'complaint_status', 'region_category', 'gender', 'preferred_offer_types', 'internet_option','medium_of_operation']
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    # replaced with tenure
    df = df.drop(columns=['joining_date'], errors='ignore')

    return df