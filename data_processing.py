"""
Healthcare Data Processing Module
This module contains functions for cleaning and preprocessing healthcare data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_patient_data(filepath):
    """
    Load patient data from CSV file
    
    Parameters:
    filepath (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(filepath)
    return df

def clean_patient_data(df):
    """
    Clean and preprocess patient data
    
    Parameters:
    df (pd.DataFrame): Raw patient dataframe
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    # Make a copy
    df_clean = df.copy()
    
    # Convert date columns to datetime
    date_columns = ['admission_date', 'discharge_date', 'birth_date']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=['patient_id', 'admission_date'], keep='first')
    
    # Handle missing values
    df_clean = df_clean.dropna(subset=['patient_id', 'admission_date'])
    
    # Remove invalid ages (negative or > 120)
    if 'age' in df_clean.columns:
        df_clean = df_clean[(df_clean['age'] >= 0) & (df_clean['age'] <= 120)]
    
    return df_clean

def calculate_patient_age(df, birth_date_col='birth_date', reference_date=None):
    """
    Calculate patient age from birth date
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    birth_date_col (str): Column name for birth date
    reference_date (datetime): Date to calculate age from (default: today)
    
    Returns:
    pd.DataFrame: Dataframe with age column added
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    df['age'] = (reference_date - df[birth_date_col]).dt.days // 365
    
    return df

def calculate_length_of_stay(df, admission_col='admission_date', discharge_col='discharge_date'):
    """
    Calculate length of stay in days
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    admission_col (str): Column name for admission date
    discharge_col (str): Column name for discharge date
    
    Returns:
    pd.DataFrame: Dataframe with length_of_stay column added
    """
    df['length_of_stay'] = (df[discharge_col] - df[admission_col]).dt.days
    
    # Handle negative or zero values
    df.loc[df['length_of_stay'] < 0, 'length_of_stay'] = np.nan
    df.loc[df['length_of_stay'] == 0, 'length_of_stay'] = 1  # Minimum 1 day
    
    return df

def categorize_age_groups(df, age_col='age'):
    """
    Categorize patients into age groups
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    age_col (str): Column name for age
    
    Returns:
    pd.DataFrame: Dataframe with age_group column added
    """
    bins = [0, 18, 35, 50, 65, 80, 120]
    labels = ['0-17', '18-34', '35-49', '50-64', '65-79', '80+']
    
    df['age_group'] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
    
    return df

def identify_readmissions(df, patient_id_col='patient_id', discharge_col='discharge_date', 
                          admission_col='admission_date', days=30):
    """
    Identify readmissions within specified days
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    patient_id_col (str): Column name for patient ID
    discharge_col (str): Column name for discharge date
    admission_col (str): Column name for admission date
    days (int): Number of days to check for readmission
    
    Returns:
    pd.DataFrame: Dataframe with readmission flag column added
    """
    # Sort by patient and admission date
    df_sorted = df.sort_values([patient_id_col, admission_col])
    
    # Calculate days since last discharge for each patient
    df_sorted['previous_discharge'] = df_sorted.groupby(patient_id_col)[discharge_col].shift(1)
    df_sorted['days_since_discharge'] = (df_sorted[admission_col] - df_sorted['previous_discharge']).dt.days
    
    # Flag readmissions
    df_sorted['is_readmission'] = (df_sorted['days_since_discharge'] <= days) & (df_sorted['days_since_discharge'] > 0)
    df_sorted['is_readmission'] = df_sorted['is_readmission'].fillna(False)
    
    return df_sorted

def anonymize_patient_data(df, patient_id_col='patient_id'):
    """
    Anonymize patient IDs for privacy
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    patient_id_col (str): Column name for patient ID
    
    Returns:
    pd.DataFrame: Dataframe with anonymized patient IDs
    """
    # Create mapping of original IDs to anonymized IDs
    unique_ids = df[patient_id_col].unique()
    id_mapping = {old_id: f'PAT_{i:06d}' for i, old_id in enumerate(unique_ids, 1)}
    
    df['anonymized_id'] = df[patient_id_col].map(id_mapping)
    
    return df

def aggregate_by_department(df, department_col='department', metric_cols=['length_of_stay']):
    """
    Aggregate metrics by department
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    department_col (str): Column name for department
    metric_cols (list): List of columns to aggregate
    
    Returns:
    pd.DataFrame: Aggregated dataframe
    """
    agg_dict = {col: ['mean', 'median', 'count', 'std'] for col in metric_cols}
    
    dept_stats = df.groupby(department_col).agg(agg_dict).reset_index()
    
    # Flatten column names
    dept_stats.columns = ['_'.join(col).strip('_') for col in dept_stats.columns.values]
    
    return dept_stats

if __name__ == "__main__":
    print("Healthcare Data Processing Module loaded successfully!")
    print("\nAvailable functions:")
    print("- load_patient_data()")
    print("- clean_patient_data()")
    print("- calculate_patient_age()")
    print("- calculate_length_of_stay()")
    print("- categorize_age_groups()")
    print("- identify_readmissions()")
    print("- anonymize_patient_data()")
    print("- aggregate_by_department()")
