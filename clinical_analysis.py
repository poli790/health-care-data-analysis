"""
Clinical Analysis Module
This module contains functions for healthcare-specific analysis and metrics.
"""

import pandas as pd
import numpy as np
from scipy import stats

def calculate_readmission_rate(df, readmission_col='is_readmission'):
    """
    Calculate overall readmission rate
    
    Parameters:
    df (pd.DataFrame): Patient dataframe with readmission flag
    readmission_col (str): Column name for readmission flag
    
    Returns:
    float: Readmission rate as percentage
    """
    readmission_rate = (df[readmission_col].sum() / len(df)) * 100
    return readmission_rate

def calculate_mortality_rate(df, outcome_col='outcome'):
    """
    Calculate mortality rate
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    outcome_col (str): Column name for patient outcome
    
    Returns:
    float: Mortality rate as percentage
    """
    if outcome_col not in df.columns:
        return None
    
    mortality_rate = (df[outcome_col] == 'deceased').sum() / len(df) * 100
    return mortality_rate

def calculate_bed_occupancy(df, total_beds, date_col='admission_date'):
    """
    Calculate daily bed occupancy rate
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    total_beds (int): Total number of available beds
    date_col (str): Column name for date
    
    Returns:
    pd.DataFrame: Daily occupancy rates
    """
    daily_admissions = df.groupby(date_col).size().reset_index(name='patients')
    daily_admissions['occupancy_rate'] = (daily_admissions['patients'] / total_beds) * 100
    
    return daily_admissions

def stratify_risk_score(df, los_col='length_of_stay', age_col='age', 
                       readmission_col='is_readmission'):
    """
    Calculate risk score for patients
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    los_col (str): Column name for length of stay
    age_col (str): Column name for age
    readmission_col (str): Column name for readmission history
    
    Returns:
    pd.DataFrame: Dataframe with risk_score and risk_category columns
    """
    df_copy = df.copy()
    
    # Calculate risk score (0-100)
    risk_score = 0
    
    # Age component (0-30 points)
    if age_col in df_copy.columns:
        age_normalized = (df_copy[age_col] / 120) * 30
        risk_score += age_normalized
    
    # Length of stay component (0-30 points)
    if los_col in df_copy.columns:
        los_normalized = np.minimum(df_copy[los_col] / 30, 1) * 30
        risk_score += los_normalized
    
    # Readmission history (0-40 points)
    if readmission_col in df_copy.columns:
        risk_score += df_copy[readmission_col].astype(int) * 40
    
    df_copy['risk_score'] = risk_score
    
    # Categorize risk
    def categorize_risk(score):
        if score < 30:
            return 'Low Risk'
        elif score < 60:
            return 'Medium Risk'
        else:
            return 'High Risk'
    
    df_copy['risk_category'] = df_copy['risk_score'].apply(categorize_risk)
    
    return df_copy

def analyze_treatment_outcomes(df, treatment_col='treatment_type', outcome_col='outcome'):
    """
    Analyze outcomes by treatment type
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    treatment_col (str): Column name for treatment type
    outcome_col (str): Column name for outcome
    
    Returns:
    pd.DataFrame: Treatment outcome statistics
    """
    treatment_outcomes = df.groupby([treatment_col, outcome_col]).size().reset_index(name='count')
    
    # Calculate success rate for each treatment
    total_by_treatment = df.groupby(treatment_col).size().reset_index(name='total')
    
    treatment_outcomes = treatment_outcomes.merge(total_by_treatment, on=treatment_col)
    treatment_outcomes['percentage'] = (treatment_outcomes['count'] / treatment_outcomes['total']) * 100
    
    return treatment_outcomes

def calculate_average_los_by_condition(df, condition_col='diagnosis', los_col='length_of_stay'):
    """
    Calculate average length of stay by medical condition
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    condition_col (str): Column name for diagnosis/condition
    los_col (str): Column name for length of stay
    
    Returns:
    pd.DataFrame: Average LOS by condition
    """
    los_stats = df.groupby(condition_col)[los_col].agg([
        ('avg_los', 'mean'),
        ('median_los', 'median'),
        ('min_los', 'min'),
        ('max_los', 'max'),
        ('patient_count', 'count')
    ]).reset_index()
    
    los_stats = los_stats.sort_values('avg_los', ascending=False)
    
    return los_stats

def identify_high_cost_patients(df, cost_col='total_cost', percentile=90):
    """
    Identify high-cost patients
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    cost_col (str): Column name for total cost
    percentile (int): Percentile threshold (default: 90)
    
    Returns:
    pd.DataFrame: High-cost patient subset
    """
    if cost_col not in df.columns:
        return None
    
    cost_threshold = df[cost_col].quantile(percentile / 100)
    high_cost_patients = df[df[cost_col] >= cost_threshold].copy()
    high_cost_patients['cost_category'] = 'High Cost'
    
    return high_cost_patients

def analyze_seasonal_patterns(df, date_col='admission_date'):
    """
    Analyze seasonal admission patterns
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    date_col (str): Column name for admission date
    
    Returns:
    dict: Seasonal statistics by month, day of week, etc.
    """
    df_copy = df.copy()
    df_copy['month'] = df_copy[date_col].dt.month
    df_copy['day_of_week'] = df_copy[date_col].dt.dayofweek
    df_copy['quarter'] = df_copy[date_col].dt.quarter
    
    monthly_admissions = df_copy.groupby('month').size().reset_index(name='admissions')
    daily_admissions = df_copy.groupby('day_of_week').size().reset_index(name='admissions')
    quarterly_admissions = df_copy.groupby('quarter').size().reset_index(name='admissions')
    
    return {
        'monthly': monthly_admissions,
        'daily': daily_admissions,
        'quarterly': quarterly_admissions
    }

def calculate_comorbidity_index(df, condition_cols):
    """
    Calculate comorbidity index (simplified Charlson-like score)
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    condition_cols (list): List of binary columns for conditions
    
    Returns:
    pd.DataFrame: Dataframe with comorbidity_index column
    """
    df_copy = df.copy()
    
    # Sum the number of conditions
    df_copy['comorbidity_index'] = df_copy[condition_cols].sum(axis=1)
    
    return df_copy

def predict_readmission_risk(df, features=['age', 'length_of_stay', 'comorbidity_index']):
    """
    Simple rule-based readmission risk prediction
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    features (list): List of feature columns to use
    
    Returns:
    pd.DataFrame: Dataframe with predicted_readmission_risk column
    """
    df_copy = df.copy()
    
    # Simple scoring system
    risk_points = 0
    
    if 'age' in features and 'age' in df_copy.columns:
        risk_points += (df_copy['age'] > 65).astype(int) * 2
    
    if 'length_of_stay' in features and 'length_of_stay' in df_copy.columns:
        risk_points += (df_copy['length_of_stay'] > 7).astype(int) * 2
    
    if 'comorbidity_index' in features and 'comorbidity_index' in df_copy.columns:
        risk_points += (df_copy['comorbidity_index'] >= 2).astype(int) * 3
    
    # Convert to probability
    df_copy['predicted_readmission_risk'] = np.minimum(risk_points / 7 * 100, 100)
    
    return df_copy

def calculate_department_efficiency(df, department_col='department', 
                                   los_col='length_of_stay', cost_col='total_cost'):
    """
    Calculate efficiency metrics by department
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    department_col (str): Column name for department
    los_col (str): Column name for length of stay
    cost_col (str): Column name for cost
    
    Returns:
    pd.DataFrame: Department efficiency metrics
    """
    efficiency_metrics = df.groupby(department_col).agg({
        'patient_id': 'count',
        los_col: 'mean',
        cost_col: 'mean' if cost_col in df.columns else 'count'
    }).reset_index()
    
    efficiency_metrics.columns = [department_col, 'patient_count', 'avg_los', 'avg_cost']
    
    # Calculate efficiency score (lower is better)
    efficiency_metrics['efficiency_score'] = (
        efficiency_metrics['avg_los'] * efficiency_metrics['avg_cost']
    ) / efficiency_metrics['patient_count']
    
    return efficiency_metrics

if __name__ == "__main__":
    print("Clinical Analysis Module loaded successfully!")
    print("\nAvailable functions:")
    print("- calculate_readmission_rate()")
    print("- calculate_mortality_rate()")
    print("- calculate_bed_occupancy()")
    print("- stratify_risk_score()")
    print("- analyze_treatment_outcomes()")
    print("- calculate_average_los_by_condition()")
    print("- identify_high_cost_patients()")
    print("- analyze_seasonal_patterns()")
    print("- calculate_comorbidity_index()")
    print("- predict_readmission_risk()")
    print("- calculate_department_efficiency()")
