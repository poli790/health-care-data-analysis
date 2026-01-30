"""
Healthcare Visualization Module
This module contains functions for creating medical data visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_patient_demographics(df, age_col='age', gender_col='gender'):
    """
    Plot patient demographic distributions
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    age_col (str): Column name for age
    gender_col (str): Column name for gender
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Age distribution
    axes[0].hist(df[age_col], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Age', fontsize=12)
    axes[0].set_ylabel('Number of Patients', fontsize=12)
    axes[0].set_title('Patient Age Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Gender distribution
    if gender_col in df.columns:
        gender_counts = df[gender_col].value_counts()
        axes[1].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
                   colors=['#e74c3c', '#3498db', '#95a5a6'], startangle=90)
        axes[1].set_title('Patient Gender Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_admission_trends(df, date_col='admission_date', title='Hospital Admissions Over Time'):
    """
    Plot admission trends over time
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    date_col (str): Column name for admission date
    title (str): Plot title
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Group by date and count
    daily_admissions = df.groupby(df[date_col].dt.date).size().reset_index(name='admissions')
    daily_admissions.columns = ['date', 'admissions']
    
    ax.plot(daily_admissions['date'], daily_admissions['admissions'], 
           linewidth=2, color='#2ecc71', marker='o', markersize=4)
    
    # Add 7-day moving average
    daily_admissions['ma_7'] = daily_admissions['admissions'].rolling(window=7).mean()
    ax.plot(daily_admissions['date'], daily_admissions['ma_7'], 
           linewidth=2, color='#e74c3c', linestyle='--', label='7-Day Moving Average')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Admissions', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def plot_length_of_stay_distribution(df, los_col='length_of_stay', department_col='department'):
    """
    Plot length of stay distribution by department
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    los_col (str): Column name for length of stay
    department_col (str): Column name for department
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    departments = df[department_col].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(departments)))
    
    for i, dept in enumerate(departments):
        dept_data = df[df[department_col] == dept][los_col]
        ax.hist(dept_data, bins=20, alpha=0.6, label=dept, color=colors[i], edgecolor='black')
    
    ax.set_xlabel('Length of Stay (days)', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title('Length of Stay Distribution by Department', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_readmission_analysis(df, readmission_col='is_readmission', risk_col='risk_category'):
    """
    Plot readmission analysis
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    readmission_col (str): Column name for readmission flag
    risk_col (str): Column name for risk category
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Readmission rate pie chart
    readmission_counts = df[readmission_col].value_counts()
    axes[0].pie(readmission_counts, labels=['No Readmission', 'Readmission'],
               autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[0].set_title('Overall Readmission Rate', fontsize=14, fontweight='bold')
    
    # Readmission by risk category
    if risk_col in df.columns:
        risk_readmission = pd.crosstab(df[risk_col], df[readmission_col], normalize='index') * 100
        risk_readmission.plot(kind='bar', stacked=True, ax=axes[1], 
                             color=['#2ecc71', '#e74c3c'], alpha=0.8)
        axes[1].set_xlabel('Risk Category', fontsize=12)
        axes[1].set_ylabel('Percentage', fontsize=12)
        axes[1].set_title('Readmission Rate by Risk Category', fontsize=14, fontweight='bold')
        axes[1].legend(['No Readmission', 'Readmission'])
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_department_performance(df, department_col='department', 
                               los_col='length_of_stay', cost_col='total_cost'):
    """
    Plot department performance metrics
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    department_col (str): Column name for department
    los_col (str): Column name for length of stay
    cost_col (str): Column name for cost
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Patient volume by department
    dept_counts = df[department_col].value_counts().sort_values()
    axes[0, 0].barh(dept_counts.index, dept_counts.values, color='#3498db', alpha=0.8)
    axes[0, 0].set_xlabel('Number of Patients', fontsize=11)
    axes[0, 0].set_title('Patient Volume by Department', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Average LOS by department
    avg_los = df.groupby(department_col)[los_col].mean().sort_values()
    axes[0, 1].barh(avg_los.index, avg_los.values, color='#2ecc71', alpha=0.8)
    axes[0, 1].set_xlabel('Average Length of Stay (days)', fontsize=11)
    axes[0, 1].set_title('Average LOS by Department', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Average cost by department (if available)
    if cost_col in df.columns:
        avg_cost = df.groupby(department_col)[cost_col].mean().sort_values()
        axes[1, 0].barh(avg_cost.index, avg_cost.values, color='#e74c3c', alpha=0.8)
        axes[1, 0].set_xlabel('Average Cost ($)', fontsize=11)
        axes[1, 0].set_title('Average Cost by Department', fontsize=12, fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
    
    # LOS box plot by department
    departments = df[department_col].unique()
    los_data = [df[df[department_col] == dept][los_col].values for dept in departments]
    axes[1, 1].boxplot(los_data, labels=departments, patch_artist=True)
    axes[1, 1].set_ylabel('Length of Stay (days)', fontsize=11)
    axes[1, 1].set_title('LOS Distribution by Department', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticklabels(departments, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_seasonal_patterns(seasonal_data):
    """
    Plot seasonal admission patterns
    
    Parameters:
    seasonal_data (dict): Dictionary with 'monthly', 'daily', 'quarterly' dataframes
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Monthly pattern
    monthly = seasonal_data['monthly']
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes[0].bar(monthly['month'], monthly['admissions'], color='#3498db', alpha=0.8)
    axes[0].set_xlabel('Month', fontsize=11)
    axes[0].set_ylabel('Admissions', fontsize=11)
    axes[0].set_title('Monthly Admission Pattern', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(month_names, rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Day of week pattern
    daily = seasonal_data['daily']
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1].bar(daily['day_of_week'], daily['admissions'], color='#2ecc71', alpha=0.8)
    axes[1].set_xlabel('Day of Week', fontsize=11)
    axes[1].set_ylabel('Admissions', fontsize=11)
    axes[1].set_title('Weekly Admission Pattern', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(day_names, rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Quarterly pattern
    quarterly = seasonal_data['quarterly']
    axes[2].bar(['Q1', 'Q2', 'Q3', 'Q4'], quarterly['admissions'], color='#e74c3c', alpha=0.8)
    axes[2].set_xlabel('Quarter', fontsize=11)
    axes[2].set_ylabel('Admissions', fontsize=11)
    axes[2].set_title('Quarterly Admission Pattern', fontsize=12, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_risk_stratification(df, risk_col='risk_category', risk_score_col='risk_score'):
    """
    Plot risk stratification analysis
    
    Parameters:
    df (pd.DataFrame): Patient dataframe
    risk_col (str): Column name for risk category
    risk_score_col (str): Column name for risk score
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Risk category distribution
    risk_counts = df[risk_col].value_counts()
    colors = {'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'}
    bar_colors = [colors.get(cat, '#95a5a6') for cat in risk_counts.index]
    
    axes[0].bar(risk_counts.index, risk_counts.values, color=bar_colors, alpha=0.8)
    axes[0].set_xlabel('Risk Category', fontsize=12)
    axes[0].set_ylabel('Number of Patients', fontsize=12)
    axes[0].set_title('Patient Risk Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Risk score distribution
    axes[1].hist(df[risk_score_col], bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1].axvline(30, color='#2ecc71', linestyle='--', linewidth=2, label='Low/Medium threshold')
    axes[1].axvline(60, color='#e74c3c', linestyle='--', linewidth=2, label='Medium/High threshold')
    axes[1].set_xlabel('Risk Score', fontsize=12)
    axes[1].set_ylabel('Number of Patients', fontsize=12)
    axes[1].set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_kpi_dashboard(metrics_dict):
    """
    Create a KPI dashboard with key healthcare metrics
    
    Parameters:
    metrics_dict (dict): Dictionary containing key metrics
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    fig = plt.figure(figsize=(14, 8))
    
    # Create grid for KPI boxes
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    kpi_data = [
        ('Total Patients', metrics_dict.get('total_patients', 0), '#3498db'),
        ('Avg Length of Stay', f"{metrics_dict.get('avg_los', 0):.1f} days", '#2ecc71'),
        ('Readmission Rate', f"{metrics_dict.get('readmission_rate', 0):.1f}%", '#e74c3c'),
        ('Bed Occupancy', f"{metrics_dict.get('bed_occupancy', 0):.1f}%", '#f39c12'),
        ('High Risk Patients', metrics_dict.get('high_risk_count', 0), '#9b59b6'),
        ('Avg Cost per Patient', f"${metrics_dict.get('avg_cost', 0):,.0f}", '#1abc9c'),
    ]
    
    for idx, (title, value, color) in enumerate(kpi_data):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        ax.text(0.5, 0.6, str(value), ha='center', va='center', 
               fontsize=28, fontweight='bold', color=color)
        ax.text(0.5, 0.3, title, ha='center', va='center', 
               fontsize=14, color='#34495e')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add background box
        ax.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8, 
                                  fill=True, facecolor=color, alpha=0.1, 
                                  edgecolor=color, linewidth=2))
    
    fig.suptitle('Healthcare KPI Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    return fig

if __name__ == "__main__":
    print("Healthcare Visualization Module loaded successfully!")
    print("\nAvailable functions:")
    print("- plot_patient_demographics()")
    print("- plot_admission_trends()")
    print("- plot_length_of_stay_distribution()")
    print("- plot_readmission_analysis()")
    print("- plot_department_performance()")
    print("- plot_seasonal_patterns()")
    print("- plot_risk_stratification()")
    print("- create_kpi_dashboard()")
