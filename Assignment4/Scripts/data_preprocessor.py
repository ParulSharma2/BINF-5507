# Import required libraries
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.model_selection import train_test_split

# Function to load data, drop rows with missing survival time or event,
# and fill missing numerical values with their median
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Remove rows with missing 'Length FU' (survival time) or 'Status' (event)
    df = df.dropna(subset=['Length FU', 'Status'])
    # Identify numeric columns, excluding the survival time column
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'Length FU']
    # Impute missing values in numeric columns with median values
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df

# Function to create an age group column for stratification
# Patients < 60 are labeled '<60', others '≥60'
def create_age_group(df):
    df['AgeGroup'] = np.where(df['Age'] < 60, '<60', '≥60')
    return df

# Function to simplify clinical stage into broader categories for analysis
# Early: Stages I, II, III; Advanced: IVA, IVB; Other: anything else or missing
def simplify_stage(df):
    stage_map = {
        'I': 'Early',
        'II': 'Early',
        'III': 'Early',
        'IVA': 'Advanced',
        'IVB': 'Advanced'
    }
    df['Stage_Simplified'] = df['Stage'].map(stage_map)
    df['Stage_Simplified'] = df['Stage_Simplified'].fillna('Other')
    return df

# Function to prepare data for Kaplan-Meier analysis by grouping based on a categorical column
def get_km_groups(df, group_col):
    groups = df[group_col].dropna().unique()
    km_data = []
    for group in groups:
        # Select rows for the current group with valid survival time and event status
        temp_df = df[(df[group_col] == group) & df['Length FU'].notna() & df['Status'].notna()]
        if not temp_df.empty:
            km_data.append({
                'group': group,
                'time': temp_df['Length FU'],
                'event': temp_df['Status'].apply(lambda x: 1 if x == 'Dead' else 0)
            })
    return km_data

# Function to prepare data for Cox regression and Random Survival Forest
# Performs one-hot encoding for categorical variables
def prepare_cox_data(df, covariates):
    X = df[covariates].copy()
    # One-hot encode categorical variables, drop the first level to avoid dummy variable trap
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    # Create structured survival array for scikit-survival models
    y = np.array([(1 if status == 'Dead' else 0, time) 
                  for status, time in zip(df['Status'], df['Length FU'])], 
                  dtype=[('event', 'bool'), ('time', 'float')])
    return X, y

# Function to split data into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Function to plot Kaplan-Meier survival curves for each group
def plot_km(km_data, title="Kaplan-Meier Curve"):
    import matplotlib.pyplot as plt
    kmf = KaplanMeierFitter()
    plt.figure()
    for group_data in km_data:
        if not group_data['time'].empty:
            kmf.fit(group_data['time'], group_data['event'], label=str(group_data['group']))
            kmf.plot_survival_function()
    plt.title(title)
    plt.xlabel('Time (Months)')
    plt.ylabel('Survival Probability')
    plt.grid(True)
    plt.show()

# Function to perform the log-rank test between two groups' survival distributions
def perform_logrank_test(km_data):
    if len(km_data) != 2:
        print("Log-rank test only for 2 groups (found {})".format(len(km_data)))
        return None
    group1, group2 = km_data
    result = logrank_test(group1['time'], group2['time'], event_observed_A=group1['event'], event_observed_B=group2['event'])
    return result.p_value
