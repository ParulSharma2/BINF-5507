# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Impute Missing Values
def impute_missing_values(data, strategy='mean'):
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """
    # TODO: Fill missing values based on the specified strategy
    data = data.copy()  # avoid modifying the original DataFrame
    data.replace("", np.nan, inplace=True)  # Convert empty strings to NaN
    for col in data.columns:
        if data[col].isnull().sum() > 0:  # Check if the column has missing values
            if data[col].dtype in ['float64', 'int64']:  # For numerical columns
                if strategy == 'mean':  # Replace missing with the mean of the column
                    data[col].fillna(data[col].mean(), inplace=True)
                elif strategy == 'median':  # Replace missing with the median
                    data[col].fillna(data[col].median(), inplace=True)
                elif strategy == 'mode':  # Replace missing with the mode
                    data[col].fillna(data[col].mode()[0], inplace=True)
            else:  # For categorical columns, always use the mode (most frequent value)
                data[col].fillna(data[col].mode()[0], inplace=True)

    return data

# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    # TODO: Remove duplicate rows
    # Drop duplicate rows and return the cleaned DataFrame
    #It compares all columns and removes rows where every value in that 
    # row is the same as another row. It returns the cleaned dataset without duplicate entries.
    return data.drop_duplicates()

# 3. Normalize Numerical Data
def normalize_data(data,method='minmax'):
    """Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
    # TODO: Normalize numerical data using Min-Max or Standard scaling
    data = data.copy()
    numeric_cols = data.select_dtypes(include=['number']).columns  # Select numeric columns

    if method == 'minmax':
        scaler = MinMaxScaler()  # Scale data to [0, 1]
    elif method == 'standard':
        scaler = StandardScaler()  # Scale data to have mean 0 and standard deviation 1
    else:
        raise ValueError("Invalid method. Choose 'minmax' or 'standard'.")  # Raise an error if invalid method is passed

    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])  # Apply scaling to numeric columns
    return data

# 4. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    # TODO: Remove redundant features based on the correlation threshold (HINT: you can use the corr() method)
    # Make a copy to avoid modifying the original data
    data = data.copy()
    # Select only numeric columns for correlation analysis
    numeric_data = data.select_dtypes(include=['number'])

    # Compute correlation matrix
    corr_matrix = numeric_data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Identify columns to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop from original data
    return data.drop(columns=to_drop, errors='ignore')

# ---------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False, target_col='target'):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    # Make a copy to avoid modifying original data
    input_data = input_data.copy()
    input_data.dropna(inplace=True)

    # Separate target and features
    target = input_data[target_col]
    features = input_data.drop(columns=[target_col])

    # Ensure the target is categorical/discrete for classification
    if target.dtype not in ['int64', 'int32', 'object', 'bool']:
        try:
            target = target.round().astype(int)
        except:
            raise ValueError("Target column must be categorical for classification.")

    # One-hot encode categorical features
    for col in features.columns:
        if features[col].dtype == 'object':
            dummies = pd.get_dummies(features[col], prefix=col)
            features.drop(columns=[col], inplace=True)
            features = pd.concat([features, dummies], axis=1)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )

    # Apply scaling if requested
    if scale_data:
        X_train = normalize_data(X_train, method='standard')
        X_test = normalize_data(X_test, method='standard')

    # Fit logistic regression model
    model = LogisticRegression(random_state=42, max_iter=100, solver='liblinear')
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.4f}")
    if print_report:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))