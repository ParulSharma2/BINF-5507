# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import r2_score, accuracy_score, classification_report

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
# 4. Simple Model   
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

    # Train-Test Split
    stratify_option = target if target_col == 'num' else None
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=stratify_option, random_state=42
    )

    # Feature Scaling if Requested
    if scale_data:
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)

    # Model Initialization
    if target_col == 'num':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif target_col == 'chol':
        model = ElasticNet(random_state=42)
    else:
        raise ValueError("Unsupported target column. Use 'num' for classification or 'chol' for regression.")

    # Model Training & Prediction
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Evaluation
    if target_col == 'num':
        acc = accuracy_score(y_test, y_pred)
        print(f"\nClassification Accuracy: {acc:.4f}")
        if print_report:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
    elif target_col == 'chol':
        r2 = r2_score(y_test, y_pred)
        print(f"\nRegression R² Score: {r2:.4f}")

# Function 6: Plot ROC and Precision-Recall Curves
def plot_roc_pr(model, X_test, y_test, model_name='Model'):
    """
    Plots ROC and Precision-Recall Curves for a classification model.
    :param model: Trained classification model with predict_proba method
    :param X_test: Features of the test set
    :param y_test: True labels of the test set
    :param model_name: str, name of the model for plot titles
    """
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt

    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} - AUROC = {auc_score:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, label=f'{model_name} - AUPRC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.show()
