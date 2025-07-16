# Assignment 2 — Data Preprocessing, Regression, and Classification Analysis

---

## Project Overview:

This project applies supervised machine learning techniques to the Heart Disease UCI dataset, covering end-to-end data preprocessing, regression analysis, classification model development, and evaluation. The workflow emphasizes data cleaning, handling missing values, feature engineering, model selection, and evaluation for meaningful clinical predictions.

The project involves:
- Preprocessing clinical heart disease data for modeling.
- Developing ElasticNet regression models for predicting cholesterol levels.
- Building and evaluating classification models (Logistic Regression, k-NN) for predicting heart disease presence.
- Tuning model hyperparameters and visualizing evaluation metrics.
- Drawing insights through model performance analysis.

---

## Data Cleaning & Preprocessing for Heart Disease Dataset:

Preprocessing followed methods discussed in **Lecture 2 - Data Preprocessing and Feature Engineering**, targeting a clean and ready dataset for analysis.

- **Imputation:** Filled missing numerical values with column means.
- **Duplicate Removal:** Removed duplicate records using `drop_duplicates()`.
- **Normalization:** Scaled numerical features using Min-Max Scaling and StandardScaler.
- **Redundancy Removal:** Dropped features with correlation > 0.9.
- **One-Hot Encoding:** Encoded categorical variables (e.g., sex, chest pain type) using one-hot encoding.

---

## Logistic Regression Model and Evaluation:

Implemented **Logistic Regression** to classify the presence of heart disease.

- Data preprocessing included feature-target split, encoding, train-test split, and optional normalization.
- Trained using `liblinear` solver.
- Evaluation with accuracy score and classification report.

**Model Training Results (Heart Disease Data):**
- **Accuracy:** 86.80%
- **Precision (Class 1):** 0.88
- **Recall (Class 1):** 0.94
- **F1-Score (Class 1):** 0.91

---

## ElasticNet Regression Analysis:

Applied ElasticNet for predicting cholesterol levels.

- Hyperparameter tuning for `alpha` (regularization strength) and `l1_ratio` (L1-L2 balance).
- Evaluation with **R² Score** and **RMSE**.
- Heatmaps visualized performance across hyperparameters.
- Best configuration: **alpha = 0.01, l1_ratio = 0.1**.

---

## Classification Analysis:

### Logistic Regression:
- Tested solvers (`liblinear`, `saga`) with L1 and L2 penalties.
- Metrics prioritized: **Accuracy** and **F1 Score**.
- Best performance with **L2 penalty and liblinear solver**.

### k-Nearest Neighbors (k-NN):
- Evaluated with `n_neighbors` values of 1, 5, and 10.
- Observed bias-variance trade-off:
  - **k = 1:** High variance.
  - **k = 5 or 10:** Balanced performance.
- Best results with **k = 5 or 10**.

---

## Model Evaluation Metrics and Visualization:
- **Regression:** R² Score, RMSE.
- **Classification:** Accuracy, F1 Score, AUROC, AUPRC.
- ROC and Precision-Recall curves plotted for best models.

---

## Data Exploration Summary for Heart Disease Dataset:

- **Records:** 1,196 with 14 clinical/demographic features (e.g., age, cholesterol, blood pressure).
- Identified missing values in cholesterol and resting blood pressure.
- Numerical variables normalized for consistent scaling.
- Highly correlated features removed after correlation analysis.
- Categorical variables encoded via one-hot encoding.
- Post-cleaning:
  - Missing values imputed.
  - Duplicates and redundant features removed.
  - Normalized numerical features.
  - Final dataset saved as `clean_data.csv`.
- Visual exploration through histograms and heatmaps supported preprocessing decisions.

---

## Data Exploration and Model Insights:

- Preprocessing enhanced data quality and consistency.
- ElasticNet effectively handled multicollinearity in regression.
- Logistic Regression provided robust binary classification.
- k-NN reflected expected performance variations with `k` tuning.
- Model metrics and visualizations validated model choices for clinical predictions.

---

## This Project Helps You Learn:

- Essential preprocessing for machine learning.
- Handling missing data, duplicates, and redundant features.
- Normalization and categorical encoding techniques.
- ElasticNet Regression tuning for clinical datasets.
- Implementing Logistic Regression and k-NN with hyperparameter tuning.
- Model evaluation with R², RMSE, Accuracy, F1 Score, AUROC, AUPRC.
- Interpreting ROC and Precision-Recall curves for model performance.

---

## References:

- **Lecture Notes:** Data Preprocessing, Supervised Learning Models, and Evaluation (BINF 5507 - Caryn Geady, May 2025).
- **Python Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, warnings.
- **Dataset:** UCI Machine Learning Repository — Heart Disease Dataset.

---

## Author:
Parul Sharma — BINF 5507 Student — Humber IGS College
