# Assignment 1: Data Preprocessing and Logistic Regression
## Project Overview
This project demonstrates the process of cleaning a messy dataset and applying logistic regression for binary classification. The goal is to highlight the importance of data preprocessing steps such as handling missing values, normalization, and feature selection in improving model performance.

## Dataset Description
- **Original Dataset**: `messy_data.csv`
- **Cleaned Dataset**: `clean_data.csv`
- **Rows**: 1,196
- **Original Columns**: 28 (with missing values and redundant features)
- **Final Columns After Cleaning**: 22

## How to Run
1. Clone or download this repository.
2. Make sure Python is installed on your system.
3. Install required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
4. Place the raw data file (`messy_data.csv`) inside the `Data/` folder.
5. Run the `main.ipy` file
6. You will get:
- A cleaned dataset saved as `clean_data.csv`
- Histograms showing data distribution before and after cleaning
- Accuracy score and classification report printed in the terminal
---
## Preprocessing Steps
All steps were implemented using the methods discussed in **Lecture 2 - Data Preprocessing and Feature Engineering**.
1. **Imputation**: Missing values were filled using the mean for numerical columns and the mode for categorical columns.
2. **Duplicate Removal**: Duplicate rows were removed to ensure data integrity using `drop_duplicates()`.
3. **Normalization**: Numerical features were scaled using Min-Max (to scale between 0 and 1 (default)) and  `StandardScaler` to make mean = 0 and standard deviation = 1 (optional) normalization.
4. **Redundancy Removal**: Highly correlated features (correlation > 0.9) confuse the model and increase computation, thus were dropped to reduce multicollinearity.
---
---
## Model: Logistic Regression
The `simple_model()` function performs the following:
- Drops rows with missing values
- Separates features and target
- One-hot encodes categorical variables
- Splits the dataset into training and testing sets (80-20)
- Optionally scales the features
- Trains a logistic regression model
- Outputs model accuracy and classification report
---
---
## Data Exploration Summary
- The messy dataset had 4,588 missing values across 18 columns.
- Several columns had inconsistent data types.
- After preprocessing, all missing values were resolved, and the dataset was normalized and reduced to 22 features.
- Visualizations included histograms to understand distributions and relationships.
## Model Training Results
A logistic regression model was trained on the cleaned dataset with the following performance:
- **Accuracy**: 77.92%
- **Precision (Class 1)**: 0.82
- **Recall (Class 1)**: 0.85
- **F1-Score (Class 1)**: 0.84
These results indicate strong performance in identifying the positive class.
---
## Reference
- Lecture Slides: *Data Preprocessing and Feature Engineering* (Instructor: Caryn Geady, May 2025)
- Python Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
---
This project helps you learn:
- Why each preprocessing step matters
- How real-world messy data is cleaned
- How to build a simple logistic regression model
---