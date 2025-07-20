# Survival Analysis of the RADCURE Clinical Dataset — Assignment 4

## Project Overview

This project analyzes survival outcomes in head and neck cancer patients using the RADCURE clinical dataset. The objective is to identify key clinical factors associated with patient survival through statistical and machine learning-based survival analysis methods.

The analysis uses Kaplan-Meier curves, Cox Proportional Hazards regression, and Random Survival Forest models to explore univariate and multivariable predictors of survival.

## Code Description

### data_preprocessor.py

* Cleans and preprocesses the dataset.
* Handles missing data and creates derived variables such as AgeGroup and Stage categories.
* Prepares data for Kaplan-Meier analysis, Cox regression, and Random Survival Forest modeling.
* Provides functions for Kaplan-Meier curve plotting and log-rank testing.

### main_analysis.ipynb

* Loads and cleans the RADCURE dataset.
* Performs Kaplan-Meier survival analysis on AgeGroup, Stage, and HPV Status.
* Fits the Cox Proportional Hazards model and evaluates its performance using the concordance index.
* Trains the Random Survival Forest model and assesses variable importance.
* Summarizes key findings and survival patterns observed in the dataset.

## Key Findings

* **Age**, **Stage**, and **Treatment Modality** were consistently strong predictors of survival across all methods applied.
* Kaplan-Meier analysis confirmed significant survival differences based on AgeGroup and HPV Status.
* The Cox model showed good predictive performance with a concordance index of 0.66, making it suitable for identifying clinical risk factors.
* The Random Survival Forest model supported the importance of the same predictors but showed slightly lower predictive accuracy.
* **HPV Status** showed a strong association with better survival in univariate analysis but was excluded from multivariable models due to nearly 49% missing data.

## Clinical Interpretation

Older patients, those with advanced-stage cancer, and those receiving single-modality treatments were associated with poorer survival outcomes.
HPV-positive patients showed better survival, consistent with existing clinical evidence suggesting that HPV-positive head and neck cancers have distinct biological behavior and respond better to treatment.

## How to Run the Code

1. Place the dataset file in the `Data/` directory.
2. Ensure required Python packages are installed: pandas, lifelines, scikit-survival, matplotlib, scikit-learn.
3. Run the `main_analysis.ipynb` notebook step by step.
4. Review the plots and outputs for analysis results and interpretation.

## Python Libraries

- pandas — for data manipulation and handling missing values
- numpy — for numerical operations and array handling
- lifelines — for Kaplan-Meier estimation and log-rank testing
- scikit-survival — for Cox Proportional Hazards modeling and Random Survival Forest analysis
- matplotlib — for data visualization and plotting survival curves
- scikit-learn — for model evaluation, splitting data, and permutation importance

## Dataset

RADCURE Clinical Dataset: RADCURE_Clinical_v04_20241219.csv

## Reference

Kuhn, J. P., Schmid, W., Körner, S., Bochen, F., Wemmert, S., Rimbach, H., Smola, S., Radosa, J. C., Wagner, M., Morris, L. G., Bozzato, V., Bozzato, A., Schick, B., & Linxweiler, M. (2021). HPV status as prognostic biomarker in Head and Neck Cancer—Which method fits the best for outcome prediction? Cancers, 13(18), 4730. https://doi.org/10.3390/cancers13184730

**Lecture Notes:** Survival Modeling (BINF 5507 - Caryn Geady, May 2025).

