# Gaussian Mixture Models for Probabilistic Clustering — Capstone Project

## Project Overview

This project implements and evaluates **Gaussian Mixture Models (GMM)** for clustering non-spherical, synthetic data, and compares its performance against **KMeans**, a traditional hard-clustering algorithm. The purpose is to demonstrate how GMM provides a more flexible and probabilistic framework for capturing overlapping and elliptical clusters—more representative of real-world data distributions.

The clustering models are applied to linearly transformed synthetic blobs to simulate complex cluster shapes. Both models are evaluated using internal validation metrics (Silhouette score, Homogeneity, Completeness, V-measure) and model selection criteria (AIC/BIC).

## Code Description

### `data_preprocessor.py`

* Generates synthetic 2D blob data and applies linear transformation to stretch them.
* Contains functions for:
  - KMeans and GMM clustering
  - Visualizing clusters and Gaussian ellipses
  - Calculating evaluation metrics (Homogeneity, Completeness, V-measure, Silhouette)
  - Plotting silhouette scores and AIC/BIC charts

### `main.py`

* Loads synthetic data using the preprocessing module.
* Runs clustering using both KMeans and GMM.
* Displays cluster visualizations with proper labels, ellipses, and legends.
* Outputs evaluation metrics for both models.
* Generates plots for silhouette score comparison and AIC/BIC-based model selection.

## Key Findings

* **GMM outperformed KMeans** in all evaluation metrics, including:
  - Higher **Homogeneity** and **Completeness** (0.980 vs 0.698)
  - Higher **V-measure** and **Silhouette score**
* **AIC/BIC scores** identified 4 as the optimal number of clusters, aligning with ground truth.
* GMM’s ability to model elliptical and overlapping clusters using full covariance matrices led to more accurate and interpretable clustering compared to KMeans.
* Visualizations confirmed KMeans struggled with distorted data, while GMM fit well using probabilistic boundaries.

## Interpretation

Gaussian Mixture Models demonstrated superior clustering performance in scenarios involving:
- **Non-spherical clusters**
- **Overlapping boundaries**
- **Soft clustering needs**, where data points can partially belong to more than one cluster

In real-world clinical applications (e.g., gene expression or imaging), such data complexities are common, making GMM a valuable tool for subgroup identification and patient stratification.

## How to Run the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gmm-clustering-capstone.git
   cd gmm-clustering-capstone
2.	Install required Python libraries: pip install numpy matplotlib scikit-learn
3.	Run the main script: python main.py
4.	The following will be generated:
    Cluster plots with ellipses
    Silhouette comparison plot
    AIC/BIC score line graphs
    Printed metric values for KMeans and GMM
# Python Libraries
    numpy — for generating and transforming synthetic data
    matplotlib — for plotting clusters, ellipses, and model evaluation graphs
    scikit-learn — for implementing KMeans, GMM, and calculating clustering metrics
# Dataset
Synthetic stretched blob data generated using make_blobs and linear transformation with np.dot().
# References
- Neal, R. M. (2007). Pattern Recognition and Machine Learning, by Christopher M. Bishop. Technometrics, 49. http://asq.org/technometrics/2007/08/pattern-recognition-and-machine-learning.pdf
- Gaussian Mixture Model Ellipsoids. (n.d.). Scikit-learn. https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html
- Lecture 5: Unsupervised Learning I — Clustering (BINF-5507, Winter 2025)

