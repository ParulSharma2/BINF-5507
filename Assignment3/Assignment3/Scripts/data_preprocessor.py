# Data Preprocessor — Clustering with Combined Evaluation and Visualization (Aligned with Class Labs)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------------
# Data Generation Functions
# -----------------------------

def generate_moons(n_samples=300, noise=0.05):
    """
    Generates a standardized two-moons dataset, commonly used to evaluate clustering algorithms on non-spherical data.
    """
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return StandardScaler().fit_transform(X)

def generate_blobs(n_samples=300, centers=3, cluster_std=[1.0, 2.5, 0.5]):
    """
    Generates a standardized blob dataset with varying cluster densities.
    Used to test clustering algorithms' performance on spherical clusters.
    """
    X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=42)
    return StandardScaler().fit_transform(X)

def generate_circles(n_samples=300, noise=0.05):
    """
    Generates standardized concentric circles dataset.
    Useful for testing algorithms' ability to handle non-linearly separable data.
    """
    X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    return StandardScaler().fit_transform(X)

# -----------------------------
# Clustering Algorithm Functions
# -----------------------------

def apply_dbscan(X, eps=0.3, min_samples=5):
    """
    Applies Density-Based Spatial Clustering (DBSCAN) and returns cluster labels.
    Suitable for discovering clusters of arbitrary shape and handling noise.
    """
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

def apply_kmeans(X, n_clusters=3):
    """
    Applies k-Means clustering to partition data into 'n_clusters' spherical clusters.
    Returns cluster labels.
    """
    return KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)

def apply_hierarchical(X, n_clusters=3):
    """
    Applies Agglomerative Hierarchical Clustering.
    Clusters data into 'n_clusters' based on linkage criteria.
    """
    return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)

# -----------------------------
# Combined Cluster and Evaluation Plotting Function
# -----------------------------

def plot_clusters_with_silhouette(X, labels, title):
    """
    Plots clustering results alongside silhouette plot for comprehensive evaluation.
    Left: Scatter plot of clustered data.
    Right: Silhouette plot showing individual silhouette scores per cluster.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Cluster Scatter Plot
    unique_labels = set(labels)
    for label in unique_labels:
        ax1.scatter(X[labels == label, 0], X[labels == label, 1], label=f'Cluster {label}')
    ax1.set_title(f'{title} - Cluster Plot')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()

    # Silhouette Plot
    if len(unique_labels) > 1 and not (len(unique_labels) == 2 and -1 in unique_labels):
        silhouette_vals = silhouette_samples(X, labels)
        y_lower = 10
        for i, label in enumerate(unique_labels):
            ith_silhouette_vals = silhouette_vals[labels == label]
            ith_silhouette_vals.sort()
            y_upper = y_lower + len(ith_silhouette_vals)
            ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_silhouette_vals)
            ax2.text(-0.05, y_lower + 0.5 * len(ith_silhouette_vals), str(label))
            y_lower = y_upper + 10
        sil_score = silhouette_score(X, labels)
        ax2.axvline(x=sil_score, color='red', linestyle='--')
        ax2.set_title(f'{title} - Silhouette Plot\nAvg Silhouette Score = {sil_score:.2f}')
        ax2.set_xlabel('Silhouette Coefficient')
        ax2.set_ylabel('Cluster')
    else:
        ax2.text(0.5, 0.5, 'Silhouette Not Applicable', horizontalalignment='center', verticalalignment='center')
        ax2.set_title(f'{title} - Silhouette Plot')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Dendrogram Plotting Function
# -----------------------------

def plot_dendrogram(X, method='ward', title='Dendrogram'):
    """
    Plots dendrogram for Hierarchical Clustering to visualize hierarchical merging.
    Helps interpret optimal cluster cut-off points based on distances.
    """
    plt.figure(figsize=(8, 5))
    linked = linkage(X, method=method)
    dendrogram(linked)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Linkage Distance')
    plt.show()
