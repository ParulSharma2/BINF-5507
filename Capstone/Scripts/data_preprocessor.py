# ===============================================
# Data Preprocessor for Clustering Capstone
# ===============================================

"""
This module contains helper functions for the Capstone Project:
- Generates synthetic stretched clustering data
- Evaluates clustering results (homogeneity, completeness, V-measure, silhouette)
- Plots clusters with KMeans circles or GMM ellipses
- Plots silhouette diagrams for visual cluster quality

Used with main.py to compare KMeans vs Gaussian Mixture Models (GMM).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.metrics import silhouette_score, silhouette_samples
from matplotlib import cm
import matplotlib as mpl
from matplotlib.colors import to_rgba


def generate_stretched_blob_data(n_samples=400, n_features=2, centers=4,
                                  cluster_std=0.6, blob_seed=0, stretch_seed=13):
    """
    Generate synthetic Gaussian blob data and apply a linear transformation to stretch it.

    Parameters:
        n_samples (int): Number of data points
        n_features (int): Number of features (typically 2 for visualization)
        centers (int): Number of true clusters
        cluster_std (float): Standard deviation of blobs
        blob_seed (int): Random seed for reproducibility of blob generation
        stretch_seed (int): Random seed for stretching

    Returns:
        X_stretch (ndarray): Transformed feature matrix
        y_true (ndarray): True labels (used only for evaluation)
    """
    np.random.seed(blob_seed)
    X, y_true = make_blobs(n_samples=n_samples, n_features=n_features,
                           centers=centers, cluster_std=cluster_std, random_state=blob_seed)

    # Apply a random transformation to introduce anisotropy (makes GMM more meaningful)
    rng = np.random.RandomState(stretch_seed)
    X_stretch = np.dot(X, rng.randn(2, 2))
    return X_stretch, y_true


def evaluate_clustering(y_true, y_pred, X=None, show_silhouette=True):
    """
    Prints standard clustering evaluation metrics.

    Parameters:
        y_true (ndarray): Ground truth labels
        y_pred (ndarray): Predicted cluster labels
        X (ndarray): Original feature matrix (required for silhouette score)
        show_silhouette (bool): If True, prints silhouette score
    """
    print('Homogeneity score  = %.3f' % metrics.homogeneity_score(y_true, y_pred))
    print('Completeness score = %.3f' % metrics.completeness_score(y_true, y_pred))
    print('V-measure score    = %.3f' % metrics.v_measure_score(y_true, y_pred))

    # Silhouette score does not need ground truth, just data and cluster labels
    if show_silhouette and X is not None:
        try:
            sil = silhouette_score(X, y_pred)
            print('Silhouette score   = %.3f' % sil)
        except Exception as e:
            print(f'Could not compute silhouette score: {e}')


def plot_clusters(X, y_pred, centers=None, method='kmeans', title='Cluster Plot', feature_names=None):
    """
    Scatter plot of clusters with optional range circles (for KMeans).

    Parameters:
        X (ndarray): Feature matrix
        y_pred (ndarray): Cluster labels
        centers (ndarray): Cluster centroids (only for KMeans)
        method (str): 'kmeans' or 'gmm' (determines whether to draw circles)
        title (str): Plot title
        feature_names (list): Custom x and y axis labels
    """
    if feature_names is None:
        feature_names = ["Feature 1", "Feature 0"]

    plt.figure(dpi=100)
    scat = plt.scatter(X[:, 1], X[:, 0], c=y_pred, s=20, alpha=0.7,
                       edgecolors='k', cmap='viridis', label='Data Points')
    ax = plt.gca()

    # If KMeans: draw a light circle around each cluster center
    if method == 'kmeans' and centers is not None:
        for i, center in enumerate(centers):
            radius = np.max(np.linalg.norm(X[y_pred == i] - center, axis=1))
            circle_color = to_rgba(cm.get_cmap('viridis')(i), alpha=0.15)
            circle = plt.Circle((center[1], center[0]), radius, color=circle_color)
            ax.add_artist(circle)

    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    ax.legend(*scat.legend_elements(), title="Predicted Clusters", bbox_to_anchor=(1.3, 1))
    ax.set_aspect(1)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def add_ellipses(gmm, ax, cmap, weight_threshold=None):
    """
    Adds ellipses to represent Gaussian components in a GMM model.

    Parameters:
        gmm (GaussianMixture): Fitted GMM object
        ax (matplotlib.axes): Axis to draw ellipses on
        cmap (Colormap): Colormap for cluster coloring
        weight_threshold (float): Skip drawing ellipses with small weights
    """
    for n in range(gmm.n_components):
        if weight_threshold is not None and gmm.weights_[n] < weight_threshold:
            continue

        # Extract appropriate 2D covariance matrix based on covariance type
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]

        # Compute size and orientation of ellipse
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.degrees(np.arctan2(u[0], u[1]))
        v = 4. * np.sqrt(2.) * np.sqrt(v)

        ellipse_color = to_rgba(cmap(n), alpha=0.15)
        ell = mpl.patches.Ellipse(
            xy=(gmm.means_[n, 1], gmm.means_[n, 0]),
            width=v[1],
            height=v[0],
            angle=90 + angle,
            color=ellipse_color
        )
        ax.add_artist(ell)


def plot_silhouette(X, y_pred, model_name="Model"):
    """
    Generate a silhouette plot showing per-cluster and average silhouette scores.

    Parameters:
        X (ndarray): Feature matrix
        y_pred (ndarray): Predicted cluster labels
        model_name (str): Label for plot title
    """
    n_clusters = len(np.unique(y_pred))
    silhouette_vals = silhouette_samples(X, y_pred)

    y_lower = 10  # space at the bottom of the plot
    plt.figure(dpi=100)

    for i in range(n_clusters):
        ith_cluster_sil = silhouette_vals[y_pred == i]
        ith_cluster_sil.sort()
        size = ith_cluster_sil.shape[0]
        y_upper = y_lower + size

        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil,
                          facecolor=color, edgecolor=color, alpha=0.7, label=f"Cluster {i}")
        y_lower = y_upper + 10

    avg_score = silhouette_score(X, y_pred)
    plt.axvline(x=avg_score, color="red", linestyle="--", label=f"Avg Silhouette = {avg_score:.3f}")
    plt.xlabel("Silhouette Coefficient Values")
    plt.ylabel("Sample Index")
    plt.title(f"Silhouette Plot for {model_name}")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
