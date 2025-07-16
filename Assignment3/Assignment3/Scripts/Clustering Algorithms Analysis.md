# Clustering Algorithms Analysis — DBSCAN, k-Means, Hierarchical Clustering

## Project Overview

This project investigates three clustering algorithms — DBSCAN, k-Means, and Hierarchical Clustering — by applying them to synthetic datasets with distinct characteristics. The main objective is to understand DBSCAN’s mechanism for detecting clusters based on density, its ability to handle noise, and how it compares to partitioning and hierarchical methods.

## Datasets Used

The project uses synthetic datasets generated with scikit-learn to simulate real-world clustering challenges. The Moons dataset tests the ability of algorithms to handle non-spherical clusters, while the Blobs dataset demonstrates performance on datasets with varying cluster densities.

## Stepwise Workflow

1. **Data Generation**
   The Moons dataset was generated with the `make_moons` function and the Blobs dataset with the `make_blobs` function. Both datasets were standardized to ensure consistent scale before clustering.

2. **Clustering Algorithms Applied**

   * **DBSCAN** was applied with carefully tuned `eps` and `min_samples` parameters. It performed well on datasets with irregular shapes (such as Moons) and detected noise automatically. However, it struggled with the Blobs dataset due to varying cluster densities.

   * **k-Means** was applied with a predefined number of clusters. It performed best on the Blobs dataset where clusters are spherical and well-separated but performed poorly on the Moons dataset due to its assumption of spherical clusters.

   * **Hierarchical Clustering (Agglomerative)** was applied using Ward’s linkage. It offered reasonable performance and visual interpretability through dendrograms but showed mixed results depending on the dataset structure.

3. **Evaluation through Visuals and Silhouette Scores**

   Each algorithm’s performance was evaluated by plotting the clustered data alongside silhouette analysis plots. Silhouette scores provided quantitative insights into the quality of clusters, helping assess the separation and compactness of the clustering outcomes. Hierarchical clustering results were also analyzed using dendrograms.

4. **Performance Observations**

   On the Moons dataset, DBSCAN accurately identified the crescent-shaped clusters while handling noise, achieving a silhouette score of 0.39. k-Means, with a silhouette score of 0.50, split the clusters unnaturally due to its spherical assumption. Hierarchical Clustering scored 0.45, performing moderately well.

   On the Blobs dataset, k-Means performed best with a silhouette score of 0.79, successfully identifying the spherical clusters. Hierarchical Clustering showed a similar performance with a silhouette score of 0.78. DBSCAN struggled in this scenario, yielding a lower silhouette score of 0.48 due to its sensitivity to density variations.

## Learning Outcomes

Through this project, the following key insights were gained:

* Different clustering algorithms are suited for different data structures and clustering challenges.
* DBSCAN excels in detecting arbitrary-shaped clusters and handling noise but is sensitive to parameter settings and struggles with varying densities.
* k-Means is efficient for spherical, evenly sized clusters but fails with irregular shapes.
* Hierarchical Clustering provides hierarchical insights through dendrograms but may not always outperform other methods.
* Evaluation metrics like silhouette scores and visual inspections are critical in interpreting clustering results.

## How to Run This Project

1. Install Python libraries required for the project using pip:
   pandas, numpy, matplotlib, scikit-learn

2. Place the code files (`data_preprocessor.py` and `main.py`) in the same project directory.

3. Run `main.py` to generate clustering results, visual plots, and silhouette analyses.

4. Review the output graphs for performance comparison.

## References

- Ester, M., Kriegel, H., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with
noise. Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 226–231.

- **Lecture Notes:** Unsupervised Learning I: Clustering (BINF 5507 - Caryn Geady, May 2025).
- **Python Libraries:** 
       - numpy — for numerical operations
       - pandas — for data handling and manipulation
       - matplotlib — for data visualization and plotting
       - scikit-learn — for clustering algorithms, datasets, scaling, and evaluation metrics
- **Dataset:** 
       - Moons Dataset (make_moons): Simulates two interleaving half circles. Used to evaluate clustering performance on non-spherical clusters with potential noise.
       - Blobs Dataset (make_blobs): Generates isotropic Gaussian blobs for clustering. Used to test algorithm performance on spherical clusters with varying densities.


