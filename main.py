"""
Iris Flower Clustering Project
- Loads the Iris dataset
- Scales features
- Applies K-Means clustering (k=3)
- Visualizes clusters and compares them with true labels
- Applies PCA to reduce dimensions to 2 components
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, adjusted_rand_score, silhouette_score

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # 1. Load dataset
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    df = X.copy()
    df["true_species"] = y.map(dict(enumerate(target_names)))

    print("Dataset shape:", X.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nMissing values:\n", X.isnull().sum())

    # 2. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Apply K-Means (k=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["predicted_cluster"] = clusters

    print("\nCluster counts:\n", df["predicted_cluster"].value_counts().sort_index())
    print("Silhouette score:", round(silhouette_score(X_scaled, clusters), 4))
    print("Adjusted Rand Index:", round(adjusted_rand_score(y, clusters), 4))

    # 4. Scatter plot using petal measurements
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="petal length (cm)", y="petal width (cm)",
        hue="predicted_cluster", palette="Set1", s=75
    )
    plt.title("Iris Flowers: K-Means Clusters (k=3)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_clusters.png"), dpi=150)
    plt.close()

    # 5. Compare predicted clusters with actual species
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(
        data=df, x="petal length (cm)", y="petal width (cm)",
        hue="true_species", palette="Set2", s=65, ax=axes[0]
    )
    axes[0].set_title("Actual Iris Species")
    sns.scatterplot(
        data=df, x="petal length (cm)", y="petal width (cm)",
        hue="predicted_cluster", palette="Set1", s=65, ax=axes[1]
    )
    axes[1].set_title("Predicted K-Means Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    # 6. PCA: reduce four features to two dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["predicted_cluster"] = clusters
    pca_df["true_species"] = y.map(dict(enumerate(target_names)))

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2",
        hue="predicted_cluster", palette="Set1", s=75
    )
    plt.title("K-Means Clusters after PCA (2D)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_clusters.png"), dpi=150)
    plt.close()

    print("\nPCA explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance retained:", round(pca.explained_variance_ratio_.sum() * 100, 2), "%")

    # Save data and comparison table
    df.to_csv(os.path.join(OUTPUT_DIR, "iris_clustered_data.csv"), index=False)
    comparison = pd.crosstab(
        df["true_species"], df["predicted_cluster"],
        rownames=["Actual species"], colnames=["Predicted cluster"]
    )
    comparison.to_csv(os.path.join(OUTPUT_DIR, "cluster_comparison.csv"))
    print("\nActual species vs predicted clusters:\n", comparison)
    print(f"\nProject completed. Charts and CSV files are saved in '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
