# Iris Flower Clustering Project

## Objective
Apply unsupervised machine learning to group Iris flowers into three clusters using K-Means, visualize the groups, compare them with the actual species, and reduce dimensions using PCA.

## Dataset
The project uses the built-in Iris dataset from `scikit-learn` (150 samples, 4 numerical features, 3 species). An internet download is not required.

## Tasks completed
1. Load and inspect the Iris dataset.
2. Standardize the four input features.
3. Apply K-Means with `k=3`.
4. Visualize predicted clusters using petal length and petal width.
5. Compare actual species with predicted clusters.
6. Apply PCA to reduce four dimensions to two.
7. Evaluate clustering with Silhouette Score and Adjusted Rand Index.
8. Export clustered data and a comparison table as CSV files.

## Run the project

### 1. Create and activate a virtual environment (optional)
Windows:
```bash
python -m venv .venv
.venv\\Scripts\\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
python main.py
```

## Output
The program creates an `outputs/` folder containing:
- `kmeans_clusters.png` — scatter plot of predicted clusters.
- `actual_vs_predicted.png` — actual species and predicted clusters side by side.
- `pca_clusters.png` — clusters in two-dimensional PCA space.
- `iris_clustered_data.csv` — dataset with predicted cluster IDs.
- `cluster_comparison.csv` — cross-tabulation of actual species and predicted clusters.

## Notes
- Cluster IDs (0, 1, 2) are arbitrary and may not correspond to species IDs.
- K-Means is unsupervised: true species labels are used only for comparison, not for training.
- Standardization is important because the features have different scales.
