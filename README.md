# DecodeLabs-Internship-Project-3
# Project 3: Customer Segmentation (Unsupervised Learning)


## Objective

Discover hidden mathematical groupings in unlabeled retail data using distance‑based algorithms.  
This project demonstrates:

- Dimensionality reduction using **Principal Component Analysis (PCA)** implemented from scratch (eigen‑decomposition of the covariance matrix).
- Determination of the optimal number of clusters using the **Elbow Method (WCSS)** and the **Silhouette Score** – both implemented without external libraries.
- **K‑Means clustering** from scratch using Euclidean distance.
- Reverse engineering of centroids to the original feature space for interpretability.
- Translation of clusters into actionable business personas.

## Dataset

- **File:** `data1.csv` (provided by DecodeLabs) – transaction data with columns:  
  `OrderID`, `Date`, `customerID`, `Product`, `Quantity`, `UnitPrice`, `TotalPrice`.
- The data is aggregated per customer to create the following features:
  - `totalspend` – sum of `TotalPrice` per customer.
  - `avgordervalue` – average `TotalPrice` per order.
  - `frequency` – number of unique `OrderID`s.
  - `avgquantity` – average `Quantity` per transaction.
  - `uniqueproducts` – number of distinct products purchased.
  - `recency` – days since the last order.

## Mathematical Implementation (From Scratch)

All core algorithms are implemented using only `numpy` and `pandas` (no `sklearn` for PCA, K‑Means, or silhouette).

### 1. Standardisation
z = (x - μ) / σ

Ensures each feature contributes equally to distance calculations.

### 2. Principal Component Analysis (PCA)
- Center the data: `X_centered = X - mean`
- Compute covariance matrix: `Σ = (1/(n-1)) * X_centered.T @ X_centered`
- Solve eigen‑decomposition: `Σ v = λ v`
- Sort eigenvalues descending, select the top `ncomp` eigenvectors.
- Project: `X_pca = X_centered @ V_k`
- Explained variance ratio: `evr = λ_i / Σλ`

### 3. K‑Means Clustering
- **Euclidean distance:** `d(p,q) = sqrt(∑(p_i - q_i)²)`
- **Initialisation:** Randomly select `k` points as centroids.
- **Assignment:** Each point is assigned to the nearest centroid.
- **Update:** New centroid = mean of points in the cluster.
- Repeat until convergence or maximum iterations.
- **WCSS (Within‑Cluster Sum of Squares):** `∑ ‖x - centroid‖²`

### 4. Silhouette Score
For each point `i`:
- `a(i)` = mean distance to points in the same cluster.
- `b(i)` = smallest mean distance to points in any other cluster.
- `s(i) = (b(i) - a(i)) / max(a(i), b(i))`
The overall silhouette score is the mean of `s(i)` across all points. Values near +1 indicate well‑separated clusters.

### 5. Optimal K Selection
- Evaluate `k` from 2 to 10.
- **Elbow Method:** Plot WCSS vs `k`; the “elbow” indicates diminishing returns.
- **Silhouette Score:** Choose `k` that maximises the score.

### 6. Reverse Engineering Centroids
- Centroids are obtained in PCA‑space.
- Inverse PCA transform: `centroids_scaled = centroids_pca @ V_k.T`
- Inverse standardisation: `centroids_original = centroids_scaled * σ + μ`

## Pipeline Steps

1. Load `data1.csv` and parse dates.
2. Aggregate transactions per customer to create feature columns.
3. Standardise the feature matrix.
4. Apply PCA to reduce to 2 dimensions for visualisation.
5. For each `k` from 2 to 10:
   - Run K‑Means on the PCA‑reduced data.
   - Compute WCSS and Silhouette Score.
6. Plot the Elbow curve and Silhouette scores.
7. Select the optimal `k` (max silhouette).
8. Run final K‑Means with optimal `k`.
9. Visualise clusters in 2D PCA space.
10. Reverse‑transform centroids to original units.
11. Print cluster profiles (centroids and actual means).
12. Generate business personas based on spending and frequency.
13. Save outputs: `customer_segments_output.csv`, `elbow_silhouette.png`, `cluster_visualisation.png`.

## Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, aggregation, manipulation |
| `numpy` | Numerical computations (mean, std, covariance, eigen‑decomposition, Euclidean distance) |
| `matplotlib` | Plotting (scree plot, elbow, silhouette, cluster visualisation) |

**No `scikit‑learn` is used for PCA, K‑Means, or silhouette – all are implemented from scratch to demonstrate deep mathematical understanding.**

## How to Run

1. Ensure `data1.csv` is in the same folder as the script `project3.py`.
2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib
