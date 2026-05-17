"""
ECG200 Clustering Validation — CL-Occam Thesis, Clustering Module Evaluation.

The ECG200 benchmark (UCR/UEA archive) is used to validate the DTW-based
clustering approach before applying it to the forecasting pipeline.
Known class labels enable ground-truth evaluation via the Adjusted Rand Index.

Three experiments are conducted:
    Experiment 1 — DTW distances + Spectral Clustering (primary CL-Occam approach).
    Experiment 2 — KMeans on z-scored flattened features (baseline comparison).
    Experiment 3 — DTW + Spectral Clustering with automatic k selection via
                    silhouette analysis (validates the internal-index criterion).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from sklearn.cluster import KMeans, SpectralClustering
from sktime.datasets import load_UCR_UEA_dataset
from fastdtw import fastdtw


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

dataset_name = "ECG200"
X_train_raw, y_train = load_UCR_UEA_dataset(dataset_name, split="train", return_X_y=True)
X_test_raw, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_X_y=True)

print("Dataset Information:")
print(f"  Training set shape: {X_train_raw.shape}")
print(f"  Test set shape:     {X_test_raw.shape}")
print(f"  Number of classes:  {len(np.unique(y_train))}")
print("\nClass distribution — training:")
print(pd.Series(y_train).value_counts())
print("\nClass distribution — test:")
print(pd.Series(y_test).value_counts())


def dataframe_to_2darray(df):
    """Convert sktime nested DataFrame to a plain 2-D numpy array."""
    num_samples = df.shape[0]
    num_timesteps = len(df.iloc[0, 0])
    array_2d = np.empty((num_samples, num_timesteps))
    for i in range(num_samples):
        array_2d[i, :] = df.iloc[i, 0]
    return array_2d


X_train = dataframe_to_2darray(X_train_raw)
X_test = dataframe_to_2darray(X_test_raw)

# Fit scaler on training data only; transform test set with the same parameters
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def create_dtw_matrix(X):
    """Compute a symmetric pairwise DTW distance matrix."""
    n = len(X)
    distance_matrix = np.zeros((n, n))
    print("Computing DTW distance matrix...")
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            distance, _ = fastdtw(X[i], X[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix


def distance_to_similarity(distance_matrix):
    """Convert a distance matrix to a Gaussian-kernel similarity matrix."""
    return np.exp(-distance_matrix / distance_matrix.std())


def calculate_cluster_stats(X, labels, y_true, n_clusters, prefix):
    """Compute per-cluster class purity and descriptive statistics."""
    stats = []
    for i in range(n_clusters):
        cluster_data = X[labels == i]
        true_classes = y_true[labels == i]
        dominant_class = pd.Series(true_classes).mode()[0]
        class_purity = (true_classes == dominant_class).mean() * 100
        stats.append({
            'Set': prefix,
            'Cluster': i,
            'Size': len(cluster_data),
            'Dominant_Class': dominant_class,
            'Purity_%': class_purity,
            'Mean': np.mean(cluster_data),
            'Std': np.std(cluster_data),
        })
    return stats


# ---------------------------------------------------------------------------
# Experiment 1: DTW + Spectral Clustering
# ---------------------------------------------------------------------------

print("\n=== Experiment 1: DTW + Spectral Clustering ===")

train_dtw_matrix = create_dtw_matrix(X_train_scaled)
test_dtw_matrix = create_dtw_matrix(X_test_scaled)

# Use number of known classes as the target cluster count for ARI evaluation
n_clusters = len(np.unique(y_train))

train_similarity = distance_to_similarity(train_dtw_matrix)
spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
train_labels = spectral.fit_predict(train_similarity)

# Assign test labels via nearest-neighbour lookup in DTW space
test_labels = np.zeros(len(X_test))
for i in range(len(X_test)):
    distances = [fastdtw(X_test_scaled[i], x)[0] for x in X_train_scaled]
    closest_idx = np.argmin(distances)
    test_labels[i] = train_labels[closest_idx]

train_ari = adjusted_rand_score(y_train, train_labels)
test_ari = adjusted_rand_score(y_test, test_labels)

print("\nClustering Performance:")
print(f"  Training ARI: {train_ari:.3f}")
print(f"  Test ARI:     {test_ari:.3f}")

colors = ['red', 'blue']
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
for i, color in enumerate(colors):
    for series in X_train_scaled[train_labels == i]:
        ax1.plot(series, color=color, alpha=0.5)
ax1.set_title('Training Set — DTW + Spectral Clustering')
ax1.set_xlabel('Time')
ax1.set_ylabel('z-score')
for i, color in enumerate(colors):
    for series in X_test_scaled[test_labels == i]:
        ax2.plot(series, color=color, alpha=0.5)
ax2.set_title('Test Set — DTW + Spectral Clustering')
ax2.set_xlabel('Time')
ax2.set_ylabel('z-score')
plt.tight_layout()
plt.show()

print("\nCluster-Class Distribution (Training):")
print(pd.crosstab(train_labels, y_train))
print("\nCluster-Class Distribution (Test):")
print(pd.crosstab(test_labels, y_test))

train_stats = calculate_cluster_stats(X_train_scaled, train_labels, y_train, n_clusters, 'Train')
test_stats = calculate_cluster_stats(X_test_scaled, test_labels, y_test, n_clusters, 'Test')
print("\nDetailed Cluster Statistics:")
print(pd.DataFrame(train_stats + test_stats).round(2))

# Overlay of ground-truth class shapes for visual comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
for cls in np.unique(y_train):
    ax1.plot(X_train_scaled[y_train == cls].T, alpha=0.5)
ax1.set_title('Training Set — True Classes')
ax1.set_xlabel('Time')
ax1.set_ylabel('z-score')
ax1.legend()
for cls in np.unique(y_test):
    ax2.plot(X_test_scaled[y_test == cls].T, alpha=0.5)
ax2.set_title('Test Set — True Classes')
ax2.set_xlabel('Time')
ax2.set_ylabel('z-score')
ax2.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# Experiment 2: KMeans on z-scored flattened features (baseline)
# ---------------------------------------------------------------------------

print("\n=== Experiment 2: KMeans Baseline ===")

n_clusters = len(np.unique(y_train))
print(f"  Number of clusters: {n_clusters} (matching known class count)")

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
kmeans.fit(X_train_scaled)
train_labels = kmeans.labels_
test_labels = kmeans.predict(X_test_scaled)

train_silhouette = silhouette_score(X_train_scaled, train_labels)
test_silhouette = silhouette_score(X_test_scaled, test_labels)
train_ari = adjusted_rand_score(y_train, train_labels)
test_ari = adjusted_rand_score(y_test, test_labels)

print("\nClustering Performance:")
print(f"  Training Silhouette: {train_silhouette:.3f}")
print(f"  Test Silhouette:     {test_silhouette:.3f}")
print(f"  Training ARI:        {train_ari:.3f}")
print(f"  Test ARI:            {test_ari:.3f}")

colors_k = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
for i, color in enumerate(colors_k):
    for series in X_train[train_labels == i]:
        ax1.plot(series, color=color, alpha=0.5)
ax1.set_title('Training Set — KMeans Clusters')
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
for i, color in enumerate(colors_k):
    for series in X_test[test_labels == i]:
        ax2.plot(series, color=color, alpha=0.5)
ax2.set_title('Test Set — KMeans Clusters')
ax2.set_xlabel('Time')
ax2.set_ylabel('Value')
plt.tight_layout()
plt.show()

print("\nCluster-Class Distribution (Training):")
print(pd.crosstab(train_labels, y_train))
print("\nCluster-Class Distribution (Test):")
print(pd.crosstab(test_labels, y_test))

train_stats = calculate_cluster_stats(X_train, train_labels, y_train, n_clusters, 'Train')
test_stats = calculate_cluster_stats(X_test, test_labels, y_test, n_clusters, 'Test')
print("\nDetailed Cluster Statistics:")
print(pd.DataFrame(train_stats + test_stats).round(2))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
for cls in np.unique(y_train):
    for series in X_train[y_train == cls]:
        ax1.plot(series, alpha=0.5)
ax1.set_title('Training Set — True Classes')
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
for cls in np.unique(y_test):
    for series in X_test[y_test == cls]:
        ax2.plot(series, alpha=0.5)
ax2.set_title('Test Set — True Classes')
ax2.set_xlabel('Time')
ax2.set_ylabel('Value')
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# Experiment 3: DTW + Spectral Clustering with silhouette-based k selection
# ---------------------------------------------------------------------------

print("\n=== Experiment 3: DTW + Spectral Clustering with Automatic k Selection ===")

train_dtw_matrix = create_dtw_matrix(X_train_scaled)
train_similarity = distance_to_similarity(train_dtw_matrix)

range_n_clusters = range(2, 6)
silhouette_avg_scores = []
silhouette_scores_all = []

print("\nSilhouette analysis over k = 2..5:")
for n_cl in range_n_clusters:
    spectral = SpectralClustering(n_clusters=n_cl, affinity='precomputed', random_state=42)
    cluster_labels = spectral.fit_predict(train_similarity)
    silhouette_avg = silhouette_score(train_dtw_matrix, cluster_labels)
    silhouette_values = silhouette_samples(train_dtw_matrix, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)
    silhouette_scores_all.append(silhouette_values)
    print(f"  k={n_cl}: avg silhouette = {silhouette_avg:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(range_n_clusters, silhouette_avg_scores, 'o-')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('Average Silhouette Score')
ax1.set_title('Silhouette Score vs Number of Clusters')
ax1.grid(True)

for idx, n_cl in enumerate(range_n_clusters):
    y_lower = 10
    silhouette_vals = silhouette_scores_all[idx]
    spectral = SpectralClustering(n_clusters=n_cl, affinity='precomputed', random_state=42)
    cl_labels = spectral.fit_predict(train_similarity)
    for i in range(n_cl):
        cluster_silhouette_vals = silhouette_vals[cl_labels == i]
        cluster_silhouette_vals.sort()
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.nipy_spectral(float(i) / n_cl)
        ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                          facecolor=color, alpha=0.7)
        y_lower = y_upper + 10
    ax2.axvline(x=silhouette_avg_scores[idx], color='red', linestyle='--')

ax2.set_xlabel('Silhouette Coefficient')
ax2.set_ylabel('Cluster Label')
ax2.set_title('Silhouette Analysis for Different Cluster Sizes')
plt.tight_layout()
plt.show()

optimal_n_clusters = list(range_n_clusters)[np.argmax(silhouette_avg_scores)]
print(f"\nOptimal k (silhouette criterion): {optimal_n_clusters}")

spectral_optimal = SpectralClustering(
    n_clusters=optimal_n_clusters, affinity='precomputed', random_state=42
)
train_labels = spectral_optimal.fit_predict(train_similarity)

colors = plt.cm.rainbow(np.linspace(0, 1, optimal_n_clusters))
plt.figure(figsize=(15, 10))
for i, color in enumerate(colors):
    plt.plot(X_train_scaled[train_labels == i].T, color=color, alpha=0.5)
plt.title(f'Final Clustering — k={optimal_n_clusters}')
plt.xlabel('Time')
plt.ylabel('z-score')
plt.legend()
plt.show()

print("\nCluster-Class Distribution:")
print(pd.crosstab(train_labels, y_train))

train_stats = calculate_cluster_stats(X_train_scaled, train_labels, y_train, optimal_n_clusters, 'Train')
cluster_stats = pd.DataFrame(train_stats)
print("\nDetailed Cluster Statistics:")
print(cluster_stats.round(2))