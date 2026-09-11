import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


# ==========================================
# Step 1: Read raw data file
# ==========================================

df = pd.read_csv("rawdataall.sd", header=None)

print("File loaded successfully!")
print("Raw data size:", df.shape)


# ==========================================
# Step 2-3: Extract waveform from every ping
# ==========================================

all_waveforms = []
valid_indices = []
bad_rows = 0

for index, value in enumerate(df.iloc[:, -1]):

    value = str(value)

    if "-" not in value or "*" not in value:
        bad_rows += 1
        continue

    try:

        wave_string = value.split("-")[1].split("*")[0]

        waveform = np.array([
            int(c) for c in wave_string
        ])

        all_waveforms.append(waveform)
        valid_indices.append(index)

    except ValueError:

        bad_rows += 1
        continue


print("Valid pings:", len(all_waveforms))
print("Skipped rows:", bad_rows)


# ==========================================
# Step 4: Feature extraction
# ==========================================

features = []

for waveform in all_waveforms:

    peak_amplitude = np.max(waveform)

    mean_amplitude = np.mean(waveform)

    std_amplitude = np.std(waveform)

    energy = np.sum(waveform ** 2)

    echo_length = len(waveform)

    peak_position = np.argmax(waveform)

    features.append([
        peak_amplitude,
        mean_amplitude,
        std_amplitude,
        energy,
        echo_length,
        peak_position
    ])


feature_matrix_all = np.array(features)

print("\nOriginal feature matrix:")
print(feature_matrix_all.shape)


# ==========================================
# Step 5: Select four features
# ==========================================

# Features used for clustering:
#
# 1. Peak amplitude
# 2. Mean amplitude
# 3. Standard deviation
# 4. Energy
#
# Echo length and peak position are excluded.

feature_matrix = feature_matrix_all[:, :4]

print("\nFeature matrix used for clustering:")
print(feature_matrix.shape)


# ==========================================
# Step 6: Standardize features
# ==========================================

scaler = StandardScaler()

features_standardized = scaler.fit_transform(
    feature_matrix
)

print("\nFeatures standardized.")


# ==========================================
# Step 7: PCA
# FOR VISUALIZATION ONLY
# ==========================================

pca = PCA(n_components=2)

features_pca = pca.fit_transform(
    features_standardized
)

print("\nPCA completed.")

print(
    "PC1 variance:",
    f"{pca.explained_variance_ratio_[0]:.2%}"
)

print(
    "PC2 variance:",
    f"{pca.explained_variance_ratio_[1]:.2%}"
)

print(
    "Total variance retained:",
    f"{np.sum(pca.explained_variance_ratio_):.2%}"
)


# ==========================================
# Step 8: Gaussian Mixture Model
# ==========================================

# Number of clusters
n_clusters = 5


gmm = GaussianMixture(

    n_components=n_clusters,

    # Each cluster has its own
    # covariance matrix
    covariance_type="full",

    # Reproducibility
    random_state=42,

    # Maximum EM iterations
    max_iter=200,

    # Number of initializations
    n_init=3
)


print("\nStarting GMM clustering...")


# Fit GMM using ALL observations
# and ALL four standardized features

cluster_labels = gmm.fit_predict(
    features_standardized
)


print("\nGMM clustering completed.")


# ==========================================
# Step 9: Silhouette score
# ==========================================

print(
    "\nCalculating silhouette score "
    "using ALL observations..."
)


silhouette = silhouette_score(

    features_standardized,

    cluster_labels

)


print(
    f"\nGMM Silhouette Score: "
    f"{silhouette:.4f}"
)


# ==========================================
# Step 10: PCA visualization
# ==========================================

plt.figure(figsize=(8, 6))


for cluster in range(n_clusters):

    mask = cluster_labels == cluster

    plt.scatter(

        features_pca[mask, 0],

        features_pca[mask, 1],

        s=5,

        label=f"Cluster {cluster}"

    )


# ==========================================
# Axis labels
# ==========================================

plt.xlabel(

    f"PC1 "
    f"({pca.explained_variance_ratio_[0] * 100:.1f}% variance)"

)

plt.ylabel(

    f"PC2 "
    f"({pca.explained_variance_ratio_[1] * 100:.1f}% variance)"

)


# ==========================================
# Plot title
# ==========================================

plt.title(

    f"Gaussian Mixture Model Clustering\n"
    f"Silhouette Score = {silhouette:.4f}"

)


plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()
#https://www.sciencedirect.com/science/article/pii/S0301479725006140
