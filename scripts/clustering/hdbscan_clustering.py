import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import hdbscan


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
# Step 8: HDBSCAN clustering
# ==========================================

print("\nStarting HDBSCAN clustering...")


clusterer = hdbscan.HDBSCAN(

    # Minimum number of observations
    # required to form a cluster
    min_cluster_size=1200,

    # Controls the minimum density required
    # for points to be considered part of
    # a cluster
    min_samples=1200,

    # Euclidean distance in the
    # standardized four-dimensional
    # feature space
    metric="euclidean",

    # Excess of mass cluster selection
    cluster_selection_method="eom",

    # Do not generate prediction data
    # because it is not needed
    prediction_data=False
)


cluster_labels = clusterer.fit_predict(
    features_standardized
)


print("\nHDBSCAN clustering completed.")


# ==========================================
# Step 9: Silhouette score
# ==========================================

print(
    "\nCalculating silhouette score "
    "using ALL observations..."
)


# ------------------------------------------
# Important:
#
# HDBSCAN uses -1 to indicate noise.
#
# Silhouette score should normally evaluate
# the actual clusters rather than treating
# noise as a normal cluster.
# ------------------------------------------

non_noise_mask = (
    cluster_labels != -1
)


non_noise_features = (
    features_standardized[
        non_noise_mask
    ]
)


non_noise_labels = (
    cluster_labels[
        non_noise_mask
    ]
)


# Check that at least two clusters remain

unique_clusters = np.unique(
    non_noise_labels
)


if len(unique_clusters) >= 2:

    silhouette = silhouette_score(

        non_noise_features,

        non_noise_labels

    )

    print(
        f"\nHDBSCAN Silhouette Score: "
        f"{silhouette:.4f}"
    )

else:

    silhouette = np.nan

    print(
        "\nSilhouette score cannot be "
        "calculated because fewer than "
        "two clusters were found."
    )


# ==========================================
# Step 10: PCA visualization
# ==========================================

plt.figure(figsize=(8, 6))


# ------------------------------------------
# Plot clusters
# ------------------------------------------

for cluster in np.unique(
    cluster_labels
):

    mask = (
        cluster_labels == cluster
    )


    if cluster == -1:

        plt.scatter(

            features_pca[mask, 0],

            features_pca[mask, 1],

            s=5,

            c="black",

            alpha=0.4,

            label="Noise"

        )

    else:

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
# Title
# ==========================================

if not np.isnan(silhouette):

    plt.title(

        f"HDBSCAN Clustering\n"
        f"Silhouette Score = {silhouette:.4f}"

    )

else:

    plt.title(
        "HDBSCAN Clustering"
    )


plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()
#https://pmc.ncbi.nlm.nih.gov/articles/PMC12978570
