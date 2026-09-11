import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

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

    if "-" not in value or "*" not in value:
        bad_rows += 1
        continue

    wave_string = value.split("-")[1].split("*")[0]

    waveform = np.array([int(c) for c in wave_string])

    all_waveforms.append(waveform)
    valid_indices.append(index)

print("Valid pings:", len(all_waveforms))
print("Skipped rows:", bad_rows)

# ==========================================
# Step 4-10: Feature extraction
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

feature_matrix = np.array(features)

print("Feature matrix:")
print(feature_matrix.shape)

# ==========================================
# Step 11: Use only the first four features
# ==========================================

feature_matrix = np.array(features)

print("Original feature matrix:")
print(feature_matrix.shape)

# Save all six features for later analysis
feature_matrix_all = feature_matrix.copy()

# Select only the first four features for clustering
feature_matrix = feature_matrix[:, :4]

print("Feature matrix used for clustering:")
print(feature_matrix.shape)

# Standardize the four selected features
scaler = StandardScaler()

features_standardized = scaler.fit_transform(feature_matrix)

# ==========================================
# Step 12: PCA (FOR VISUALIZATION ONLY)
# ==========================================

pca = PCA(n_components=2)

features_pca = pca.fit_transform(features_standardized)

print("\nPCA completed")

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

print(f"Total variance retained: {np.sum(pca.explained_variance_ratio_):.2%}")

# Visualise PCA

plt.figure(figsize=(8,6))

plt.scatter(
    features_pca[:,0],
    features_pca[:,1],
    s=5
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Echo Features")

plt.grid(True)

plt.show()

# ==========================================
# Step 13: DBSCAN clustering
# ==========================================

dbscan = DBSCAN(
    eps=0.16,
    min_samples=1200
)

# IMPORTANT:
# Cluster using ALL standardized features
cluster_labels = dbscan.fit_predict(features_standardized)

print("\nDBSCAN completed")

unique_clusters = np.unique(cluster_labels)

print("Clusters found:")
print(unique_clusters)

for cluster in unique_clusters:

    count = np.sum(cluster_labels == cluster)

    if cluster == -1:
        print("Noise points:", count)
    else:
        print(f"Cluster {cluster}: {count} pings")

# ==========================================
# Step 14: Visualise clusters using PCA
# ==========================================

plt.figure(figsize=(8,6))

for cluster in unique_clusters:

    mask = cluster_labels == cluster

    if cluster == -1:

        plt.scatter(
            features_pca[mask,0],
            features_pca[mask,1],
            s=5,
            c="black",
            alpha=0.5,
            label="Noise"
        )

    else:

        plt.scatter(
            features_pca[mask,0],
            features_pca[mask,1],
            s=5,
            label=f"Cluster {cluster}"
        )

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)"
)

plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)"
)

plt.title("DBSCAN Clusters (PCA Visualization)")

plt.legend()

plt.grid(True)

plt.show()

# ==========================================
# Step 15: Save cluster labels
# ==========================================

results = df.iloc[valid_indices].copy()

results["Cluster"] = cluster_labels

print(results.head())

results.to_csv(
    "classified_echo_clusters.csv",
    index=False
)

print("\nSaved file:")
print("classified_echo_clusters.csv")

# ==========================================
# Step 16: Average feature values
# ==========================================

feature_names = [
    "Peak Amplitude",
    "Mean Amplitude",
    "Std Amplitude",
    "Energy",
    "Echo Length",
    "Peak Position"
]

feature_df = pd.DataFrame(
    feature_matrix_all,
    columns=feature_names
)

feature_df["Cluster"] = cluster_labels

cluster_summary = feature_df.groupby("Cluster").mean()

print("\n========================================")
print("Average feature values for each cluster")
print("========================================")

print(cluster_summary)
# ==========================================
# Step 17: Plot random waveforms
# ==========================================

unique_clusters = np.unique(cluster_labels)

for cluster in unique_clusters:

    print(f"\n========== Cluster {cluster} ==========")

    indices = np.where(cluster_labels == cluster)[0]

    print(f"Number of pings: {len(indices)}")

    sample_size = min(10, len(indices))

    selected = random.sample(list(indices), sample_size)

    plt.figure(figsize=(12,6))

    for idx in selected:

        plt.plot(
            all_waveforms[idx],
            label=f"Ping {idx}"
        )

    plt.title(f"Cluster {cluster} - Random Waveforms")

    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")

    plt.grid(True)

    plt.legend(fontsize=8)

    plt.show()
    # ==========================================
# Step 18: Average Waveform of Each Cluster
# Using zero-padding for different lengths
# ==========================================

print("\n========================================")
print("Average waveform of each cluster")
print("========================================")


# Find maximum waveform length in the dataset
max_waveform_length = max(
    len(waveform) for waveform in all_waveforms
)

print("Maximum waveform length:", max_waveform_length)


for cluster in unique_clusters:

    # Skip noise (optional)
    if cluster == -1:
        continue


    indices = np.where(cluster_labels == cluster)[0]

    print(f"\nCluster {cluster}")
    print("Number of waveforms:", len(indices))


    padded_waveforms = []


    # -------------------------------
    # Zero padding
    # -------------------------------

    for idx in indices:

        waveform = all_waveforms[idx]

        padding_length = (
            max_waveform_length - len(waveform)
        )

        padded_waveform = np.pad(
            waveform,
            (0, padding_length),
            mode='constant',
            constant_values=0
        )

        padded_waveforms.append(padded_waveform)


    # Convert to matrix
    padded_waveforms = np.array(padded_waveforms)


    # Calculate average waveform
    average_waveform = np.mean(
        padded_waveforms,
        axis=0
    )


    # Plot average waveform

    plt.figure(figsize=(12,5))

    plt.plot(
        average_waveform,
        linewidth=2
    )

    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")

    plt.title(
        f"Average Echo Waveform - Cluster {cluster}"
    )

    plt.grid(True)

    plt.show()
