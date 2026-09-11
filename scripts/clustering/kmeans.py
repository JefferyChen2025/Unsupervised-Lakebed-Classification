import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# ==========================================
# Step 1: Read raw data file
# ==========================================

df = pd.read_csv("rawdataall.sd", header=None)

print("File loaded successfully!")
print("Raw data size:", df.shape)

# ==========================================
# Step 2: Extract waveform from every ping
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

print("\nWaveform extraction complete")
print("Valid pings:", len(all_waveforms))
print("Skipped rows:", bad_rows)

# ==========================================
# Step 3: Feature extraction
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

print("\nFeature matrix size:")
print(feature_matrix.shape)

# ==========================================
# Step 4: Select first four features
# ==========================================

# Keep all six features for possible later analysis
feature_matrix_all = feature_matrix.copy()

# Select only:
# 0: Peak Amplitude
# 1: Mean Amplitude
# 2: Std Amplitude
# 3: Energy

feature_matrix = feature_matrix[:, :4]

print("\nFeature matrix used for K-Means:")
print(feature_matrix.shape)


# ==========================================
# Step 5: Standardise four features
# ==========================================

scaler = StandardScaler()

features_standardized = scaler.fit_transform(feature_matrix)

print("\nFour features standardised.")

# ==========================================
# Step 5: K-Means Clustering (6-dimensional)
# ==========================================

k = 5

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

# ***** IMPORTANT CHANGE *****
cluster_labels = kmeans.fit_predict(features_standardized)

print("\nK-Means completed.")

# ==========================================
# Calculate Silhouette Score
# ==========================================

sil_score = silhouette_score(
    features_standardized,
    cluster_labels,
    sample_size=10000,
    random_state=42
)

print(f"\nSilhouette Score: {sil_score:.4f}")

for cluster in range(k):
    count = np.sum(cluster_labels == cluster)
    print(f"Cluster {cluster}: {count} pings")

# ==========================================
# Step 6: PCA for VISUALIZATION ONLY
# ==========================================

pca = PCA(n_components=2)

features_pca = pca.fit_transform(features_standardized)

print("\nPCA complete")
print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
print("Total explained variance:",
      np.sum(pca.explained_variance_ratio_))

# ==========================================
# Plot clustering result using PCA
# ==========================================

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    features_pca[:,0],
    features_pca[:,1],
    c=cluster_labels,
    cmap="viridis",
    s=5
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clusters (PCA Visualization)")

plt.colorbar(scatter, label="Cluster")

plt.grid(True)

plt.show()

# ==========================================
# Step 7: Save clustering results
# ==========================================

results = df.iloc[valid_indices].copy()

results["Cluster"] = cluster_labels

results.to_csv(
    "kmeans_echo_clusters.csv",
    index=False
)

print("\nResults saved to:")
print("kmeans_echo_clusters.csv")
