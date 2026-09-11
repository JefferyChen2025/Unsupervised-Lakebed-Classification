import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# Features used:
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
# Step 8: K-medoids settings
# ==========================================

n_clusters = 5   

max_iterations = 20

random_state = 42

rng = np.random.default_rng(random_state)


# ==========================================
# Step 9: Initialize medoids
# ==========================================

print("\nInitializing K-medoids...")


# Randomly select two observations
# from the FULL dataset as initial medoids.

medoid_indices = rng.choice(
    len(features_standardized),
    size=n_clusters,
    replace=False
)


medoids = features_standardized[
    medoid_indices
].copy()


# ==========================================
# Step 10: K-medoids optimization
# ==========================================

print("\nStarting K-medoids clustering...")


for iteration in range(max_iterations):

    print(
        f"\nIteration {iteration + 1}"
    )


    # --------------------------------------
    # Assign every observation to its
    # nearest medoid
    # --------------------------------------

    distances = np.zeros(
        (
            len(features_standardized),
            n_clusters
        )
    )


    for cluster in range(n_clusters):

        distances[:, cluster] = np.linalg.norm(

            features_standardized
            - medoids[cluster],

            axis=1

        )


    cluster_labels = np.argmin(
        distances,
        axis=1
    )


    # --------------------------------------
    # Update medoids
    # --------------------------------------

    new_medoids = np.zeros_like(
        medoids
    )


    for cluster in range(n_clusters):

        cluster_indices = np.where(
            cluster_labels == cluster
        )[0]


        print(
            f"Cluster {cluster}: "
            f"{len(cluster_indices)} observations"
        )


        if len(cluster_indices) == 0:

            # If cluster becomes empty,
            # choose a random observation.

            random_index = rng.integers(
                len(features_standardized)
            )

            new_medoids[cluster] = (
                features_standardized[
                    random_index
                ]
            )

            continue


        cluster_points = (
            features_standardized[
                cluster_indices
            ]
        )


        # ----------------------------------
        # Calculate the point whose total
        # distance to all other points in
        # the cluster is smallest.
        #
        # This point becomes the medoid.
        #
        # WARNING:
        # This operation can be extremely
        # expensive for very large clusters.
        # ----------------------------------

        n_points = len(cluster_points)


        best_total_distance = np.inf

        best_position = 0


        # Process candidate medoids in
        # manageable blocks.

        block_size = 1000


        for start in range(
            0,
            n_points,
            block_size
        ):

            end = min(
                start + block_size,
                n_points
            )


            candidates = cluster_points[
                start:end
            ]


            # Distance from candidates
            # to every point in the cluster

            candidate_distances = np.linalg.norm(

                candidates[:, np.newaxis, :]
                - cluster_points[np.newaxis, :, :],

                axis=2

            )


            total_distances = (
                candidate_distances.sum(
                    axis=1
                )
            )


            local_position = np.argmin(
                total_distances
            )


            local_distance = (
                total_distances[
                    local_position
                ]
            )


            if local_distance < best_total_distance:

                best_total_distance = (
                    local_distance
                )

                best_position = (
                    start + local_position
                )


        new_medoids[cluster] = (
            cluster_points[
                best_position
            ]
        )


    # --------------------------------------
    # Check convergence
    # --------------------------------------

    medoid_change = np.max(

        np.linalg.norm(

            new_medoids
            - medoids,

            axis=1

        )

    )


    print(
        "Medoid change:",
        medoid_change
    )


    medoids = new_medoids


    if medoid_change < 1e-5:

        print(
            "\nK-medoids converged."
        )

        break


print(
    "\nK-medoids clustering completed."
)


# ==========================================
# Step 11: Final cluster assignment
# ==========================================

print(
    "\nPerforming final cluster assignment..."
)


distances = np.zeros(
    (
        len(features_standardized),
        n_clusters
    )
)


for cluster in range(n_clusters):

    distances[:, cluster] = np.linalg.norm(

        features_standardized
        - medoids[cluster],

        axis=1

    )


cluster_labels = np.argmin(
    distances,
    axis=1
)


print(
    "Final cluster assignment completed."
)


# ==========================================
# Step 12: Silhouette score
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
    f"\nK-medoids Silhouette Score: "
    f"{silhouette:.4f}"
)


# ==========================================
# Step 13: PCA visualization
# ==========================================

plt.figure(figsize=(8, 6))


for cluster in range(n_clusters):

    mask = (
        cluster_labels == cluster
    )


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

    f"K-medoids Clustering\n"
    f"Silhouette Score = {silhouette:.4f}"

)


plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()
#https://doi.org/10.3390/jmse9050508
