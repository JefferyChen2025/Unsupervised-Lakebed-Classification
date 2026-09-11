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
# Step 8: CLARA settings
# ==========================================

n_clusters = 5

# Number of observations in each
# CLARA subsample.
#
# CLARA uses subsamples because applying
# exact PAM/K-medoids to 179,375 observations
# is computationally impractical.

sample_size = 5000

# Number of CLARA samples

n_samples = 5

random_state = 42

rng = np.random.default_rng(
    random_state
)


# ==========================================
# Step 9: PAM function for each CLARA sample
# ==========================================

def pam_kmedoids(
    data,
    n_clusters,
    max_iterations=100
):

    n = len(data)

    # --------------------------------------
    # Initial medoids
    # --------------------------------------

    medoid_indices = rng.choice(
        n,
        size=n_clusters,
        replace=False
    )

    medoids = data[
        medoid_indices
    ].copy()


    # --------------------------------------
    # PAM iterations
    # --------------------------------------

    for iteration in range(
        max_iterations
    ):

        # Distance from every point
        # to every medoid

        distances = np.zeros(
            (n, n_clusters)
        )


        for cluster in range(
            n_clusters
        ):

            distances[:, cluster] = (
                np.linalg.norm(
                    data
                    - medoids[cluster],
                    axis=1
                )
            )


        # Assign each observation
        # to nearest medoid

        labels = np.argmin(
            distances,
            axis=1
        )


        new_medoids = np.zeros_like(
            medoids
        )


        # ----------------------------------
        # Find new medoid for each cluster
        # ----------------------------------

        for cluster in range(
            n_clusters
        ):

            cluster_indices = np.where(
                labels == cluster
            )[0]


            # Empty cluster

            if len(cluster_indices) == 0:

                random_index = rng.integers(
                    n
                )

                new_medoids[cluster] = (
                    data[random_index]
                )

                continue


            cluster_data = data[
                cluster_indices
            ]


            # Calculate pairwise distances
            # within this sample cluster

            pairwise_distances = np.linalg.norm(

                cluster_data[:, np.newaxis, :]
                -
                cluster_data[np.newaxis, :, :],

                axis=2

            )


            # Total distance from each
            # candidate medoid to all
            # observations in the cluster

            total_distance = (
                pairwise_distances.sum(
                    axis=1
                )
            )


            best_index = np.argmin(
                total_distance
            )


            new_medoids[cluster] = (
                cluster_data[best_index]
            )


        # ----------------------------------
        # Check convergence
        # ----------------------------------

        change = np.max(

            np.linalg.norm(
                new_medoids
                - medoids,
                axis=1
            )

        )


        medoids = new_medoids


        if change < 1e-5:

            break


    return medoids


# ==========================================
# Step 10: CLARA
# ==========================================

print("\n========================================")
print("Starting CLARA")
print("========================================")

best_medoids = None

best_cost = np.inf


for sample_number in range(
    n_samples
):

    print(
        f"\nCLARA sample "
        f"{sample_number + 1}/{n_samples}"
    )


    # --------------------------------------
    # Draw a random sample from the
    # COMPLETE dataset
    # --------------------------------------

    sample_indices = rng.choice(

        len(features_standardized),

        size=sample_size,

        replace=False

    )


    sample_data = (
        features_standardized[
            sample_indices
        ]
    )


    # --------------------------------------
    # Run PAM/K-medoids on the sample
    # --------------------------------------

    medoids = pam_kmedoids(

        sample_data,

        n_clusters=n_clusters,

        max_iterations=50

    )


    # --------------------------------------
    # Assign ALL observations in the
    # dataset to the candidate medoids
    # --------------------------------------

    full_distances = np.zeros(

        (
            len(features_standardized),
            n_clusters
        )

    )


    for cluster in range(
        n_clusters
    ):

        full_distances[:, cluster] = (
            np.linalg.norm(
                features_standardized
                - medoids[cluster],
                axis=1
            )
        )


    nearest_distance = np.min(

        full_distances,

        axis=1

    )


    # --------------------------------------
    # CLARA cost
    #
    # Mean distance of all observations
    # to their closest medoid
    # --------------------------------------

    cost = np.mean(
        nearest_distance
    )


    print(
        f"CLARA cost: {cost:.6f}"
    )


    # --------------------------------------
    # Keep best medoids
    # --------------------------------------

    if cost < best_cost:

        best_cost = cost

        best_medoids = medoids.copy()


        print(
            "New best medoids found."
        )


# ==========================================
# Step 11: Final assignment
# ==========================================

print(
    "\nFinal CLARA medoids selected."
)


final_distances = np.zeros(

    (
        len(features_standardized),
        n_clusters
    )

)


for cluster in range(
    n_clusters
):

    final_distances[:, cluster] = (
        np.linalg.norm(
            features_standardized
            - best_medoids[cluster],
            axis=1
        )
    )


cluster_labels = np.argmin(

    final_distances,

    axis=1

)


print(
    "All observations assigned "
    "to their nearest medoid."
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
    f"\nCLARA Silhouette Score: "
    f"{silhouette:.4f}"
)


# ==========================================
# Step 13: PCA visualization
# ==========================================

plt.figure(figsize=(8, 6))


for cluster in range(
    n_clusters
):

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
# Title
# ==========================================

plt.title(

    f"CLARA Clustering\n"
    f"Silhouette Score = {silhouette:.4f}"

)


plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()
#Hamilton, L. J. (2011)
