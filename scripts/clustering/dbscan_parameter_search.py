import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score



# ==========================================
# Step 1: Load raw .sd file
# ==========================================

df = pd.read_csv(
    "rawdataall.sd",
    header=None
)

print("Raw data loaded")
print("Data size:", df.shape)



# ==========================================
# Step 2: Extract waveform from each ping
# ==========================================

all_waveforms = []

bad_rows = 0


for value in df.iloc[:, -1]:

    if "-" not in value or "*" not in value:

        bad_rows += 1
        continue


    wave_string = value.split("-")[1].split("*")[0]


    waveform = np.array(
        [int(c) for c in wave_string]
    )


    all_waveforms.append(waveform)



print("Valid pings:", len(all_waveforms))
print("Skipped rows:", bad_rows)



# ==========================================
# Step 3: Extract echo features
# ==========================================

features = []


for waveform in all_waveforms:

    peak_amplitude = np.max(waveform)

    mean_amplitude = np.mean(waveform)

    std_amplitude = np.std(waveform)

    energy = np.sum(waveform ** 2)

    echo_length = len(waveform)

    peak_position = np.argmax(waveform)


    features.append(
        [
            peak_amplitude,
            mean_amplitude,
            std_amplitude,
            energy,
            echo_length,
            peak_position
        ]
    )



feature_matrix = np.array(features)


print("Feature matrix shape:")
print(feature_matrix.shape)



# ==========================================
# Step 4: Standardize six features
# ==========================================

scaler = StandardScaler()

X = scaler.fit_transform(
    feature_matrix
)



# ==========================================
# Step 5: Search best DBSCAN parameters
# ==========================================

best_score = -1

best_eps = None

best_min_samples = None

best_clusters = None



# Search range
eps_values = np.arange(
    0.1,
    0.4,
    0.01
)


min_samples_values = [
    1200
]


results = []


for eps in eps_values:

    for min_samples in min_samples_values:

        print(f"Testing eps={eps}, min_samples={min_samples}")

        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )


        labels = dbscan.fit_predict(X)
    


        # ==========================================
        # Count noise points
        # ==========================================

        noise_points = np.sum(labels == -1)
        print(f"Noise points: {noise_points}")

        # Ignore parameter combinations with too much noise
        if noise_points > 170000:

            print(f"Skipped (noise={noise_points})")

            continue


        # Remove noise points
        mask = labels != -1

        cluster_labels = labels[mask]



        # Silhouette requires at least 2 clusters

        if len(np.unique(cluster_labels)) < 2:

            continue



        score = silhouette_score(
            X[mask],
            cluster_labels
        )


        number_clusters = len(
            np.unique(cluster_labels)
        )


        results.append(
            [
                eps,
                min_samples,
                score,
                number_clusters,
                noise_points
            ]
        )


        if score > best_score:

            best_score = score

            best_eps = eps

            best_min_samples = min_samples

            best_clusters = number_clusters



# ==========================================
# Step 6: Display best result
# ==========================================

print("\n==============================")
print("Best DBSCAN Parameters")
print("==============================")


print(
    "Best eps:",
    best_eps
)


print(
    "Best min_samples:",
    best_min_samples
)


print(
    "Best silhouette score:",
    round(best_score,4)
)


print(
    "Number of clusters:",
    best_clusters
)



# ==========================================
# Step 7: Save search results
# ==========================================

results_df = pd.DataFrame(
    results,
    columns=[
        "eps",
        "min_samples",
        "silhouette_score",
        "number_of_clusters",
        "noise_points"
    ]
)


results_df.to_csv(
    "dbscan_silhouette_search_results.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "dbscan_silhouette_search_results.csv"
)
