import os
import pandas as pd

# ----------------------------
# Read clustered .sd
# ----------------------------
cluster_file = "classified_echo_clusters.csv"
if not os.path.exists(cluster_file):
    raise FileNotFoundError(f"Missing input file: {cluster_file}")

sd = pd.read_csv(cluster_file, header=None, low_memory=False)

sd.columns = [
    "DateTime",
    "Unknown1",
    "SoundSpeed",
    "Unknown2",
    "Depth",
    "Echo",
    "Cluster"
]

# Extract only the time
sd["DateTime"] = sd["DateTime"].astype(str).str.strip()
sd["Time"] = pd.to_datetime(
    sd["DateTime"],
    format="%Y-%m-%d %H:%M:%S.%f",
    errors="coerce"
)
# Keep only rows with a valid timestamp so the merge is reliable
sd = sd.dropna(subset=["Time"])
sd["Time"] = sd["Time"].dt.strftime("%H:%M:%S.%f").str[:-4]

# ----------------------------
# Read .dep
# ----------------------------
dep = pd.read_csv("surveydepthfiltered.dep", header=None)

dep.columns = [
    "DataID",
    "Time",
    "GPSState",
    "Latitude",
    "NS",
    "Longitude",
    "EW",
    "Height",
    "Azimuth",
    "Pitch",
    "Roll",
    "Depth_dep",
    "BaseID",
    "DiffDelay",
    "HRMS",
    "VRMS",
    "Speed",
    "FlowVelocity",
    "FlowDirection",
    "LineDistance",
    "TrailDistance",
    "TrailDirection"
]

# ----------------------------
# Exact timestamp matching
# ----------------------------
merged = pd.merge(
    sd,
    dep[["Time", "Latitude", "NS", "Longitude", "EW"]],
    on="Time",
    how="inner"
)

# ----------------------------
# Save
# ----------------------------
merged.to_csv("classified_echo_clusters_with_gps.csv", index=False)

print("Done!")
print("Matched rows:", len(merged))
