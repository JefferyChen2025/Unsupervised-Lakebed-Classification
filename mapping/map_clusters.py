import pandas as pd
import folium
import matplotlib.pyplot as plt


# -----------------------------
# Convert DMS to decimal degrees
# -----------------------------
def dms_to_decimal(dms):

    dms = dms.replace("°", " ").replace("′", " ").replace("″", "")
    parts = dms.split()

    deg = float(parts[0])
    minute = float(parts[1])
    second = float(parts[2])

    return deg + minute/60 + second/3600


# -----------------------------
# Load CSV
# -----------------------------

df = pd.read_csv("classified_echo_clusters_with_gps.csv")


# Convert coordinates

df["Lat"] = df["Latitude"].apply(dms_to_decimal)
df["Lon"] = df["Longitude"].apply(dms_to_decimal)


# Correct hemisphere

df.loc[df["NS"] == "S", "Lat"] *= -1
df.loc[df["EW"] == "W", "Lon"] *= -1

# -----------------------------
# Remove noise points (Cluster = -1)
# -----------------------------

df = df[df["Cluster"] != -1].copy()

print("Removed noise points (Cluster -1)")
print("Remaining waveforms:", len(df))

# -----------------------------
# Automatically detect clusters
# -----------------------------

clusters = sorted(df["Cluster"].unique())

print("Detected clusters:", clusters)


# Create colors automatically

colors = [
    "red",
    "blue",
    "green",
    "purple",
    "orange",
    "black",
    "pink",
    "gray"
]

cluster_color = {}

for i, c in enumerate(clusters):
    cluster_color[c] = colors[i % len(colors)]



# -----------------------------
# Create map
# -----------------------------

center = [
    df["Lat"].mean(),
    df["Lon"].mean()
]


m = folium.Map(
    location=center,
    zoom_start=18
)



# -----------------------------
# Plot cluster points
# -----------------------------

for _, row in df.iterrows():

    cluster = row["Cluster"]

    folium.CircleMarker(
        location=[
            row["Lat"],
            row["Lon"]
        ],

        radius=2,

        color=cluster_color[cluster],

        fill=True,
        fill_color=cluster_color[cluster],
        fill_opacity=0.8,

        popup=f"Cluster: {cluster}"

    ).add_to(m)



# -----------------------------
# Add cluster legend
# -----------------------------

legend_html = """
<div style="
position: fixed;
bottom: 50px;
left: 50px;
width: 200px;
background-color: white;
border: 2px solid grey;
z-index: 9999;
font-size: 14px;
padding: 10px;
border-radius: 5px;
">

<b>Cluster Classification</b>
<br><br>
"""

for cluster, color in cluster_color.items():

    legend_html += f"""
    <div>
        <span style="
        display:inline-block;
        width:12px;
        height:12px;
        background:{color};
        margin-right:8px;
        border-radius:50%;
        ">
        </span>
        Cluster {cluster}
    </div>
    """

legend_html += """
</div>
"""


m.get_root().html.add_child(
    folium.Element(legend_html)
)



# Save map

m.save("cluster_only_map.html")

print("Finished: cluster_only_map.html")
