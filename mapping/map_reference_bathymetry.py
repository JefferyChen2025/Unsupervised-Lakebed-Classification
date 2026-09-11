# ============================================================
# INTERACTIVE NONNA-10 BATHYMETRY MAP
# Frenchman's Bay
#
# Creates an interactive HTML map with:
#   - Satellite imagery
#   - NONNA-10 bathymetry points
#   - Depth-based colours
#   - Clickable points showing coordinates and depth
#   - Layer controls
#
# Input:
#   OUTPUT.csv
#
# Input CRS:
#   EPSG:3857
#
# Output:
#   NONNA10_interactive_map.html
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import folium

from pyproj import Transformer
from branca.colormap import LinearColormap


# ============================================================
# 2. SETTINGS
# ============================================================

INPUT_FILE = "OUTPUT.csv"

OUTPUT_HTML = "NONNA10_interactive_map.html"

# NONNA-10 NoData value
NODATA_VALUE = 3.4028235e38

# Mean water level above chart datum
WATER_LEVEL = 1.137

# Coordinate systems
INPUT_CRS = "EPSG:3857"
OUTPUT_CRS = "EPSG:4326"


# ============================================================
# 3. READ CSV
# ============================================================

print("=" * 70)
print("READING NONNA-10 DATA")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    sep=r"\s+|,",
    engine="python",
    header=None,
    names=["X", "Y", "Depth"]
)


print("\nTotal rows:", len(df))

print("\nFirst five rows:")
print(df.head())


# ============================================================
# 4. CONVERT TO NUMERIC
# ============================================================

df["X"] = pd.to_numeric(
    df["X"],
    errors="coerce"
)

df["Y"] = pd.to_numeric(
    df["Y"],
    errors="coerce"
)

df["Depth"] = pd.to_numeric(
    df["Depth"],
    errors="coerce"
)


# ============================================================
# 5. REMOVE INVALID COORDINATES
# ============================================================

df = df.dropna(
    subset=["X", "Y"]
).copy()

if df.empty:
    raise ValueError(
        f"No valid coordinate rows found in {INPUT_FILE}. "
        "Check that the file contains X, Y, and Depth columns "
        "separated by commas or whitespace."
    )


# ============================================================
# 6. REMOVE NONNA-10 NODATA
# ============================================================

# Any value greater than 1e30 is treated as NoData

df.loc[
    df["Depth"] >= 1e30,
    "Depth"
] = np.nan


print(
    "\nValid depth cells:",
    df["Depth"].notna().sum()
)

print(
    "NoData cells:",
    df["Depth"].isna().sum()
)


# ============================================================
# 7. REMOVE CELLS AT OR ABOVE WATER SURFACE
# ============================================================

# NONNA-10 depth is relative to chart datum.
#
# Survey water surface = +1.137 m
#
# Therefore cells with:
#
#     Depth <= -1.137
#
# are at or above the water surface.


df = df[
    df["Depth"].notna()
].copy()


df = df[
    df["Depth"] > -WATER_LEVEL
].copy()


print(
    "Underwater cells after waterline masking:",
    len(df)
)

if df.empty:
    raise ValueError(
        "No underwater cells remain after NoData and waterline masking. "
        "Check the depth values and WATER_LEVEL setting."
    )


# ============================================================
# 8. CALCULATE WATER DEPTH
# ============================================================

# Convert:
#
# depth below chart datum
#
# to:
#
# depth below survey water surface


df["WaterDepth"] = (
    df["Depth"] + WATER_LEVEL
)


# ============================================================
# 9. CONVERT EPSG:3857 TO LAT/LON
# ============================================================

print("\nConverting coordinates...")


transformer = Transformer.from_crs(
    INPUT_CRS,
    OUTPUT_CRS,
    always_xy=True
)


longitude, latitude = transformer.transform(
    df["X"].values,
    df["Y"].values
)


df["Longitude"] = longitude

df["Latitude"] = latitude


print(
    "\nLongitude range:",
    df["Longitude"].min(),
    "to",
    df["Longitude"].max()
)

print(
    "Latitude range:",
    df["Latitude"].min(),
    "to",
    df["Latitude"].max()
)


# ============================================================
# 10. DETERMINE MAP CENTER
# ============================================================

map_center_lat = df["Latitude"].mean()

map_center_lon = df["Longitude"].mean()


print(
    "\nMap center:",
    map_center_lat,
    map_center_lon
)


# ============================================================
# 11. CREATE BASE MAP
# ============================================================

m = folium.Map(

    location=[
        map_center_lat,
        map_center_lon
    ],

    zoom_start=14,

    control_scale=True
)


# ============================================================
# 12. ADD SATELLITE IMAGERY
# ============================================================

# Esri World Imagery provides satellite imagery
# and works well as a geographic background.


folium.TileLayer(

    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/World_Imagery/"
        "MapServer/tile/{z}/{y}/{x}"
    ),

    attr=(
        "Tiles © Esri — "
        "Source: Esri, Maxar, Earthstar Geographics, "
        "and the GIS User Community"
    ),

    name="Satellite imagery",

    overlay=False,

    control=True

).add_to(m)


# ============================================================
# 13. ADD STREET MAP
# ============================================================

folium.TileLayer(

    tiles=(
        "https://{s}.tile.openstreetmap.org/"
        "{z}/{x}/{y}.png"
    ),

    attr="© OpenStreetMap contributors",

    name="Street map",

    overlay=False,

    control=True

).add_to(m)


# ============================================================
# 14. CREATE DEPTH COLOUR SCALE
# ============================================================

min_depth = df["WaterDepth"].min()

max_depth = df["WaterDepth"].max()


# Use the actual data range

colormap = LinearColormap(

    colors=[
        "blue",
        "cyan",
        "green",
        "yellow",
        "orange",
        "red"
    ],

    vmin=min_depth,

    vmax=max_depth,

    caption="Water depth (m)"

)


# ============================================================
# 15. CREATE BATHYMETRY LAYER
# ============================================================

bathymetry_layer = folium.FeatureGroup(

    name="NONNA-10 bathymetry",

    show=True

)


# ============================================================
# 16. PLOT EVERY VALID POINT
# ============================================================

print("\nPlotting bathymetry points...")


for _, row in df.iterrows():

    depth = row["WaterDepth"]

    # Get colour from depth
    point_color = colormap(depth)

    # Information shown when point is clicked
    popup_text = f"""
    <b>CHS NONNA-10</b><br>
    <br>
    <b>Water depth:</b> {depth:.3f} m<br>
    <b>Depth below chart datum:</b> {row["Depth"]:.3f} m<br>
    <b>X:</b> {row["X"]:.3f} m<br>
    <b>Y:</b> {row["Y"]:.3f} m<br>
    <b>Longitude:</b> {row["Longitude"]:.7f}°<br>
    <b>Latitude:</b> {row["Latitude"]:.7f}°
    """

    folium.CircleMarker(

        location=[
            row["Latitude"],
            row["Longitude"]
        ],

        radius=4,

        color=point_color,

        fill=True,

        fill_color=point_color,

        fill_opacity=0.8,

        weight=0.5,

        popup=folium.Popup(
            popup_text,
            max_width=300
        )

    ).add_to(
        bathymetry_layer
    )


# Add layer to map

bathymetry_layer.add_to(m)


# ============================================================
# 17. ADD COLOURBAR
# ============================================================

colormap.add_to(m)


# ============================================================
# 18. FIT MAP TO DATA EXTENT
# ============================================================

m.fit_bounds(

    [
        [
            df["Latitude"].min(),
            df["Longitude"].min()
        ],

        [
            df["Latitude"].max(),
            df["Longitude"].max()
        ]
    ]

)


# ============================================================
# 19. ADD LAYER CONTROL
# ============================================================

folium.LayerControl(
    collapsed=False
).add_to(m)



# ============================================================
# 21. SAVE HTML
# ============================================================

m.save(
    OUTPUT_HTML
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MAP CREATED SUCCESSFULLY")
print("=" * 70)

print(
    "\nTotal underwater points:",
    len(df)
)

print(
    "Minimum water depth:",
    round(min_depth, 3),
    "m"
)

print(
    "Maximum water depth:",
    round(max_depth, 3),
    "m"
)

print(
    "\nMap center:"
)

print(
    "Latitude:",
    map_center_lat
)

print(
    "Longitude:",
    map_center_lon
)

print(
    "\nOutput:"
)

print(
    OUTPUT_HTML
)

print(
    "\nOpen the HTML file in Microsoft Edge or Google Chrome."
)

print("\nDone!")
