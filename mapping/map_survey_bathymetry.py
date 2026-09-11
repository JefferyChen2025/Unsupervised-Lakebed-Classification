# ============================================================
# INTERACTIVE SURVEYED BATHYMETRY MAP
# Frenchman's Bay
#
# Data format:
# Column 4  = Latitude DMS
# Column 5  = Latitude hemisphere
# Column 6  = Longitude DMS
# Column 7  = Longitude hemisphere
# Column 12 = Surveyed depth (m)
#
# Output:
# surveyed_bathymetry_map.html
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import folium

from branca.colormap import LinearColormap


# ============================================================
# 2. SETTINGS
# ============================================================

INPUT_FILE = "surveydepth.dep"

OUTPUT_HTML = "surveyed_bathymetry_map.html"


# ============================================================
# 3. READ DATA
# ============================================================

print("=" * 70)
print("READING SURVEYED BATHYMETRY DATA")
print("=" * 70)


df = pd.read_csv(
    INPUT_FILE,
    header=None,
    sep=",",
    dtype=str
)


print("\nTotal rows:", len(df))

print("\nFirst five rows:")
print(df.head())


# ============================================================
# 4. CHECK NUMBER OF COLUMNS
# ============================================================

print("\nNumber of columns:", df.shape[1])


if df.shape[1] < 12:

    raise ValueError(
        "The input file contains fewer than 12 columns. "
        "Column 12 is required for surveyed depth."
    )


# ============================================================
# 5. EXTRACT REQUIRED COLUMNS
# ============================================================

# Python uses zero-based indexing.
#
# Column 4  -> index 3
# Column 5  -> index 4
# Column 6  -> index 5
# Column 7  -> index 6
# Column 12 -> index 11


df["Latitude_DMS"] = df.iloc[:, 3]

df["Lat_Hemisphere"] = df.iloc[:, 4]

df["Longitude_DMS"] = df.iloc[:, 5]

df["Lon_Hemisphere"] = df.iloc[:, 6]

df["Depth"] = pd.to_numeric(
    df.iloc[:, 11],
    errors="coerce"
)


# ============================================================
# 6. DMS TO DECIMAL DEGREES
# ============================================================

def dms_to_decimal(dms):

    """
    Convert coordinates such as:

        043°48′44.9809884″

    into decimal degrees.
    """

    if pd.isna(dms):
        return np.nan

    dms = str(dms).strip()

    try:

        # Replace possible Unicode symbols
        dms = (
            dms
            .replace("°", " ")
            .replace("′", " ")
            .replace("″", " ")
            .replace("'", " ")
            .replace('"', " ")
        )

        parts = dms.split()

        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])

        decimal = (
            degrees
            + minutes / 60
            + seconds / 3600
        )

        return decimal

    except Exception:

        return np.nan


# ============================================================
# 7. CONVERT LATITUDE
# ============================================================

df["Latitude"] = df["Latitude_DMS"].apply(
    dms_to_decimal
)


# Apply hemisphere

df.loc[
    df["Lat_Hemisphere"].str.upper() == "S",
    "Latitude"
] *= -1


# ============================================================
# 8. CONVERT LONGITUDE
# ============================================================

df["Longitude"] = df["Longitude_DMS"].apply(
    dms_to_decimal
)


# Apply hemisphere

df.loc[
    df["Lon_Hemisphere"].str.upper() == "W",
    "Longitude"
] *= -1


# ============================================================
# 9. REMOVE INVALID DATA
# ============================================================

df = df.dropna(
    subset=[
        "Latitude",
        "Longitude",
        "Depth"
    ]
).copy()


# Remove impossible depth values

df = df[
    df["Depth"] >= 0
].copy()


if df.empty:

    raise ValueError(
        "No valid surveyed bathymetry points remain."
    )


# ============================================================
# 10. PRINT DATA INFORMATION
# ============================================================

print("\nValid surveyed points:", len(df))

print(
    "\nLatitude range:",
    df["Latitude"].min(),
    "to",
    df["Latitude"].max()
)

print(
    "Longitude range:",
    df["Longitude"].min(),
    "to",
    df["Longitude"].max()
)

print(
    "\nDepth range:",
    df["Depth"].min(),
    "to",
    df["Depth"].max(),
    "m"
)


# ============================================================
# 11. MAP CENTER
# ============================================================

map_center_lat = df["Latitude"].mean()

map_center_lon = df["Longitude"].mean()


print(
    "\nMap center:",
    map_center_lat,
    map_center_lon
)


# ============================================================
# 12. CREATE MAP
# ============================================================

m = folium.Map(

    location=[
        map_center_lat,
        map_center_lon
    ],

    zoom_start=15,

    control_scale=True

)


# ============================================================
# 13. SATELLITE IMAGERY
# ============================================================

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
# 14. STREET MAP
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
# 15. DEPTH COLOUR SCALE
# ============================================================

min_depth = df["Depth"].min()

max_depth = df["Depth"].max()


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

    caption="Surveyed water depth (m)"

)


# ============================================================
# 16. CREATE SURVEY LAYER
# ============================================================

survey_layer = folium.FeatureGroup(

    name="Surveyed bathymetry",

    show=True

)


# ============================================================
# 17. PLOT SURVEYED POINTS
# ============================================================

print("\nPlotting surveyed bathymetry points...")


for _, row in df.iterrows():

    depth = row["Depth"]

    point_color = colormap(depth)


    # --------------------------------------------------------
    # Popup information
    # --------------------------------------------------------

    popup_text = f"""

    <b>Surveyed Bathymetry</b><br>
    <br>

    <b>Depth:</b>
    {depth:.3f} m
    <br>

    <b>Latitude:</b>
    {row["Latitude"]:.7f}°
    <br>

    <b>Longitude:</b>
    {row["Longitude"]:.7f}°
    <br>

    <b>Original latitude:</b>
    {row["Latitude_DMS"]}
    <br>

    <b>Original longitude:</b>
    {row["Longitude_DMS"]}

    """


    folium.CircleMarker(

        location=[
            row["Latitude"],
            row["Longitude"]
        ],

        radius=3,

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
        survey_layer
    )


# Add survey layer to map

survey_layer.add_to(m)


# ============================================================
# 18. ADD COLOURBAR
# ============================================================

colormap.add_to(m)


# ============================================================
# 19. FIT MAP TO SURVEY EXTENT
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
# 20. LAYER CONTROL
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
    "\nTotal surveyed points:",
    len(df)
)

print(
    "Minimum surveyed depth:",
    round(min_depth, 3),
    "m"
)

print(
    "Maximum surveyed depth:",
    round(max_depth, 3),
    "m"
)

print(
    "\nOutput:",
    OUTPUT_HTML
)

print(
    "\nOpen the HTML file in Microsoft Edge or Google Chrome."
)

print("\nDone!")
