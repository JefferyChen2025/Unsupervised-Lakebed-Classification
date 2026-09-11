# ============================================================
# BATHYMETRY COMPARISON MAP
# Surveyed Bathymetry vs CHS NONNA-10
# Frenchman's Bay
#
# SURVEY DATA:
#   Column 4  = Latitude DMS
#   Column 5  = Latitude hemisphere
#   Column 6  = Longitude DMS
#   Column 7  = Longitude hemisphere
#   Column 12 = Surveyed water depth (m)
#
# NONNA-10 DATA:
#   Column 1 = X coordinate, EPSG:3857
#   Column 2 = Y coordinate, EPSG:3857
#   Column 3 = Bottom elevation relative to chart datum
#
# VERTICAL DATUM:
#   Water surface = +1.137 m relative to chart datum
#
# NONNA-10 water depth is calculated as:
#
#   NONNA water depth = WATER_LEVEL - NONNA elevation
#
# Depth difference is:
#
#   Difference = Survey depth - NONNA water depth
#
# Points where:
#
#   |Difference| > 0.5 m
#
# are highlighted in RED.
#
# OUTPUT:
#   bathymetry_comparison_map.html
#   bathymetry_depth_comparison.csv
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import folium

from pyproj import Transformer

from scipy.spatial import cKDTree


# ============================================================
# 2. SETTINGS
# ============================================================

# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

SURVEY_FILE = "surveydepth.dep"

NONNA_FILE = "OUTPUT.csv"


# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

OUTPUT_HTML = "bathymetry_comparison_map.html"

OUTPUT_CSV = "bathymetry_depth_comparison.csv"


# ------------------------------------------------------------
# Difference threshold
# ------------------------------------------------------------

DEPTH_THRESHOLD = 0.5


# ------------------------------------------------------------
# Water surface elevation
#
# Relative to NONNA-10 chart datum
# ------------------------------------------------------------

WATER_LEVEL = 1.137


# ------------------------------------------------------------
# Coordinate systems
# ------------------------------------------------------------

NONNA_CRS = "EPSG:3857"

LATLON_CRS = "EPSG:4326"


# ============================================================
# 3. FUNCTION: DMS TO DECIMAL DEGREES
# ============================================================

def dms_to_decimal(dms):

    """
    Convert DMS coordinate such as:

        043°48′44.9809884″

    into decimal degrees.
    """

    if pd.isna(dms):

        return np.nan


    dms = str(dms).strip()


    try:

        # Replace degree/minute/second symbols

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
# 4. READ SURVEY DATA
# ============================================================

print("=" * 70)

print("READING SURVEY BATHYMETRY DATA")

print("=" * 70)


survey = pd.read_csv(

    SURVEY_FILE,

    header=None,

    sep=",",

    dtype=str

)


print(

    "\nTotal survey rows:",

    len(survey)

)


print(

    "\nNumber of columns:",

    survey.shape[1]

)


# ============================================================
# 5. CHECK SURVEY COLUMN COUNT
# ============================================================

if survey.shape[1] < 12:

    raise ValueError(

        "Survey file has fewer than 12 columns. "
        "Column 12 is required for surveyed depth."

    )


# ============================================================
# 6. EXTRACT SURVEY INFORMATION
# ============================================================

# Python uses zero-based indexing.
#
# Column 4  -> index 3
# Column 5  -> index 4
# Column 6  -> index 5
# Column 7  -> index 6
# Column 12 -> index 11


survey["Latitude_DMS"] = survey.iloc[:, 3]

survey["Lat_Hemisphere"] = survey.iloc[:, 4]

survey["Longitude_DMS"] = survey.iloc[:, 5]

survey["Lon_Hemisphere"] = survey.iloc[:, 6]


survey["SurveyDepth"] = pd.to_numeric(

    survey.iloc[:, 11],

    errors="coerce"

)


# ============================================================
# 7. CONVERT SURVEY LATITUDE
# ============================================================

survey["Latitude"] = (

    survey["Latitude_DMS"]

    .apply(dms_to_decimal)

)


# Southern hemisphere correction

survey.loc[

    survey["Lat_Hemisphere"]
    .str.upper()
    == "S",

    "Latitude"

] *= -1


# ============================================================
# 8. CONVERT SURVEY LONGITUDE
# ============================================================

survey["Longitude"] = (

    survey["Longitude_DMS"]

    .apply(dms_to_decimal)

)


# Western hemisphere correction

survey.loc[

    survey["Lon_Hemisphere"]
    .str.upper()
    == "W",

    "Longitude"

] *= -1


# ============================================================
# 9. REMOVE INVALID SURVEY DATA
# ============================================================

survey = survey.dropna(

    subset=[

        "Latitude",

        "Longitude",

        "SurveyDepth"

    ]

).copy()


# Remove negative measured water depths

survey = survey[

    survey["SurveyDepth"] >= 0

].copy()


if survey.empty:

    raise ValueError(

        "No valid survey points remain."

    )


print(

    "\nValid survey points:",

    len(survey)

)


# ============================================================
# 10. DISPLAY SURVEY DEPTH RANGE
# ============================================================

print(

    "\nSurvey depth range:",

    round(survey["SurveyDepth"].min(), 3),

    "to",

    round(survey["SurveyDepth"].max(), 3),

    "m"

)


# ============================================================
# 11. READ NONNA-10 DATA
# ============================================================

print("\n" + "=" * 70)

print("READING CHS NONNA-10 DATA")

print("=" * 70)


nonna = pd.read_csv(

    NONNA_FILE,

    sep=r"\s+|,",

    engine="python",

    header=None,

    names=[

        "X",

        "Y",

        "Depth"

    ]

)


print(

    "\nTotal NONNA-10 rows:",

    len(nonna)

)


# ============================================================
# 12. CONVERT NONNA DATA TO NUMERIC
# ============================================================

nonna["X"] = pd.to_numeric(

    nonna["X"],

    errors="coerce"

)


nonna["Y"] = pd.to_numeric(

    nonna["Y"],

    errors="coerce"

)


nonna["Depth"] = pd.to_numeric(

    nonna["Depth"],

    errors="coerce"

)


# ============================================================
# 13. REMOVE NONNA-10 NODATA
# ============================================================

# NONNA-10 NoData value:
#
# 3.4028235e38
#
# Anything >= 1e30 is treated as NoData.


nonna.loc[

    nonna["Depth"] >= 1e30,

    "Depth"

] = np.nan


nonna = nonna.dropna(

    subset=[

        "X",

        "Y",

        "Depth"

    ]

).copy()


print(

    "Valid NONNA-10 points:",

    len(nonna)

)


# ============================================================
# 14. IMPORTANT:
#     INTERPRET NONNA-10 DEPTH
# ============================================================

print("\nConverting NONNA-10 elevation to water depth...")


# NONNA-10 Depth is assumed to be:
#
#     bottom elevation relative to chart datum
#
# Water surface:
#
#     +1.137 m
#
# Therefore:
#
#     water depth
#     =
#     water surface elevation
#     -
#     bottom elevation
#
#
# Example:
#
# NONNA elevation = -0.147 m
#
# Water surface = +1.137 m
#
# Water depth =
#
#     1.137 - (-0.147)
#
#     = 1.284 m


nonna["NONNA_WaterDepth"] = (

    WATER_LEVEL
    - nonna["Depth"]

)


# ============================================================
# 15. DISPLAY NONNA DEPTH RANGE
# ============================================================

print(

    "\nNONNA-10 elevation range:",

    round(nonna["Depth"].min(), 3),

    "to",

    round(nonna["Depth"].max(), 3),

    "m"

)


print(

    "\nNONNA-10 calculated water depth range:",

    round(nonna["NONNA_WaterDepth"].min(), 3),

    "to",

    round(nonna["NONNA_WaterDepth"].max(), 3),

    "m"

)


# ============================================================
# 16. CREATE COORDINATE TRANSFORMER
# ============================================================

transformer = Transformer.from_crs(

    LATLON_CRS,

    NONNA_CRS,

    always_xy=True

)


# ============================================================
# 17. CONVERT SURVEY LAT/LON TO EPSG:3857
# ============================================================

print(

    "\nConverting survey coordinates to EPSG:3857..."

)


survey_x, survey_y = transformer.transform(

    survey["Longitude"].values,

    survey["Latitude"].values

)


survey["X_3857"] = survey_x

survey["Y_3857"] = survey_y


# ============================================================
# 18. BUILD NONNA-10 KD TREE
# ============================================================

print(

    "\nBuilding NONNA-10 spatial index..."

)


nonna_coordinates = np.column_stack(

    (

        nonna["X"].values,

        nonna["Y"].values

    )

)


tree = cKDTree(

    nonna_coordinates

)


# ============================================================
# 19. FIND NEAREST NONNA-10 POINT
# ============================================================

print(

    "\nMatching survey points to nearest NONNA-10 point..."

)


survey_coordinates = np.column_stack(

    (

        survey["X_3857"].values,

        survey["Y_3857"].values

    )

)


distances, indices = tree.query(

    survey_coordinates,

    k=1

)


# ============================================================
# 20. ADD MATCHED NONNA INFORMATION
# ============================================================

survey["MatchDistance_m"] = distances


survey["NONNA_Elevation"] = (

    nonna.iloc[indices]["Depth"]

    .values

)


survey["NONNA_WaterDepth"] = (

    nonna.iloc[indices]["NONNA_WaterDepth"]

    .values

)


# ============================================================
# 21. CALCULATE DEPTH DIFFERENCE
# ============================================================

# Positive difference:
#
# Survey measured deeper than NONNA-10
#
# Negative difference:
#
# Survey measured shallower than NONNA-10


survey["DepthDifference"] = (

    survey["SurveyDepth"]

    - survey["NONNA_WaterDepth"]

)


# Absolute difference

survey["AbsoluteDifference"] = (

    survey["DepthDifference"]

    .abs()

)


# ============================================================
# 22. FLAG DIFFERENCES > 0.5 m
# ============================================================

survey["DifferenceOver0.5m"] = (

    survey["AbsoluteDifference"]

    > DEPTH_THRESHOLD

)


# ============================================================
# 23. PRINT COMPARISON STATISTICS
# ============================================================

print("\n" + "=" * 70)

print("DEPTH COMPARISON")

print("=" * 70)


print(

    "\nMatched survey points:",

    len(survey)

)


print(

    "\nMean survey depth:",

    round(

        survey["SurveyDepth"].mean(),

        3

    ),

    "m"

)


print(

    "Mean NONNA-10 water depth:",

    round(

        survey["NONNA_WaterDepth"].mean(),

        3

    ),

    "m"

)


print(

    "\nMean depth difference:",

    round(

        survey["DepthDifference"].mean(),

        3

    ),

    "m"

)


print(

    "Mean absolute difference:",

    round(

        survey["AbsoluteDifference"].mean(),

        3

    ),

    "m"

)


print(

    "Maximum absolute difference:",

    round(

        survey["AbsoluteDifference"].max(),

        3

    ),

    "m"

)


# ============================================================
# 24. COUNT LARGE DIFFERENCES
# ============================================================

large_difference_count = (

    survey["DifferenceOver0.5m"]

    .sum()

)


large_difference_percentage = (

    large_difference_count

    / len(survey)

    * 100

)


print(

    "\nPoints with |difference| > 0.5 m:",

    large_difference_count

)


print(

    "Percentage:",

    round(

        large_difference_percentage,

        2

    ),

    "%"

)


# ============================================================
# 25. MATCH DISTANCE STATISTICS
# ============================================================

print(

    "\nNearest NONNA-10 point distance:"

)


print(

    "Minimum:",

    round(

        survey["MatchDistance_m"].min(),

        2

    ),

    "m"

)


print(

    "Mean:",

    round(

        survey["MatchDistance_m"].mean(),

        2

    ),

    "m"

)


print(

    "Maximum:",

    round(

        survey["MatchDistance_m"].max(),

        2

    ),

    "m"

)


# ============================================================
# 26. SAVE COMPARISON CSV
# ============================================================

survey.to_csv(

    OUTPUT_CSV,

    index=False

)


print(

    "\nComparison CSV saved:",

    OUTPUT_CSV

)


# ============================================================
# 27. CREATE MAP
# ============================================================

map_center_lat = (

    survey["Latitude"].mean()

)


map_center_lon = (

    survey["Longitude"].mean()

)


m = folium.Map(

    location=[

        map_center_lat,

        map_center_lon

    ],

    zoom_start=15,

    control_scale=True

)


# ============================================================
# 28. SATELLITE IMAGERY
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
# 29. STREET MAP
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
# 30. CREATE MAP LAYERS
# ============================================================

normal_layer = folium.FeatureGroup(

    name="Difference ≤ 0.5 m",

    show=True

)


large_difference_layer = folium.FeatureGroup(

    name="Difference > 0.5 m",

    show=True

)


# ============================================================
# 31. PLOT SURVEY POINTS
# ============================================================

print(

    "\nPlotting comparison points..."

)


for _, row in survey.iterrows():

    difference = row["DepthDifference"]

    absolute_difference = row["AbsoluteDifference"]


    # --------------------------------------------------------
    # Determine colour
    # --------------------------------------------------------

    if absolute_difference > DEPTH_THRESHOLD:

        point_color = "red"

        target_layer = large_difference_layer

    else:

        point_color = "blue"

        target_layer = normal_layer


    # --------------------------------------------------------
    # Determine interpretation
    # --------------------------------------------------------

    if difference > 0:

        interpretation = (

            "Survey is deeper than NONNA-10"

        )

    elif difference < 0:

        interpretation = (

            "Survey is shallower than NONNA-10"

        )

    else:

        interpretation = (

            "Survey and NONNA-10 are equal"

        )


    # --------------------------------------------------------
    # Popup
    # --------------------------------------------------------

    popup_text = f"""

    <b>Bathymetry Comparison</b><br>
    <br>

    <b>Surveyed depth:</b>
    {row["SurveyDepth"]:.3f} m
    <br>

    <b>NONNA-10 elevation:</b>
    {row["NONNA_Elevation"]:.3f} m
    <br>

    <b>NONNA-10 water depth:</b>
    {row["NONNA_WaterDepth"]:.3f} m
    <br>

    <b>Depth difference:</b>
    {difference:+.3f} m
    <br>

    <b>Absolute difference:</b>
    {absolute_difference:.3f} m
    <br>

    <b>Interpretation:</b>
    {interpretation}
    <br>

    <b>Distance to NONNA-10 point:</b>
    {row["MatchDistance_m"]:.2f} m
    <br>

    <br>

    <b>Latitude:</b>
    {row["Latitude"]:.7f}°
    <br>

    <b>Longitude:</b>
    {row["Longitude"]:.7f}°

    """


    # --------------------------------------------------------
    # Add point
    # --------------------------------------------------------

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

            max_width=350

        )

    ).add_to(

        target_layer

    )


# ============================================================
# 32. ADD MAP LAYERS
# ============================================================

normal_layer.add_to(m)

large_difference_layer.add_to(m)


# ============================================================
# 33. ADD LEGEND
# ============================================================

legend_html = """

<div style="

position: fixed;

bottom: 40px;

left: 40px;

z-index: 9999;

background-color: white;

padding: 12px;

border: 2px solid grey;

border-radius: 5px;

font-size: 14px;

">

<b>Depth Difference</b>

<br><br>

<span style="
color:blue;
font-size:18px;
">●</span>

Difference ≤ 0.5 m

<br>

<span style="
color:red;
font-size:18px;
">●</span>

Difference > 0.5 m

</div>

"""


m.get_root().html.add_child(

    folium.Element(

        legend_html

    )

)


# ============================================================
# 34. FIT MAP TO SURVEY EXTENT
# ============================================================

m.fit_bounds(

    [

        [

            survey["Latitude"].min(),

            survey["Longitude"].min()

        ],

        [

            survey["Latitude"].max(),

            survey["Longitude"].max()

        ]

    ]

)


# ============================================================
# 35. LAYER CONTROL
# ============================================================

folium.LayerControl(

    collapsed=False

).add_to(m)


# ============================================================
# 36. SAVE HTML
# ============================================================

m.save(

    OUTPUT_HTML

)


# ============================================================
# 37. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("COMPARISON MAP CREATED SUCCESSFULLY")

print("=" * 70)


print(

    "\nSurvey points:",

    len(survey)

)


print(

    "Points with |difference| > 0.5 m:",

    large_difference_count

)


print(

    "Percentage > 0.5 m:",

    round(

        large_difference_percentage,

        2

    ),

    "%"

)


print(

    "\nMean absolute difference:",

    round(

        survey["AbsoluteDifference"].mean(),

        3

    ),

    "m"

)


print(

    "\nHTML map:",

    OUTPUT_HTML

)


print(

    "Comparison CSV:",

    OUTPUT_CSV

)


print(

    "\nOpen the HTML file in Microsoft Edge or Google Chrome."

)


print("\nDone!")
