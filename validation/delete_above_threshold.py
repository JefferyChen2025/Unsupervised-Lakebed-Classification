import pandas as pd
import os

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "bathymetry_depth_comparison.csv"
OUTPUT_FILE = "bathymetry_depth_comparison_filtered.csv"


# ============================================================
# READ ORIGINAL COMPARISON FILE
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Cannot find {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)


# ============================================================
# REMOVE ROWS WHERE FINAL COLUMN IS TRUE
# ============================================================

# The last column is DifferenceOver0.5m
# Do not modify any other column.

last_column = df.columns[-1]

filtered = df[
    df[last_column].astype(str).str.strip().str.lower() != "true"
].copy()


# ============================================================
# SAVE
# ============================================================

filtered.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULT
# ============================================================

print("Done!")

print("Original rows:", len(df))

print("Filtered rows:", len(filtered))

print("Rows removed:", len(df) - len(filtered))

print("Output file:", OUTPUT_FILE)
