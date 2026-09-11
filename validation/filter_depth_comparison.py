import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

ORIGINAL_DEP = "surveydepth.dep"

FILTERED_CSV = "bathymetry_depth_comparison_filtered.csv"

OUTPUT_DEP = "surveydepthfiltered.dep"


# ============================================================
# 1. CHECK FILES
# ============================================================

if not os.path.exists(ORIGINAL_DEP):
    raise FileNotFoundError(
        f"Missing file: {ORIGINAL_DEP}"
    )

if not os.path.exists(FILTERED_CSV):
    raise FileNotFoundError(
        f"Missing file: {FILTERED_CSV}"
    )


# ============================================================
# 2. READ FILTERED COMPARISON CSV
# ============================================================

print("=" * 70)
print("READING FILTERED COMPARISON FILE")
print("=" * 70)


filtered = pd.read_csv(
    FILTERED_CSV,
    dtype=str,
    low_memory=False
)


print(
    "\nFiltered CSV rows:",
    len(filtered)
)


print(
    "Filtered CSV columns:"
)

print(
    filtered.columns.tolist()
)


# ============================================================
# 3. GET TIMESTAMP FROM FILTERED CSV
# ============================================================

# Your original .dep structure is:
#
# Column 0 = DataID
# Column 1 = Time
# Column 2 = GPS state
# ...
#
# The comparison CSV retains these original columns,
# therefore column "1" contains the survey time.


if "1" not in filtered.columns:

    raise ValueError(
        "Column '1' was not found in the filtered CSV. "
        "Column '1' should contain the survey timestamp."
    )


filtered_times = (

    filtered["1"]
    .astype(str)
    .str.strip()

)


# ============================================================
# 4. NORMALIZE TIMESTAMPS
# ============================================================

# Your timestamps look like:
#
# 23:28:05.58
# 23:28:06.88
# 23:28:07.68
#
# Convert them to a consistent time format.


filtered_time_values = set()


for t in filtered_times:

    try:

        parsed = pd.to_datetime(
            t,
            format="%H:%M:%S.%f",
            errors="coerce"
        )

        if pd.notna(parsed):

            normalized = parsed.strftime(
                "%H:%M:%S.%f"
            )

            # Keep milliseconds / hundredths precision
            normalized = normalized[:11]

            filtered_time_values.add(
                normalized
            )

    except Exception:
        pass


print(
    "\nUnique timestamps in filtered CSV:",
    len(filtered_time_values)
)


# ============================================================
# 5. READ ORIGINAL .DEP AS RAW TEXT
# ============================================================

print("\n" + "=" * 70)
print("READING ORIGINAL .DEP FILE")
print("=" * 70)


with open(
    ORIGINAL_DEP,
    "r",
    encoding="utf-8",
    errors="replace"
) as f:

    original_lines = f.readlines()


print(
    "\nOriginal .dep lines:",
    len(original_lines)
)


# ============================================================
# 6. MATCH ORIGINAL .DEP TIMESTAMPS
# ============================================================

kept_lines = []

removed_lines = []

kept_count = 0

removed_count = 0

invalid_time_count = 0


for line in original_lines:

    stripped = line.strip()


    # Skip blank lines

    if not stripped:

        continue


    # --------------------------------------------------------
    # Split original .dep line
    # --------------------------------------------------------

    fields = stripped.split(",")


    # Need at least two columns:
    #
    # column 0 = DataID
    # column 1 = Time

    if len(fields) < 2:

        invalid_time_count += 1

        continue


    # --------------------------------------------------------
    # Get timestamp from COLUMN 1
    # --------------------------------------------------------

    time_value = (
        fields[1]
        .strip()
    )


    # --------------------------------------------------------
    # Normalize timestamp
    # --------------------------------------------------------

    parsed = pd.to_datetime(
        time_value,
        format="%H:%M:%S.%f",
        errors="coerce"
    )


    if pd.isna(parsed):

        invalid_time_count += 1

        continue


    normalized_time = (
        parsed.strftime(
            "%H:%M:%S.%f"
        )[:11]
    )


    # ========================================================
    # MATCH BY TIME
    # ========================================================

    if normalized_time in filtered_time_values:

        # KEEP THE ORIGINAL LINE EXACTLY AS IT WAS

        kept_lines.append(line)

        kept_count += 1

    else:

        removed_lines.append(line)

        removed_count += 1


# ============================================================
# 7. WRITE FILTERED .DEP
# ============================================================

with open(
    OUTPUT_DEP,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    f.writelines(kept_lines)


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TIME-BASED FILTERING COMPLETE")
print("=" * 70)


print(
    "\nOriginal .dep lines:",
    len(original_lines)
)


print(
    "Lines kept:",
    kept_count
)


print(
    "Lines removed:",
    removed_count
)


print(
    "Invalid timestamp lines:",
    invalid_time_count
)


print(
    "\nFiltered CSV rows:",
    len(filtered)
)


print(
    "Unique filtered timestamps:",
    len(filtered_time_values)
)


print(
    "\nOutput file:"
)

print(
    OUTPUT_DEP
)


# ============================================================
# 9. VERIFY OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING OUTPUT")
print("=" * 70)


output_times = set()


for line in kept_lines:

    fields = line.strip().split(",")


    if len(fields) >= 2:

        time_value = fields[1].strip()


        parsed = pd.to_datetime(
            time_value,
            format="%H:%M:%S.%f",
            errors="coerce"
        )


        if pd.notna(parsed):

            normalized_time = (
                parsed.strftime(
                    "%H:%M:%S.%f"
                )[:11]
            )

            output_times.add(
                normalized_time
            )


# Check whether every output timestamp exists
# in the filtered comparison file.

unexpected_times = (
    output_times
    - filtered_time_values
)


print(
    "\nUnique timestamps in output:",
    len(output_times)
)


print(
    "Unexpected timestamps:",
    len(unexpected_times)
)


if len(unexpected_times) == 0:

    print(
        "\nVERIFICATION SUCCESSFUL!"
    )

    print(
        "Every timestamp in the output .dep "
        "exists in the filtered comparison file."
    )

else:

    print(
        "\nWARNING: Unexpected timestamps found."
    )

    print(
        list(unexpected_times)[:10]
    )


print("\nDone!")
