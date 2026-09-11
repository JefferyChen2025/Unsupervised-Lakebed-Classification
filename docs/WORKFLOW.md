# Workflow

This document describes the analysis workflow used in this project.

The project focuses on extracting acoustic waveform features, clustering acoustic responses, comparing clustering methods, linking classified observations with geographic coordinates, and comparing surveyed bathymetry with reference bathymetric data.

---

## 1. Waveform Feature Extraction

The waveform-processing workflow starts by reading echo-sounder waveform records and extracting valid acoustic envelopes.

Each waveform is represented as a sequence of amplitude values.

For each valid waveform, the following features are calculated:

| Feature | Description |
|---|---|
| Peak Amplitude | Maximum amplitude in the waveform |
| Mean Amplitude | Mean amplitude of all waveform samples |
| Standard Deviation | Standard deviation of waveform amplitudes |
| Energy | Sum of squared waveform amplitudes |

Earlier versions of the workflow also considered:

- Echo Length
- Peak Position

These two features were later excluded from the final clustering workflow because they can depend strongly on acquisition geometry and water depth.

---

## 2. Feature Standardization

The four selected waveform features have different numerical ranges.

For example, peak amplitude may remain within a small range while waveform energy may be much larger.

To ensure that each feature contributes more equally to the distance calculation used during clustering, all four features are standardized using `StandardScaler`.

The standardized feature matrix is then used for clustering.

---

## 3. PCA Visualization

Principal Component Analysis is used to generate a two-dimensional representation of the standardized feature space.

PCA is used only for visualization.

The clustering algorithms operate on the standardized four-dimensional feature space rather than on the PCA-reduced coordinates.

This allows the data distribution and resulting clusters to be inspected visually without discarding information before clustering.

---

## 4. DBSCAN Parameter Search

DBSCAN requires two major parameters:

- `eps`
- `min_samples`

The script:

```text
scripts/clustering/dbscan_parameter_search.py
