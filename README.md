# Exhibited at University of Toronto Undergraduate Engineering Research Day 2026

Python workflow for extracting acoustic-envelope features, clustering single-beam echo-sounder responses, comparing clustering methods, attaching navigation coordinates, visualizing spatial classes, and checking surveyed bathymetry against a reference surface.

The repository is organized as a **code-first reproducible workflow**. Input datasets are intentionally not versioned; the scripts expect user-supplied files in `data/input/` and write generated products to `results/`.

## Project workflow

```text
Echo-sounder input (.sd)
        |
        v
Feature extraction
Peak amplitude / Mean amplitude / Standard deviation / Energy
        |
        v
Standardization (StandardScaler)
        |
        +--------------------------+
        |                          |
        v                          v
DBSCAN parameter search       PCA visualization
        |                     (visualization only)
        v
DBSCAN clustering
        |
        +--------------------------+
        |                          |
        v                          v
Method comparison          Timestamp + position matching
(K-means, CLARA, FCM,      (.dep navigation file)
 GMM, HDBSCAN, K-medoids)          |
                                   v
                            Spatial cluster map

Survey bathymetry + reference bathymetry
        |
        v
Nearest-neighbour depth comparison
        |
        v
Threshold filtering + comparison map
```

## Main methodology

Each valid acoustic envelope is represented by four features used for clustering:

| Feature | Definition |
|---|---|
| Peak amplitude | Maximum amplitude in the envelope |
| Mean amplitude | Mean envelope amplitude |
| Standard deviation | Standard deviation of the amplitude sequence |
| Energy | Sum of squared amplitudes |

Two additional features—echo length and peak position—were explored during development but were excluded from the final clustering feature space because they can be strongly affected by acquisition geometry and depth. PCA is used only to visualize the standardized feature space; clustering is performed in the original standardized four-dimensional feature space.

The final DBSCAN configuration used in this project is `eps = 0.16` and `min_samples = 1200`, producing four clusters in the documented analysis. The repository also includes scripts for comparing DBSCAN with K-means, CLARA, fuzzy C-means, Gaussian mixture models, HDBSCAN, and K-medoids.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── input/                 # place local input files here
├── docs/
│   └── WORKFLOW.md            # detailed run instructions
├── scripts/
│   ├── clustering/            # feature extraction and clustering
│   ├── mapping/               # coordinate matching and interactive maps
│   ├── validation/            # bathymetry comparison / filtering
│   └── visualization/         # parameter and summary plots
└── results/
    ├── figures/
    ├── maps/
    └── tables/
```

## Quick start

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Then place the required local input files in `data/input/` using the filenames described in [`docs/WORKFLOW.md`](docs/WORKFLOW.md).

A typical clustering run is:

```bash
python scripts/clustering/dbscan_parameter_search.py
python scripts/clustering/dbscan_clustering.py
python scripts/mapping/match_geolocation.py
python scripts/mapping/map_clusters.py
```

For the bathymetry comparison workflow:

```bash
python scripts/mapping/map_reference_bathymetry.py
python scripts/mapping/map_survey_bathymetry.py
python scripts/validation/compare_bathymetry.py
```

See the detailed workflow for parameter locations, expected outputs, and optional filtering steps.

## Notes

- The scripts use relative paths and should be run from the repository root.
- Parameter-search sweeps can be computationally expensive.
- Generated `.csv` and `.html` products are excluded from Git by default to keep the repository lightweight.
- Some clustering scripts expose `n_clusters`, `min_samples`, or related parameters directly near the clustering section for easy experimentation.
