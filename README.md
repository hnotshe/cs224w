# CS224W Datacenter Placement

## Overview
We are creating a graph such that US counties as the basic node in the graph and enrich each node with environmental, socioeconomic, and infrastructure signals sourced from recent public datasets. The graph is consumed by a Graph Neural Network (GNN) that learns how neighboring counties influence one another when selecting new datacenter locations. The updated pipeline fuses NOAA weather summaries, Census population estimates, housing and distance-to-metro metrics, electricity prices, available water (nice-to-have), and eGRID infrastructure indicators (powerplants, datacenters) before training a two-layer GCN that predicts regional suitability scores.

## Feature Set
- **Environmental:** Long-term NOAA temperature summaries (value, anomaly, mean, percentile rank) plus a pending binary `has_water` feature derived from OSM hydrography around each county centroid.
- **Weather:** Seasonal weather snapshots are appended via `merge_weather_to_counties.py`, letting us plug different months/quantiles into the feature table without changing downstream code.
- **Socio-economic:** Latest Census population estimates (`pop_2020_base`, `pop_2024_est`), median house price pulls, and the distance to the nearest city with >800k people to capture market access and labor availability.
- **Infrastructure:** State-level electricity price (cents/kWh), powerplant counts and capacity (from `powerplants.csv`), datacenter counts/capacity (`datacenters.csv`), and an explicit state identity embedding.
- **Edge Features:** Counties connect to all bordering counties, and every edge carries a binary flag denoting whether the two endpoints belong to the same state. This encourages the GNN to separate intra-state relationships (shared regulation) from cross-state spillovers.

## Data Sources & Processing
- `data/counties.py` downloads the 2024 TIGER/Line county shapefile, computes centroids/geometries, and writes `data/processed/counties.csv` plus `counties.geojson`.
- Population estimates from the Census “co-est2024-pop.csv”-style file are merged via `merge_population_to_counties.py`, producing `data/processed/counties_with_population.csv`.
- NOAA county-level weather summaries are merged in `merge_weather_to_counties.py`, yielding `data/processed/counties_with_weather.csv`.
- `merge_population_and_weather.py` combines the two into `data/processed/counties_merged.csv`, which is the canonical node feature table fed to the trainer.
- `data/processed/powerplants.csv` (eGRID) and `data/processed/datacenters.csv` store infrastructure context for feature engineering—counts per county or per balancing authority can be joined before training.
- `data/OSM_scraper.py` (optional) tags counties that border water bodies for the `has_water` nice-to-have feature.

You can rerun the entire preprocessing stack with:
```
python data/counties.py
python data/merge_population_to_counties.py --counties data/processed/counties.csv --population <census_csv> --out data/processed/counties_with_population.csv
python data/merge_weather_to_counties.py --counties data/processed/counties.csv --weather <noaa_csv> --out data/processed/counties_with_weather.csv --prefix aug_2025_temp_
python data/merge_population_and_weather.py --population data/processed/counties_with_population.csv --weather data/processed/counties_with_weather.csv --out data/processed/counties_merged.csv
```
Augment `counties_merged.csv` with the latest housing, distance, datacenter, and powerplant features before training.

## Graph Construction
- **Nodes:** Each county row in `data/processed/counties_merged.csv` (joined with infrastructure metrics) becomes one node.
- **Edges:** Counties that share a border (derived from the TIGER geometry) are connected, ensuring geographic continuity and allowing information to diffuse along state lines.
- **Edge Metadata:** Every edge stores a `same_state` indicator (1 if both counties share the same state abbreviation, else 0). These features can be passed to PyG layers that accept edge attributes or used for filtering/weighting.
- **Targets:** By default we train on a `suitability_score` column (see `src/train.py --target-column`). Substitute any binary or continuous label column you add to `counties_merged.csv`.

## Model Architecture
`src/models.py` now houses a concise `DataCenterGCN`:
- Two stacked `GCNConv` layers (hidden size configurable via `--hidden`) with ReLU + dropout after each pass capture both local and second-order neighborhood effects.
- A linear prediction head outputs a single logit per county, enabling binary classification or ranking after applying a sigmoid.
- Xavier initialization plus explicit `reset_parameters` keep training stable as we iterate on the expanding feature set.
This setup is a strong baseline that can later be extended with edge-aware convolutions once we begin consuming the `same_state` flag or other relational features.

## Repository Layout
- `data/`: Convenience scripts and storage for fetched datasets (`raw/` for downloads, `processed/` for cleaned artifacts). Large files should be ignored via `.gitignore`.
- `notebooks/`: Colab-friendly notebooks for exploration, graph construction, and model training.
- `src/`: Core Python package containing dataset loaders, the GCN model, utilities, and the training loop.
- `report/`: Project writeups, figures, and the proposal draft.
- `run.sh`: Shortcut script that forwards arguments to the training entry point.

## Getting Started
1. Create a Python 3.11 environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the preprocessing scripts above (or your own notebooks) so that `data/processed/counties_merged.csv` contains the latest environmental + socioeconomic features and any new labels.
4. Generate or update the sparse adjacency / edge-feature matrices that encode bordering counties and the `same_state` bit (store them under `data/processed/`).

## Running Experiments
```
./run.sh --features data/processed/counties_merged.csv \
         --adjacency data/processed/county_adjacency.npz \
         --target-column suitability_score \
         --device cpu
```
Arguments are forwarded to `python -m src.train`; run `./run.sh --help` to see all options, including hidden size, dropout, and checkpoint directory.

## Notes
- Checkpoints are stored in `checkpoints/` by default; override with `--checkpoint-dir`.
- Update the notebooks as you iterate on feature engineering, especially when adding new data sources such as updated NOAA snapshots or revised housing metrics.
- Replace `report/proposal.pdf` with your actual proposal and future milestone reports, and document any new data assumptions there for reproducibility.
