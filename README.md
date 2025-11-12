# CS224W Datacenter Placement

## Overview
We model regional interdependencies with Graph Neural Networks to rank candidate locations for new data centers.
Nodes correspond to geographic regions (cities or grid cells) enriched with features such as electricity cost, renewable energy potential, latency to population centers, and climate risk. Edges encode spatial distance and network connectivity. A GNN learns embeddings that capture local and global trade-offs, yielding an overall suitability score for each region and enabling scenario analysis when new data centers are added. We will validate the approach on publicly available datasets (eGRID, aterio) and compare against baselines.

## Repository Layout
- `data/`: Convenience scripts and storage for fetched datasets (`raw/` for downloads, `processed/` for cleaned artifacts). Large files should be ignored via `.gitignore`.
- `notebooks/`: Colab-friendly notebooks for exploration, graph construction, and model training.
- `src/`: Core Python package containing dataset loaders, the GCN model, utilities, and the training loop.
- `report/`: Project writeups, figures, and the proposal draft.
- `run.sh`: Shortcut script that forwards arguments to the training entry point.

## Getting Started
1. Create a Python 3.11 environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Download raw datasets (e.g., eGRID, Aterio, latency data) into `data/raw/` using your preferred scripts.
4. Transform raw inputs into model-ready tables and adjacency matrices under `data/processed/` (e.g., `regions.csv`, `adjacency.npz`).

## Running Experiments
```
./run.sh --features data/processed/regions.csv \
         --adjacency data/processed/adjacency.npz \
         --target-column suitability_score \
         --device cpu
```
Arguments are forwarded to `python -m src.train`; run `./run.sh --help` to see all options, including hidden size, dropout, and checkpoint directory.

## Notes
- Checkpoints are stored in `checkpoints/` by default; override with `--checkpoint-dir`.
- Update the notebooks as you iterate on feature engineering and graph construction.
- Replace `report/proposal.pdf` with your actual proposal and future milestone reports.
