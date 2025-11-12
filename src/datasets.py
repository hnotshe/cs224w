"""Dataset loading and graph construction utilities."""

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

from .utils import normalize_features


def load_region_features(path: Path) -> pd.DataFrame:
    """Load raw region-level features (cost, latency, climate risk, etc.)."""

    if not path.exists():
        raise FileNotFoundError(f"Missing features file: {path}")
    return pd.read_csv(path)


def build_graph_from_adjacency(features: pd.DataFrame, adjacency_matrix) -> Data:
    """Combine tabular features with adjacency information into a PyG graph."""

    if adjacency_matrix.shape[0] != len(features):
        raise ValueError("Adjacency matrix size must match number of regions")

    edge_index, edge_weight = from_scipy_sparse_matrix(adjacency_matrix)
    x = torch.tensor(features.values, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    return data


def prepare_dataset(
    features_path: Path,
    adjacency_matrix,
    target_column: str,
    feature_ranges: Optional[dict] = None,
) -> Data:
    """High-level helper to fetch data and return a graph ready for training."""

    features = load_region_features(features_path)
    if feature_ranges:
        features = normalize_features(features, feature_ranges)

    if target_column not in features:
        raise KeyError(f"Target column '{target_column}' missing from features table")

    targets = torch.tensor(features[target_column].values, dtype=torch.float)
    node_features = features.drop(columns=[target_column])

    graph = build_graph_from_adjacency(node_features, adjacency_matrix)
    graph.y = targets
    return graph
