"""Utility helpers for preprocessing, evaluation, and visualization."""

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_features(frame: pd.DataFrame, feature_ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Scale selected columns into [0, 1] based on provided (min, max) ranges."""

    normalized = frame.copy()
    for column, (min_val, max_val) in feature_ranges.items():
        if column not in normalized:
            raise KeyError(f"Feature '{column}' missing from dataframe")
        span = max_val - min_val
        if span == 0:
            raise ValueError(f"Feature '{column}' has zero span; cannot normalize")
        normalized[column] = (normalized[column] - min_val) / span
    return normalized


def plot_suitability_map(coordinates: np.ndarray, scores: np.ndarray, save_path: Path | None = None) -> None:
    """Render a scatter plot summarizing regional suitability scores."""

    if coordinates.shape[1] != 2:
        raise ValueError("Coordinates must be 2-dimensional for plotting")
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=scores, cmap="viridis", s=40)
    plt.colorbar(scatter, label="Suitability Score")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Datacenter Suitability Map")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def train_val_test_split(items: Iterable, ratios: Tuple[float, float, float]) -> Tuple[list, list, list]:
    """Split an iterable into train, validation, and test subsets."""

    train_ratio, val_ratio, test_ratio = ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    items = list(items)
    n = len(items)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    return items[:train_end], items[train_end:val_end], items[val_end:]
