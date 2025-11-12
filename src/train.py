"""Training utilities for optimizing datacenter placement models."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import scipy.sparse as sp
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader

from .datasets import prepare_dataset
from .models import DataCenterGCN


@dataclass
class TrainingConfig:
    in_channels: int
    hidden_channels: int = 64
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    max_epochs: int = 200
    device: str = "cpu"
    checkpoint_dir: Path = Path("checkpoints")


class Trainer:
    """Encapsulates the training/evaluation loop for DataCenterGCN models."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = DataCenterGCN(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            dropout=config.dropout,
        ).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        self.criterion = torch.nn.MSELoss()
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._evaluate(val_loader) if val_loader is not None else train_loss
            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                break

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(batch.x, batch.edge_index, getattr(batch, "edge_weight", None))
            loss = self.criterion(preds, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item()) * batch.num_graphs
        return total_loss / max(1, len(loader.dataset))

    @torch.no_grad()
    def _evaluate(self, loader: Optional[DataLoader]) -> float:
        if loader is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(self.device)
            preds = self.model(batch.x, batch.edge_index, getattr(batch, "edge_weight", None))
            loss = self.criterion(preds, batch.y)
            total_loss += float(loss.item()) * batch.num_graphs
        return total_loss / max(1, len(loader.dataset))

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        checkpoint_path = self.config.checkpoint_dir / f"epoch_{epoch:03d}_loss_{val_loss:.4f}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)


def train_model(
    dataset: Iterable[Data],
    config: TrainingConfig,
    val_split: float = 0.2,
    batch_size: int = 32,
) -> Trainer:
    """Convenience wrapper to train on an in-memory dataset."""

    data_list = list(dataset)
    split_idx = int((1 - val_split) * len(data_list))
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    trainer = Trainer(config)
    trainer.fit(train_loader, val_loader)
    return trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DataCenterGCN on processed datasets")
    parser.add_argument("--features", type=Path, required=True, help="Path to processed feature CSV")
    parser.add_argument(
        "--adjacency",
        type=Path,
        required=True,
        help="Path to sparse adjacency matrix stored as .npz",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="suitability_score",
        help="Column name containing suitability labels",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of samples used for validation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device identifier (e.g. cpu, cuda, cuda:1)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=64,
        help="Number of hidden channels in the GCN",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="L2 regularization strength",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to store model checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    adjacency = sp.load_npz(args.adjacency)
    graph = prepare_dataset(
        features_path=args.features,
        adjacency_matrix=adjacency,
        target_column=args.target_column,
        feature_ranges=None,
    )

    config = TrainingConfig(
        in_channels=graph.num_features,
        hidden_channels=args.hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer = train_model([graph], config=config, val_split=args.val_split, batch_size=args.batch_size)
    print("Training complete. Best checkpoint stored in", config.checkpoint_dir.resolve())


if __name__ == "__main__":
    main()
