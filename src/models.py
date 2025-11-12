"""Model definitions for datacenter suitability scoring with GNNs."""

from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class DataCenterGCN(nn.Module):
    """Two-layer GCN that produces regional suitability scores."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.predict_head = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        nn.init.xavier_uniform_(self.predict_head.weight)
        if self.predict_head.bias is not None:
            nn.init.zeros_(self.predict_head.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return suitability logits for every node in the graph."""

        x = self.conv1(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.dropout(x)
        scores = self.predict_head(x)
        return scores.squeeze(-1)
