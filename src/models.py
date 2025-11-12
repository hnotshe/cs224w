"""Model definitions for datacenter node classification with GNNs.

Predicts which geographic regions are most likely to host datacenters,
given spatial, economic, and environmental features. This is framed as
a binary node classification task on a geographic graph where:
- y_i = 1 if region i hosts one or more datacenters
- y_i = 0 otherwise

Each node represents a region (e.g., city, county, or grid cell), and
edges capture spatial or geopolitical relationships between regions.
"""

from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class DataCenterGCN(nn.Module):
    """Two-layer GCN for binary node classification of datacenter presence.

    The model learns from existing datacenter placements to infer what
    combinations of features and neighborhood context historically make
    a region suitable. Through message passing, each region aggregates
    contextual information from its neighbors, capturing spatial and
    structural correlations that tabular models cannot.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # Binary classification: output single logit per node
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
        """Return binary classification logits for every node in the graph.

        Args:
            x: Node feature matrix of shape [num_nodes, in_channels]
            edge_index: Graph connectivity in COO format [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            Logits tensor of shape [num_nodes] for binary classification.
            Apply sigmoid to get probabilities, or use BCEWithLogitsLoss
            during training.
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.predict_head(x)
        return logits.squeeze(-1)
