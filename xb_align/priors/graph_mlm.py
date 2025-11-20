# xb_align/priors/graph_mlm.py
"""Graph-based masked atom-type language model."""

from typing import Tuple

import torch
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool


class GraphMLM(nn.Module):
    """Simple graph-based masked atom-type language model.

    Uses GIN (Graph Isomorphism Network) layers to encode molecular structure
    and predict masked atom types.
    """

    def __init__(self, num_atom_types: int, hidden_dim: int = 128):
        """Initialize Graph-MLM model.

        Args:
            num_atom_types: Total number of atom type classes (including mask token)
            hidden_dim: Hidden dimension for GNN layers
        """
        super().__init__()
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)

        nn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)

        self.out_head = nn.Linear(hidden_dim, num_atom_types)

    def forward(
        self,
        x_atom_type: torch.Tensor,  # [num_nodes]
        edge_index: torch.Tensor,   # [2, num_edges]
        batch: torch.Tensor,        # [num_nodes]
    ) -> torch.Tensor:
        """Forward pass to predict atom types.

        Args:
            x_atom_type: Atom type indices (may include masked tokens)
            edge_index: Edge connectivity
            batch: Batch assignment for each node

        Returns:
            Per-node logits over atom types [num_nodes, num_atom_types]
        """
        h = self.atom_embedding(x_atom_type)  # [num_nodes, hidden_dim]

        h = self.conv1(h, edge_index)
        h = self.norm1(h)
        h = torch.relu(h)

        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = torch.relu(h)

        logits = self.out_head(h)
        return logits
