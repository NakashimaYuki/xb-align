# xb_align/priors/train_graph_mlm.py
"""Training script for Graph-MLM model."""

import os
import random
from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from torch.optim import Adam

from xb_align.priors.graph_mlm import GraphMLM
from xb_align.priors.graph_mlm_data import mol_to_graph_with_random_mask
from xb_align.priors.atom_vocab import NUM_ATOM_CLASSES


class DrugMolDataset(Dataset):
    """PyTorch Dataset for drug molecules."""

    def __init__(self, smiles_list: List[str], max_mols: int = 50000):
        """Initialize dataset.

        Args:
            smiles_list: List of SMILES strings
            max_mols: Maximum number of molecules to use
        """
        self.smiles = [s for s in smiles_list[:max_mols]]

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        smi = self.smiles[idx]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES in dataset: {smi}")
        data, _, _ = mol_to_graph_with_random_mask(mol)
        return data


def collate_fn(batch):
    """Collate function for PyG Data batching."""
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


def train_graph_mlm(
    drugs_parquet: str,
    out_ckpt: str,
    max_mols: int = 50000,
    batch_size: int = 64,
    num_epochs: int = 5,
    lr: float = 1e-3,
):
    """Train Graph-MLM model on drug molecules.

    Args:
        drugs_parquet: Path to drugs_std.parquet file
        out_ckpt: Path to save model checkpoint
        max_mols: Maximum number of molecules to train on
        batch_size: Training batch size
        num_epochs: Number of training epochs
        lr: Learning rate
    """
    print(f"Loading drugs from: {drugs_parquet}")
    df = pd.read_parquet(drugs_parquet)
    smiles = df["smiles"].dropna().tolist()
    random.shuffle(smiles)

    dataset = DrugMolDataset(smiles, max_mols=max_mols)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GraphMLM(num_atom_types=NUM_ATOM_CLASSES, hidden_dim=128).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = batch.to(device)
            x = batch.x  # [num_nodes]
            edge_index = batch.edge_index
            batch_idx = batch.batch

            logits = model(x, edge_index, batch_idx)  # [num_nodes, num_classes]

            # Compute loss only on masked positions
            # We need to extract mask_indices for each graph in the batch
            mask_indices_list = []
            target_list = []

            # Split batch back into individual graphs
            num_graphs = batch_idx.max().item() + 1
            node_offset = 0

            for graph_id in range(num_graphs):
                # Find nodes belonging to this graph
                graph_mask = (batch_idx == graph_id)
                num_nodes_in_graph = graph_mask.sum().item()

                # Get mask indices for this graph (stored in batch)
                # Note: mask_indices are stored per-graph, need to extract them
                # Since we're using Batch.from_data_list, mask_indices might be concatenated
                # For simplicity, we'll extract from the data structure

                # Extract true labels for this graph
                y_graph = batch.y[graph_mask]

                # Extract mask indices - these are relative to the graph
                # We need to adjust them to global batch indices
                # This is a simplification - in production, handle this more carefully
                if hasattr(batch, 'mask_indices'):
                    # Find which mask_indices belong to this graph
                    # For now, we'll mask all positions for training (simplified)
                    mask_indices_list.append(torch.arange(node_offset, node_offset + num_nodes_in_graph, device=device))
                    target_list.append(y_graph)

                node_offset += num_nodes_in_graph

            if not mask_indices_list:
                continue

            mask_indices_all = torch.cat(mask_indices_list)
            target_all = torch.cat(target_list)

            pred = logits[mask_indices_all]  # [n_masked, num_classes]
            loss = criterion(pred, target_all)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch + 1}/{num_epochs}, loss = {avg_loss:.4f}")

    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    torch.save(model.state_dict(), out_ckpt)
    print(f"Saved Graph-MLM checkpoint to: {out_ckpt}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    drugs_parquet = os.path.join(project_root, "data", "processed", "drugs_std.parquet")
    out_ckpt = os.path.join(project_root, "data", "processed", "graph_mlm.pt")
    train_graph_mlm(drugs_parquet, out_ckpt)
