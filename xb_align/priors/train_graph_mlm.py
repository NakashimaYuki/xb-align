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
from torch_geometric.data import Batch

from xb_align.priors.graph_mlm import GraphMLM
from xb_align.priors.graph_mlm_data import mol_to_graph_with_random_mask
from xb_align.priors.atom_vocab import NUM_ATOM_CLASSES


class DrugMolDataset(Dataset):
    """Simple dataset of drug molecules for Graph-MLM training."""

    def __init__(self, smiles_list: List[str], max_mols: int = 50000, mask_ratio: float = 0.15):
        self.smiles = [s for s in smiles_list[:max_mols]]
        self.mask_ratio = mask_ratio

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        smi = self.smiles[idx]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError("Invalid SMILES in dataset.")
        data, _ = mol_to_graph_with_random_mask(mol, mask_ratio=self.mask_ratio)
        return data


def collate_fn(batch):
    return Batch.from_data_list(batch)


def train_graph_mlm(
    drugs_parquet: str,
    out_ckpt: str,
    max_mols: int = 50000,
    batch_size: int = 64,
    num_epochs: int = 5,
    lr: float = 1e-3,
):
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
            y_true = batch.y  # [num_nodes]
            mask = batch.mask  # [num_nodes] boolean

            logits = model(x, edge_index, batch_idx)  # [num_nodes, num_classes]

            # only compute loss on masked positions
            if mask.sum() == 0:
                continue
            pred = logits[mask]
            target = y_true[mask]

            loss = criterion(pred, target)

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
