# xb_align/priors/graph_mlm_data.py
"""Data utilities for Graph-MLM training."""
from typing import Tuple
import random

import torch
from rdkit import Chem
from torch_geometric.data import Data

from xb_align.priors.atom_vocab import ATOM2IDX, MASK_TOKEN_IDX


def mol_to_graph_with_random_mask(
    mol: Chem.Mol,
    mask_ratio: float = 0.15,
) -> Tuple[Data, int]:
    """Convert RDKit Mol to PyG Data and randomly mask a subset of atom types.

    Returns:
        data: PyG Data with fields:
            - x: masked atom-type indices
            - y: true atom-type indices
            - edge_index: undirected edges
            - mask: boolean mask over nodes indicating which positions were masked
        num_masked: number of masked atoms
    """
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        raise ValueError("Empty molecule.")

    atom_types = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        idx = ATOM2IDX.get(sym, ATOM2IDX["C"])
        atom_types.append(idx)

    x_true = torch.tensor(atom_types, dtype=torch.long)

    edges_src = []
    edges_dst = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges_src.append(i)
        edges_dst.append(j)
        edges_src.append(j)
        edges_dst.append(i)
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    # Randomly choose masked indices
    num_mask = max(1, int(mask_ratio * num_atoms))
    mask_indices = random.sample(range(num_atoms), num_mask)

    x_masked = x_true.clone()
    mask = torch.zeros(num_atoms, dtype=torch.bool)
    for mi in mask_indices:
        x_masked[mi] = MASK_TOKEN_IDX
        mask[mi] = True

    data = Data(
        x=x_masked,
        edge_index=edge_index,
        y=x_true,
        mask=mask,
    )
    return data, num_mask
