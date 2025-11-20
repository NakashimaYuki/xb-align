# xb_align/priors/graph_mlm_data.py
"""Data utilities for Graph-MLM training."""

from typing import List, Tuple
import random

import torch
from rdkit import Chem
from torch_geometric.data import Data

from xb_align.priors.atom_vocab import ATOM2IDX, MASK_TOKEN_IDX, NUM_ATOM_CLASSES


def mol_to_graph_with_random_mask(
    mol: Chem.Mol,
    mask_ratio: float = 0.15,
) -> Tuple[Data, List[int], List[int]]:
    """Convert RDKit Mol to PyG Data and randomly mask a subset of atom types.

    Args:
        mol: RDKit molecule object
        mask_ratio: Fraction of atoms to mask

    Returns:
        Tuple of (PyG Data object, original atom types, mask indices)
    """
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        raise ValueError("Empty molecule.")

    # Extract atom types
    atom_types = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        idx = ATOM2IDX.get(sym, ATOM2IDX["C"])  # default to C if unknown
        atom_types.append(idx)

    x = torch.tensor(atom_types, dtype=torch.long)

    # Extract edges (undirected)
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

    # Create masked input
    x_masked = x.clone()
    for mi in mask_indices:
        x_masked[mi] = MASK_TOKEN_IDX

    # Create PyG Data object
    data = Data(
        x=x_masked,
        edge_index=edge_index,
    )
    data.y = x  # true atom types
    data.mask_indices = torch.tensor(mask_indices, dtype=torch.long)

    return data, atom_types, mask_indices
