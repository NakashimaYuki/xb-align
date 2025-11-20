# xb_align/core/env_featurizer.py
from typing import List, Tuple
from rdkit import Chem


class SimpleEnvFeaturizer:
    """Simple, deterministic encoding of local atom environment into an integer env_id."""

    def encode(self, mol: Chem.Mol, atom_idx: int) -> int:
        """Encode atom environment as a hash-based integer ID.

        Args:
            mol: RDKit molecule object
            atom_idx: Index of the atom to encode

        Returns:
            Integer environment ID (non-negative)
        """
        atom = mol.GetAtomWithIdx(atom_idx)
        sym = atom.GetSymbol()
        aromatic = atom.GetIsAromatic()
        degree = atom.GetDegree()
        neigh_syms = sorted(n.GetSymbol() for n in atom.GetNeighbors())
        key = (sym, aromatic, degree, tuple(neigh_syms))
        return hash(key) & 0x7FFFFFFF  # keep it non-negative
