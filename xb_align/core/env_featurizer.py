# xb_align/core/env_featurizer.py
import hashlib
from typing import List, Tuple
from rdkit import Chem


class SimpleEnvFeaturizer:
    """Simple, deterministic encoding of local atom environment into an integer env_id."""

    @staticmethod
    def version() -> str:
        """Return the version identifier for this featurizer implementation.

        This version string encodes the featurization logic (symbol, aromatic, degree, neighbors).
        Change this version if you modify the encode() method to ensure compatibility checking.

        Version history:
        - v1.0: Initial implementation (sym, aromatic, degree, sorted neighbor symbols) - NON-DETERMINISTIC
        - v1.1: Fixed to use deterministic hash (md5) instead of Python's hash()
        """
        return "simple_env_v1.1"

    def encode(self, mol: Chem.Mol, atom_idx: int) -> int:
        """Encode atom environment as a hash-based integer ID.

        Args:
            mol: RDKit molecule object
            atom_idx: Index of the atom to encode

        Returns:
            Integer environment ID (non-negative, deterministic across runs)
        """
        atom = mol.GetAtomWithIdx(atom_idx)
        sym = atom.GetSymbol()
        aromatic = atom.GetIsAromatic()
        degree = atom.GetDegree()
        neigh_syms = sorted(n.GetSymbol() for n in atom.GetNeighbors())

        # Create a string representation of the environment
        key_str = f"{sym}_{aromatic}_{degree}_{'_'.join(neigh_syms)}"

        # Use MD5 for deterministic hashing across Python sessions
        hash_obj = hashlib.md5(key_str.encode('utf-8'))
        # Take first 8 bytes and convert to int
        hash_bytes = hash_obj.digest()[:8]
        hash_int = int.from_bytes(hash_bytes, byteorder='big')
        # Keep it positive and within 32-bit range for compatibility
        return hash_int & 0x7FFFFFFF
