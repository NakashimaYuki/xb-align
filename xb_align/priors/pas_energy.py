# xb_align/priors/pas_energy.py
"""Position-specific aromatic substitution (PAS) energy model."""

from rdkit import Chem


class PASEnergy:
    """Placeholder for position-specific aromatic substitution energy.

    For M1, this returns zero. Later you can implement ring-position statistics
    based on aromatic ring positions (ortho, meta, para) and substitution patterns.
    """

    def __init__(self):
        """Initialize PAS energy model."""
        pass

    def log_score(self, mol: Chem.Mol, atom_indices):
        """Compute log-score for aromatic position preferences.

        Args:
            mol: RDKit molecule object
            atom_indices: List of atom indices to score

        Returns:
            Log-probability score (currently 0.0 as placeholder)
        """
        return 0.0
