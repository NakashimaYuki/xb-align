# xb_align/priors/envfrag_energy.py
"""Environment x Fragment co-occurrence energy model."""

from typing import Dict, Tuple

from rdkit import Chem

from xb_align.core.env_featurizer import SimpleEnvFeaturizer


class EnvFragEnergy:
    """Simple env x element co-occurrence log-probability table.

    This model scores atom positions based on how frequently a given element
    (e.g., F, Cl, N, O) appears in a given local environment in drug molecules.
    """

    def __init__(self, table: Dict[Tuple[int, str], float], default_logp: float = -10.0):
        """Initialize EnvFragEnergy model.

        Args:
            table: Dictionary mapping (env_id, elem) -> log-probability
            default_logp: Default log-probability for unseen (env_id, elem) pairs
        """
        self.table = table
        self.default_logp = default_logp
        self.featurizer = SimpleEnvFeaturizer()

    def log_score(self, mol: Chem.Mol, atom_indices):
        """Compute log-probability score for given atom positions.

        Args:
            mol: RDKit molecule object
            atom_indices: List of atom indices to score

        Returns:
            Sum of log-probabilities for the given positions
        """
        score = 0.0
        for idx in atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            env_id = self.featurizer.encode(mol, idx)
            elem = atom.GetSymbol()
            key = (env_id, elem)
            score += self.table.get(key, self.default_logp)
        return float(score)
