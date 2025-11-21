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

    @staticmethod
    def version() -> str:
        """Return the version identifier for EnvFragEnergy table format.

        This version tracks the env_featurizer used and the table construction logic.
        Change this version if you modify the featurizer or table building process.

        Version history:
        - v1.0: Initial implementation with env_id from SimpleEnvFeaturizer v1.0 (non-deterministic)
        - v1.1: Updated to use SimpleEnvFeaturizer v1.1 (deterministic MD5 hash)
        """
        # Version should match the env_featurizer version it depends on
        return f"envfrag_{SimpleEnvFeaturizer.version()}"

    def __init__(self, table: Dict[Tuple[int, str], float], default_logp: float = -10.0):
        """Initialize EnvFragEnergy model.

        Args:
            table: Dictionary mapping (env_id, elem) -> log-probability
            default_logp: Default log-probability for unseen (env_id, elem) pairs
        """
        self.table = table
        self.default_logp = default_logp
        self.featurizer = SimpleEnvFeaturizer()

    @classmethod
    def load(cls, npz_path, default_logp: float = -10.0):
        """Load EnvFragEnergy from NPZ file.

        Args:
            npz_path: Path to envfrag_table.npz file
            default_logp: Default log-probability for unseen pairs

        Returns:
            EnvFragEnergy instance

        Raises:
            ValueError: If envfrag table version mismatch is detected
        """
        import numpy as np
        from pathlib import Path
        import warnings

        data = np.load(npz_path, allow_pickle=True)
        keys = data["keys"].tolist()  # List of (env_id, elem) tuples
        log_probs = data["log_probs"]

        # Version checking: ensure env_featurizer compatibility
        table_version = data.get("table_version", None)
        current_version = cls.version()

        if table_version is not None:
            # Cast to string in case it's stored as numpy object
            table_version_str = str(table_version) if not isinstance(table_version, str) else table_version
            if table_version_str != current_version:
                raise ValueError(
                    f"EnvFrag table version mismatch:\n"
                    f"  Table file:  {table_version_str}\n"
                    f"  Current code: {current_version}\n"
                    f"Rebuild {npz_path} with current env_featurizer using build_envfrag_table.py"
                )
        else:
            # Warn if no version is found (old format)
            warnings.warn(
                f"No table version found in {npz_path}. "
                f"This file may be incompatible with current code (version {current_version}). "
                f"Consider rebuilding with build_envfrag_table.py",
                UserWarning
            )

        table = {tuple(k): float(lp) for k, lp in zip(keys, log_probs)}
        return cls(table=table, default_logp=default_logp)

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
