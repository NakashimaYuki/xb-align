# xb_align/baseline/generator.py
"""Baseline molecule generator from NP scaffolds."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from rdkit import Chem

from xb_align.baseline.random_doping import random_single_atom_doping


@dataclass
class BaselineSample:
    """A single baseline-generated molecule.

    Attributes:
        scaffold_id: Index of the scaffold in the scaffold library
        scaffold_smiles: SMILES of the original NP scaffold
        generated_smiles: SMILES of the generated molecule after doping
        changed_atoms: List of atom indices that were modified
    """
    scaffold_id: int
    scaffold_smiles: str
    generated_smiles: str
    changed_atoms: List[int]


def load_np_scaffolds(path: Path) -> pd.DataFrame:
    """Load NP scaffold library from parquet file.

    Expected columns:
    - scaffold_smiles: SMILES string of the scaffold
    - example_smiles: Example molecule with this scaffold
    - n_members: Number of molecules with this scaffold
    - Additional metadata columns (fsp3, n_rings, etc.)

    Args:
        path: Path to np_scaffolds.parquet file

    Returns:
        DataFrame containing scaffold library
    """
    return pd.read_parquet(path)


def sample_baseline(
    scaffolds: pd.DataFrame,
    n_samples: int,
    max_changes: int = 5,
    seed: Optional[int] = None,
    max_attempts: Optional[int] = None,
) -> List[BaselineSample]:
    """Sample baseline molecules by randomly doping NP scaffolds.

    This is a simple baseline that does not use GFlowNet. It uniformly
    samples scaffolds and applies random single-atom substitutions.

    Args:
        scaffolds: DataFrame with 'scaffold_smiles' column
        n_samples: Number of baseline molecules to generate
        max_changes: Maximum number of substitutions per molecule
        seed: Random seed for reproducibility
        max_attempts: Maximum number of attempts before giving up.
            If None, defaults to n_samples * 10 to prevent infinite loops.

    Returns:
        List of BaselineSample objects (may be fewer than n_samples if max_attempts reached)
    """
    import random

    rng = random.Random(seed)
    samples: List[BaselineSample] = []

    if "scaffold_smiles" not in scaffolds.columns:
        raise ValueError("scaffolds DataFrame must contain 'scaffold_smiles' column")

    scaffold_indices = list(scaffolds.index)

    if not scaffold_indices:
        return samples

    # Safety guard: prevent infinite loops
    if max_attempts is None:
        max_attempts = n_samples * 10

    attempts = 0
    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1

        scaffold_id = rng.choice(scaffold_indices)
        row = scaffolds.loc[scaffold_id]
        scaf_smi = row["scaffold_smiles"]

        mol, changed_atoms = random_single_atom_doping(
            scaffold_smiles=scaf_smi,
            max_changes=max_changes,
            rng=rng,
        )
        if mol is None or not changed_atoms:
            continue

        gen_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        samples.append(
            BaselineSample(
                scaffold_id=int(scaffold_id),
                scaffold_smiles=scaf_smi,
                generated_smiles=gen_smi,
                changed_atoms=list(changed_atoms),
            )
        )

    if len(samples) < n_samples:
        import warnings
        warnings.warn(
            f"Only generated {len(samples)}/{n_samples} samples after {attempts} attempts. "
            f"Consider increasing max_attempts or using easier scaffolds.",
            UserWarning
        )

    return samples
