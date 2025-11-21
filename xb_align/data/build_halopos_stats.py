# xb_align/data/build_halopos_stats.py
"""Build reference distribution of halogen/heteroatom positions from drug data."""

import os
from collections import Counter
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from rdkit import Chem

from xb_align.priors.position_descriptor import PositionDescriptor
from xb_align.core.env_featurizer import SimpleEnvFeaturizer


ALLOWED_ELEMS = ("F", "Cl", "Br", "I", "N", "O", "S", "P")


def build_drug_halopos_ref(
    drugs_parquet: str,
    out_npz: str,
) -> None:
    """Build reference distribution of (env_id, elem) over drug molecules.

    Args:
        drugs_parquet: Path to drugs_std.parquet file
        out_npz: Path to output NPZ file
    """
    print(f"Reading drugs from: {drugs_parquet}")
    df = pd.read_parquet(drugs_parquet)

    if "smiles" not in df.columns:
        raise ValueError("drugs_std.parquet must contain 'smiles' column.")

    featurizer = SimpleEnvFeaturizer()
    counter: Counter = Counter()

    for idx, row in df.iterrows():
        smi = row["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym not in ALLOWED_ELEMS:
                continue
            env_id = featurizer.encode(mol, atom.GetIdx())
            desc = PositionDescriptor(env_id=env_id, elem=sym)
            counter[desc] += 1

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} drugs, positions counted: {len(counter)}")

    if not counter:
        raise RuntimeError("No allowed atoms found in drug set.")

    keys: List[PositionDescriptor] = list(counter.keys())
    counts = np.array([counter[k] for k in keys], dtype=np.float64)
    freqs = counts / counts.sum()

    # Save as numpy arrays; PositionDescriptor list saved via np.object dtype
    # Include env_featurizer version for compatibility checking
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    env_version = SimpleEnvFeaturizer.version()
    np.savez(
        out_npz,
        keys=np.array(keys, dtype=object),
        freqs=freqs,
        env_version=env_version,
    )
    print(f"Saved drug_halopos_ref to: {out_npz}")
    print(f"Total unique (env_id, elem) pairs: {len(keys)}")
    print(f"Env featurizer version: {env_version}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    drugs_parquet = os.path.join(project_root, "data", "processed", "drugs_std.parquet")
    out_npz = os.path.join(project_root, "data", "processed", "drug_halopos_ref.npz")
    build_drug_halopos_ref(drugs_parquet, out_npz)
