# xb_align/data/build_envfrag_table.py
"""Build environment x fragment co-occurrence energy table."""

import os
from collections import Counter
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

from xb_align.core.env_featurizer import SimpleEnvFeaturizer
from xb_align.data.build_halopos_stats import ALLOWED_ELEMS


def build_envfrag_table(
    drugs_parquet: str,
    out_npz: str,
) -> None:
    """Build Env x Frag co-occurrence log-probability table.

    Args:
        drugs_parquet: Path to drugs_std.parquet file
        out_npz: Path to output NPZ file
    """
    print(f"Reading drugs from: {drugs_parquet}")
    df = pd.read_parquet(drugs_parquet)

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
            key = (env_id, sym)
            counter[key] += 1

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} drugs")

    if not counter:
        raise RuntimeError("No allowed atoms found in drug set.")

    keys = list(counter.keys())
    counts = np.array([counter[k] for k in keys], dtype=np.float64)
    probs = counts / counts.sum()
    log_probs = np.log(probs + 1e-12)

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(
        out_npz,
        keys=np.array(keys, dtype=object),
        log_probs=log_probs,
    )
    print(f"Saved envfrag_table to: {out_npz}")
    print(f"Total unique (env_id, elem) pairs: {len(keys)}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    drugs_parquet = os.path.join(project_root, "data", "processed", "drugs_std.parquet")
    out_npz = os.path.join(project_root, "data", "processed", "envfrag_table.npz")
    build_envfrag_table(drugs_parquet, out_npz)
