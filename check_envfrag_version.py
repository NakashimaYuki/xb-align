#!/usr/bin/env python
"""Check EnvFragEnergy table version consistency."""

import numpy as np
from pathlib import Path
from collections import Counter

from xb_align.priors.envfrag_energy import EnvFragEnergy
from xb_align.core.env_featurizer import SimpleEnvFeaturizer
from rdkit import Chem
import pandas as pd


def check_envfrag_table_version(npz_path: Path):
    """Check version info in envfrag_table.npz."""
    print(f"\n{'='*60}")
    print(f"Checking: {npz_path}")
    print(f"{'='*60}")

    if not npz_path.exists():
        print(f"ERROR: File not found!")
        return None, None

    data = np.load(npz_path, allow_pickle=True)

    # Check version
    table_version = data.get("table_version", None)
    current_version = EnvFragEnergy.version()

    print(f"\nVersion Info:")
    print(f"  File version:    {table_version if table_version is not None else 'NOT FOUND'}")
    print(f"  Current version: {current_version}")

    if table_version is None:
        print(f"  Status: WARNING - No version info in file!")
    elif str(table_version) == current_version:
        print(f"  Status: MATCH")
    else:
        print(f"  Status: MISMATCH - Need to rebuild!")

    # Load keys
    keys_arr = data["keys"]
    keys = list(keys_arr.tolist())
    log_probs = data["log_probs"]

    print(f"\nTable Statistics:")
    print(f"  Total keys: {len(keys)}")

    # Count by element
    elem_counter = Counter(k[1] for k in keys)
    print(f"  Elements: {dict(elem_counter)}")

    # Show sample keys
    print(f"\n  Sample keys (first 5):")
    for i, (k, lp) in enumerate(zip(keys[:5], log_probs[:5])):
        env_id, elem = k
        print(f"    {i+1}. env_id={env_id}, elem={elem}, log_prob={lp:.6f}")

    return keys, table_version


def check_overlap_with_drugs(table_keys, drugs_parquet: Path, n_samples=None):
    """Check env_id overlap between table and current drug data."""
    print(f"\n{'='*60}")
    print(f"Checking overlap with current env_featurizer on drugs")
    print(f"{'='*60}")

    if not drugs_parquet.exists():
        print(f"ERROR: {drugs_parquet} not found!")
        return

    df = pd.read_parquet(drugs_parquet)
    print(f"\nLoaded {len(df)} drugs from {drugs_parquet}")

    # Sample drugs
    if n_samples is None:
        print(f"Using all {len(df)} drugs for overlap check...")
        df_sample = df
    elif len(df) > n_samples:
        print(f"Sampling {n_samples} drugs for overlap check...")
        df_sample = df.sample(n=n_samples, random_state=42)
    else:
        print(f"Using all {len(df)} drugs for overlap check...")
        df_sample = df

    featurizer = SimpleEnvFeaturizer()
    ALLOWED_ELEMS = ("F", "Cl", "Br", "I", "N", "O", "S", "P")

    # Count descriptors in current drugs
    current_counts = Counter()
    total_atoms = 0

    for idx, row in df_sample.iterrows():
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
            current_counts[key] += 1
            total_atoms += 1

    print(f"\nCurrent Drug Statistics:")
    print(f"  Total allowed atoms: {total_atoms}")
    print(f"  Unique (env_id, elem) pairs: {len(current_counts)}")

    # Check overlap with table (convert lists to tuples for hashing)
    table_set = set(tuple(k) if isinstance(k, list) else k for k in table_keys)
    current_set = set(current_counts.keys())

    shared = table_set & current_set
    table_only = table_set - current_set
    current_only = current_set - table_set

    print(f"\nOverlap Analysis:")
    print(f"  Table keys:      {len(table_set)}")
    print(f"  Current keys:    {len(current_set)}")
    print(f"  Shared keys:     {len(shared)} ({len(shared)/len(table_set)*100:.1f}% of table)")
    print(f"  Table-only keys: {len(table_only)}")
    print(f"  Current-only:    {len(current_only)}")

    if len(shared) < len(table_set) * 0.5:
        print(f"\n  Status: LOW OVERLAP - Consider rebuilding envfrag_table.npz!")
    elif len(shared) < len(table_set) * 0.9:
        print(f"\n  Status: MODERATE OVERLAP - May want to rebuild")
    else:
        print(f"\n  Status: GOOD OVERLAP")

    # Show examples of non-overlapping keys
    if len(current_only) > 0:
        print(f"\n  Sample current-only keys (not in table):")
        for i, key in enumerate(list(current_only)[:5]):
            env_id, elem = key
            count = current_counts[key]
            print(f"    {i+1}. env_id={env_id}, elem={elem}, count={count}")


def main():
    print("\n" + "="*60)
    print("ENVFRAG TABLE VERSION CONSISTENCY CHECK")
    print("="*60)

    # Check paths
    project_root = Path(__file__).parent
    npz_path = project_root / "data" / "processed" / "envfrag_table.npz"
    drugs_parquet = project_root / "data" / "processed" / "drugs_std.parquet"

    # Check envfrag table version
    table_keys, table_version = check_envfrag_table_version(npz_path)

    if table_keys is None:
        return

    # Check overlap (use all drugs for accurate overlap check)
    check_overlap_with_drugs(table_keys, drugs_parquet, n_samples=None)

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print("="*60)

    if table_version is None:
        print("\n1. The envfrag_table.npz file has NO version info.")
        print("   This suggests it was created before version tracking was added.")
        print("\n2. To add version info and ensure consistency, rebuild it:")
        print("   python -m xb_align.data.build_envfrag_table")
    else:
        current_version = EnvFragEnergy.version()
        if str(table_version) != current_version:
            print(f"\n1. Version MISMATCH detected!")
            print(f"   File: {table_version} vs Current: {current_version}")
            print("\n2. You MUST rebuild envfrag_table.npz:")
            print("   python -m xb_align.data.build_envfrag_table")
        else:
            print("\n1. Version info is present and matches current code. OK")
            print("2. If overlap is low, env_ids may have changed due to hashing.")
            print("   Consider rebuilding if you made changes to SimpleEnvFeaturizer.encode()")


if __name__ == "__main__":
    main()
