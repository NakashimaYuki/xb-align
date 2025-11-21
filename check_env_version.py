#!/usr/bin/env python
"""Check env_featurizer version consistency between current code and npz files."""

import numpy as np
from pathlib import Path
from collections import Counter

from xb_align.core.env_featurizer import SimpleEnvFeaturizer
from xb_align.priors.position_descriptor import PositionDescriptor
from rdkit import Chem
import pandas as pd


def check_halopos_ref_version(npz_path: Path):
    """Check version info in drug_halopos_ref.npz."""
    print(f"\n{'='*60}")
    print(f"Checking: {npz_path}")
    print(f"{'='*60}")

    if not npz_path.exists():
        print(f"ERROR: File not found!")
        return

    data = np.load(npz_path, allow_pickle=True)

    # Check version
    env_version = data.get("env_version", None)
    current_version = SimpleEnvFeaturizer.version()

    print(f"\nVersion Info:")
    print(f"  File version:    {env_version if env_version is not None else 'NOT FOUND'}")
    print(f"  Current version: {current_version}")

    if env_version is None:
        print(f"  Status: WARNING - No version info in file!")
    elif str(env_version) == current_version:
        print(f"  Status: MATCH")
    else:
        print(f"  Status: MISMATCH - Need to rebuild!")

    # Load keys
    keys_arr = data["keys"]
    keys = list(keys_arr.tolist())
    freqs = data["freqs"]

    print(f"\nReference Statistics:")
    print(f"  Total keys: {len(keys)}")

    # Count by element
    elem_counter = Counter(k.elem for k in keys)
    print(f"  Elements: {dict(elem_counter)}")

    # Show sample keys
    print(f"\n  Sample keys (first 5):")
    for i, k in enumerate(keys[:5]):
        print(f"    {i+1}. env_id={k.env_id}, elem={k.elem}, freq={freqs[i]:.6f}")

    return keys, env_version


def check_overlap_with_drugs(ref_keys, drugs_parquet: Path, n_samples = None):
    """Check env_id overlap between reference and current drug data."""
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
            desc = PositionDescriptor(env_id=env_id, elem=sym)
            current_counts[desc] += 1
            total_atoms += 1

    print(f"\nCurrent Drug Statistics:")
    print(f"  Total allowed atoms: {total_atoms}")
    print(f"  Unique descriptors: {len(current_counts)}")

    # Check overlap with reference
    ref_set = set(ref_keys)
    current_set = set(current_counts.keys())

    shared = ref_set & current_set
    ref_only = ref_set - current_set
    current_only = current_set - ref_set

    print(f"\nOverlap Analysis:")
    print(f"  Ref keys:        {len(ref_set)}")
    print(f"  Current keys:    {len(current_set)}")
    print(f"  Shared keys:     {len(shared)} ({len(shared)/len(ref_set)*100:.1f}% of ref)")
    print(f"  Ref-only keys:   {len(ref_only)}")
    print(f"  Current-only:    {len(current_only)}")

    if len(shared) < len(ref_set) * 0.5:
        print(f"\n  Status: LOW OVERLAP - Consider rebuilding drug_halopos_ref.npz!")
    elif len(shared) < len(ref_set) * 0.9:
        print(f"\n  Status: MODERATE OVERLAP - May want to rebuild")
    else:
        print(f"\n  Status: GOOD OVERLAP")

    # Show examples of non-overlapping keys
    if len(current_only) > 0:
        print(f"\n  Sample current-only keys (env_ids not in ref):")
        for i, desc in enumerate(list(current_only)[:5]):
            count = current_counts[desc]
            print(f"    {i+1}. env_id={desc.env_id}, elem={desc.elem}, count={count}")


def main():
    print("\n" + "="*60)
    print("ENV_FEATURIZER VERSION CONSISTENCY CHECK")
    print("="*60)

    # Check paths
    project_root = Path(__file__).parent
    npz_path = project_root / "data" / "processed" / "drug_halopos_ref.npz"
    drugs_parquet = project_root / "data" / "processed" / "drugs_std.parquet"

    # Check halopos ref version
    ref_keys, ref_version = check_halopos_ref_version(npz_path)

    # Check overlap (use all drugs for accurate overlap check)
    check_overlap_with_drugs(ref_keys, drugs_parquet, n_samples=None)

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print("="*60)

    if ref_version is None:
        print("\n1. The drug_halopos_ref.npz file has NO version info.")
        print("   This suggests it was created before version tracking was added.")
        print("\n2. To add version info and ensure consistency, rebuild it:")
        print("   python xb_align/data/build_halopos_stats.py")
    else:
        current_version = SimpleEnvFeaturizer.version()
        if str(ref_version) != current_version:
            print(f"\n1. Version MISMATCH detected!")
            print(f"   File: {ref_version} vs Current: {current_version}")
            print("\n2. You MUST rebuild drug_halopos_ref.npz:")
            print("   python xb_align/data/build_halopos_stats.py")
        else:
            print("\n1. Version info is present and matches current code. OK")
            print("2. If overlap is low, env_ids may have changed due to hashing.")
            print("   Consider rebuilding if you made changes to SimpleEnvFeaturizer.encode()")


if __name__ == "__main__":
    main()
