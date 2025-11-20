# xb_align/data/prepare_np_scaffolds.py
"""Extract and deduplicate NP scaffolds from CNPD-ETCM merged data."""

import os
from typing import Optional, List, Dict

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors


def normalize_smiles(smi: str) -> Optional[str]:
    """Normalize raw SMILES into canonical isomeric SMILES. Return None if invalid."""
    if not isinstance(smi, str):
        return None
    # Filter out 'SDF' placeholders (case-insensitive)
    if smi.strip().upper() == 'SDF':
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def get_scaffold_smiles(smi: str) -> Optional[str]:
    """Extract Bemis-Murcko scaffold SMILES from a molecule SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaf, isomericSmiles=True)


def calc_scaffold_props(scaffold_smiles: str) -> Dict[str, float]:
    """Compute simple scaffold-level descriptors."""
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None:
        return {
            "fsp3": None,
            "n_rings": None,
            "n_atoms": None,
        }
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_atoms = mol.GetNumAtoms()
    return {
        "fsp3": float(fsp3),
        "n_rings": int(n_rings),
        "n_atoms": int(n_atoms),
    }


def build_np_scaffolds(
    xlsx_path: str,
    out_parquet: str,
    max_ids_per_scaffold: int = 20,
) -> None:
    """Build NP scaffold table from CNPD-ETCM merged Excel file.

    Args:
        xlsx_path: Path to input Excel file with 'Smiles' and 'ID' columns
        out_parquet: Path to output parquet file
        max_ids_per_scaffold: Maximum number of source IDs to store per scaffold
    """
    print(f"Reading NP Excel from: {xlsx_path}")
    print(f"File size: {os.path.getsize(xlsx_path) / 1024 / 1024:.2f} MB")
    df = pd.read_excel(xlsx_path)

    if "Smiles" not in df.columns:
        raise ValueError("Input Excel must contain 'Smiles' column.")
    if "ID" not in df.columns:
        raise ValueError("Input Excel must contain 'ID' column.")

    print(f"Loaded {len(df)} rows")
    print("Starting SMILES normalization and scaffold extraction...")

    records: List[Dict[str, str]] = []
    skipped_sdf = 0
    skipped_invalid = 0
    skipped_no_scaffold = 0

    for idx, row in df.iterrows():
        raw_smi = row["Smiles"]
        np_id = row["ID"]
        name = row.get("Name", "")

        # Check for SDF placeholder
        if isinstance(raw_smi, str) and raw_smi.strip().upper() == 'SDF':
            skipped_sdf += 1
            if (idx + 1) % 10000 == 0:
                print(f"Processed {idx + 1}/{len(df)} rows, valid: {len(records)}, skipped SDF: {skipped_sdf}, invalid: {skipped_invalid}, no scaffold: {skipped_no_scaffold}")
            continue

        smi = normalize_smiles(raw_smi)
        if smi is None:
            skipped_invalid += 1
            if (idx + 1) % 10000 == 0:
                print(f"Processed {idx + 1}/{len(df)} rows, valid: {len(records)}, skipped SDF: {skipped_sdf}, invalid: {skipped_invalid}, no scaffold: {skipped_no_scaffold}")
            continue

        scaf_smi = get_scaffold_smiles(smi)
        if scaf_smi is None:
            skipped_no_scaffold += 1
            if (idx + 1) % 10000 == 0:
                print(f"Processed {idx + 1}/{len(df)} rows, valid: {len(records)}, skipped SDF: {skipped_sdf}, invalid: {skipped_invalid}, no scaffold: {skipped_no_scaffold}")
            continue

        records.append(
            {
                "np_id": str(np_id),
                "name": str(name),
                "smiles": smi,
                "scaffold_smiles": scaf_smi,
            }
        )

        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1}/{len(df)} rows, valid: {len(records)}, skipped SDF: {skipped_sdf}, invalid: {skipped_invalid}, no scaffold: {skipped_no_scaffold}")

    if not records:
        raise RuntimeError("No valid NP records with scaffolds. Check input data.")

    print(f"\n" + "="*80)
    print(f"Processing complete!")
    print(f"  Total input rows: {len(df)}")
    print(f"  Skipped SDF placeholders: {skipped_sdf} ({skipped_sdf/len(df)*100:.1f}%)")
    print(f"  Skipped invalid SMILES: {skipped_invalid} ({skipped_invalid/len(df)*100:.1f}%)")
    print(f"  Skipped no scaffold: {skipped_no_scaffold} ({skipped_no_scaffold/len(df)*100:.1f}%)")
    print(f"  Valid records with scaffolds: {len(records)} ({len(records)/len(df)*100:.1f}%)")
    print("="*80 + "\n")

    sdf = pd.DataFrame(records)
    print(f"Deduplicating scaffolds...")

    # Group by scaffold and aggregate
    groups = []
    for scaf, g in sdf.groupby("scaffold_smiles"):
        example = g["smiles"].iloc[0]
        n_members = len(g)
        ids = ";".join(map(str, g["np_id"].iloc[:max_ids_per_scaffold]))

        props = calc_scaffold_props(scaf)

        rec = {
            "scaffold_smiles": scaf,
            "example_smiles": example,
            "n_members": int(n_members),
            "source_ids": ids,
        }
        rec.update(props)
        groups.append(rec)

    out_df = pd.DataFrame(groups)
    out_dir = os.path.dirname(out_parquet)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Writing {len(out_df)} scaffolds to: {out_parquet}")
    out_df.to_parquet(out_parquet, index=False)
    print("Done.")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Try to find the CNPD-ETCM file (with or without Chinese characters)
    xlsx_candidates = [
        "CNPD-ETCM-合并去重.xlsx",  # Original Chinese name
        "CNPD_ETCM_merged.xlsx",      # English fallback
    ]

    xlsx_path = None
    for candidate in xlsx_candidates:
        test_path = os.path.join(project_root, "data", "raw", candidate)
        if os.path.exists(test_path):
            xlsx_path = test_path
            break

    if xlsx_path is None:
        raise FileNotFoundError(
            f"Could not find CNPD-ETCM data file. Tried: {xlsx_candidates}\n"
            f"Please place your file in: {os.path.join(project_root, 'data', 'raw')}"
        )

    out_parquet = os.path.join(project_root, "data", "processed", "np_scaffolds.parquet")
    build_np_scaffolds(xlsx_path, out_parquet)
