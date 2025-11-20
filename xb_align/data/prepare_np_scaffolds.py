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
    df = pd.read_excel(xlsx_path)

    if "Smiles" not in df.columns:
        raise ValueError("Input Excel must contain 'Smiles' column.")
    if "ID" not in df.columns:
        raise ValueError("Input Excel must contain 'ID' column.")

    records: List[Dict[str, str]] = []

    for idx, row in df.iterrows():
        raw_smi = row["Smiles"]
        np_id = row["ID"]
        name = row.get("Name", "")

        smi = normalize_smiles(raw_smi)
        if smi is None:
            continue

        scaf_smi = get_scaffold_smiles(smi)
        if scaf_smi is None:
            continue

        records.append(
            {
                "np_id": str(np_id),
                "name": str(name),
                "smiles": smi,
                "scaffold_smiles": scaf_smi,
            }
        )

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} rows, valid records: {len(records)}")

    if not records:
        raise RuntimeError("No valid NP records with scaffolds. Check input data.")

    sdf = pd.DataFrame(records)
    print(f"Total valid NP records with scaffolds: {len(sdf)}")

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
    xlsx_path = os.path.join(project_root, "data", "raw", "CNPD_ETCM_merged.xlsx")
    out_parquet = os.path.join(project_root, "data", "processed", "np_scaffolds.parquet")
    build_np_scaffolds(xlsx_path, out_parquet)
