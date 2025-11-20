# xb_align/data/prepare_drugs.py
"""Standardize drug SMILES and compute physicochemical descriptors."""

import os
from typing import Optional, Dict

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED

from xb_align.data.prepare_np_scaffolds import normalize_smiles, get_scaffold_smiles


def basic_props(mol: Chem.Mol) -> Dict[str, float]:
    """Compute basic drug-like physicochemical properties.

    Args:
        mol: RDKit molecule object

    Returns:
        Dictionary of molecular descriptors
    """
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": int(rdMolDescriptors.CalcNumHBA(mol)),
        "rot_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "fsp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "qed": float(QED.qed(mol)),
    }


def build_drugs_std(
    in_csv: str,
    out_parquet: str,
) -> None:
    """Standardize drug SMILES and compute basic descriptors.

    Args:
        in_csv: Path to input CSV with 'drug_id', 'name', 'smiles' columns
        out_parquet: Path to output parquet file
    """
    print(f"Reading drugs from: {in_csv}")
    df = pd.read_csv(in_csv)

    required_cols = ["drug_id", "name", "smiles"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain '{col}' column.")

    records = []

    for idx, row in df.iterrows():
        drug_id = row["drug_id"]
        name = row["name"]
        raw_smi = row["smiles"]

        smi = normalize_smiles(raw_smi)
        if smi is None:
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        scaf_smi = get_scaffold_smiles(smi)
        if scaf_smi is None:
            continue

        props = basic_props(mol)

        has_halogen = any(a.GetSymbol() in ("F", "Cl", "Br", "I") for a in mol.GetAtoms())
        has_hetero = any(a.GetSymbol() in ("N", "O", "S", "P") for a in mol.GetAtoms())

        rec = {
            "drug_id": str(drug_id),
            "name": str(name),
            "smiles": smi,
            "scaffold_smiles": scaf_smi,
            "has_halogen": bool(has_halogen),
            "has_hetero": bool(has_hetero),
        }
        rec.update(props)
        records.append(rec)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} rows, valid records: {len(records)}")

    if not records:
        raise RuntimeError("No valid drug records found. Check input data.")

    out_df = pd.DataFrame(records)
    out_dir = os.path.dirname(out_parquet)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Writing {len(out_df)} drugs to: {out_parquet}")
    out_df.to_parquet(out_parquet, index=False)
    print("Done.")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    in_csv = os.path.join(project_root, "data", "raw", "drugs_raw.csv")
    out_parquet = os.path.join(project_root, "data", "processed", "drugs_std.parquet")
    build_drugs_std(in_csv, out_parquet)
