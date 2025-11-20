"""Convert DrugBank cleaned CSV into standard drugs_raw.csv format.

Expected input: data/raw/drugbank_data_cleaned.csv
Required column: 'smiles' or 'SMILES'
Output: data/raw/drugs_raw.csv with columns: drug_id, name, smiles
"""

import os
import pandas as pd


def build_drugs_raw_from_drugbank(in_csv: str, out_csv: str) -> None:
    """Convert DrugBank CSV to standardized drugs_raw.csv format.

    Args:
        in_csv: Path to drugbank_data_cleaned.csv
        out_csv: Output path for drugs_raw.csv
    """
    print(f"Reading DrugBank data from: {in_csv}")
    df = pd.read_csv(in_csv)

    # Try to locate SMILES column
    smiles_col = None
    for candidate in ["smiles", "SMILES"]:
        if candidate in df.columns:
            smiles_col = candidate
            break
    if smiles_col is None:
        raise ValueError("Input CSV must contain a 'smiles' or 'SMILES' column.")

    smiles = df[smiles_col].astype(str)

    # Drop obviously invalid placeholders
    mask_valid = ~smiles.str.strip().str.upper().isin(["SDF", "NAN", "NONE", ""])
    smiles = smiles[mask_valid].reset_index(drop=True)

    out_df = pd.DataFrame(
        {
            "drug_id": [f"DB_{i+1:05d}" for i in range(len(smiles))],
            "name": [f"DrugBank_{i+1:05d}" for i in range(len(smiles))],
            "smiles": smiles,
        }
    )

    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Writing standardized drugs_raw.csv with {len(out_df)} entries to: {out_csv}")
    out_df.to_csv(out_csv, index=False)
    print("Done.")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    in_csv = os.path.join(project_root, "data", "raw", "drugbank_data_cleaned.csv")
    out_csv = os.path.join(project_root, "data", "raw", "drugs_raw.csv")
    build_drugs_raw_from_drugbank(in_csv, out_csv)
