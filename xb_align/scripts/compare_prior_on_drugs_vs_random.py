# xb_align/scripts/compare_prior_on_drugs_vs_random.py
"""Compare log_prior_micro scores between real drugs and randomly perturbed molecules."""

import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from xb_align.priors.graph_mlm import GraphMLM
from xb_align.priors.envfrag_energy import EnvFragEnergy
from xb_align.priors.pas_energy import PASEnergy
from xb_align.rewards.prior_micro import PriorMicroScorer
from xb_align.priors.atom_vocab import NUM_ATOM_CLASSES, ATOM_TYPES


def load_envfrag_from_npz(path: str) -> EnvFragEnergy:
    """Load EnvFragEnergy model from NPZ file.

    Args:
        path: Path to envfrag_table.npz file

    Returns:
        EnvFragEnergy instance
    """
    data = np.load(path, allow_pickle=True)
    keys = data["keys"]
    log_probs = data["log_probs"]
    table = {}
    for k, lp in zip(keys, log_probs):
        env_id, elem = k  # k is (env_id, elem)
        table[(int(env_id), str(elem))] = float(lp)
    return EnvFragEnergy(table=table, default_logp=-10.0)


def random_perturb_mol(mol: Chem.Mol, num_changes: int = 2):
    """Randomly change atom types at a few positions to another allowed type.

    Args:
        mol: RDKit molecule object
        num_changes: Number of atoms to randomly change

    Returns:
        Tuple of (perturbed molecule, list of changed atom indices)
    """
    rw = Chem.RWMol(mol)
    num_atoms = rw.GetNumAtoms()
    if num_atoms == 0:
        return mol, []

    num_changes = min(num_changes, num_atoms)
    indices = random.sample(range(num_atoms), num_changes)

    changed_indices: List[int] = []
    for idx in indices:
        atom = rw.GetAtomWithIdx(idx)
        old_sym = atom.GetSymbol()
        choices = [e for e in ATOM_TYPES if e != old_sym]
        if not choices:
            continue
        new_sym = random.choice(choices)
        z = Chem.GetPeriodicTable().GetAtomicNumber(new_sym)
        atom.SetAtomicNum(z)
        changed_indices.append(idx)

    new_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
    except Exception:
        # If sanitization fails, return original
        return mol, []
    return new_mol, changed_indices


def main():
    """Main comparison script."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    drugs_parquet = os.path.join(project_root, "data", "processed", "drugs_std.parquet")
    envfrag_npz = os.path.join(project_root, "data", "processed", "envfrag_table.npz")
    gmlm_ckpt = os.path.join(project_root, "data", "processed", "graph_mlm.pt")

    print("Loading data...")
    df = pd.read_parquet(drugs_parquet)
    smiles_list = df["smiles"].tolist()
    random.shuffle(smiles_list)
    smiles_list = smiles_list[:1000]  # Small sample

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gmlm = GraphMLM(num_atom_types=NUM_ATOM_CLASSES, hidden_dim=128)
    if os.path.exists(gmlm_ckpt):
        print(f"Loading Graph-MLM checkpoint from: {gmlm_ckpt}")
        gmlm.load_state_dict(torch.load(gmlm_ckpt, map_location=device))
    else:
        print(f"Warning: Graph-MLM checkpoint not found at {gmlm_ckpt}")
        print("Using untrained model (scores will be meaningless)")
    gmlm.to(device)

    print(f"Loading EnvFrag table from: {envfrag_npz}")
    envfrag = load_envfrag_from_npz(envfrag_npz)

    pas = PASEnergy()
    scorer = PriorMicroScorer(
        graph_mlm=gmlm,
        envfrag_energy=envfrag,
        pas_energy=pas,
        device=device,
        alpha=1.0,
        beta=0.0,
        gamma=1.0,
    )

    real_scores = []
    fake_scores = []

    print("\nComparing real drugs vs randomly perturbed molecules...")
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # For real drug, define changed_atoms as halogen/hetero atoms
        changed_real = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetSymbol() in ("F", "Cl", "Br", "I", "N", "O", "S", "P")
        ]
        if not changed_real:
            continue
        s_real = scorer.log_prior_micro(mol, changed_real)
        real_scores.append(s_real)

        # Randomly perturb the molecule
        try:
            mol_fake, changed_fake = random_perturb_mol(mol, num_changes=3)
        except Exception:
            continue
        if not changed_fake:
            continue
        s_fake = scorer.log_prior_micro(mol_fake, changed_fake)
        fake_scores.append(s_fake)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} molecules...")

    print(f"\nResults:")
    print(f"Real drugs: {len(real_scores)} molecules")
    print(f"Fake (perturbed): {len(fake_scores)} molecules")

    if real_scores and fake_scores:
        print(f"\nMean log_prior_micro (real): {np.mean(real_scores):.3f} +/- {np.std(real_scores):.3f}")
        print(f"Mean log_prior_micro (fake): {np.mean(fake_scores):.3f} +/- {np.std(fake_scores):.3f}")
        print(f"\nDifference: {np.mean(real_scores) - np.mean(fake_scores):.3f}")
        print("\nExpected: Real drugs should have higher (less negative) scores than fake molecules")
    else:
        print("Error: Not enough valid molecules to compare")


if __name__ == "__main__":
    main()
