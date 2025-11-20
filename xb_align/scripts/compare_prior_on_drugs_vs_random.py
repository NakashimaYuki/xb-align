# xb_align/scripts/compare_prior_on_drugs_vs_random.py
"""Compare log_prior_micro scores between real drugs and randomly perturbed molecules.

This script ensures fair comparison by:
1. Selecting k random positions in each molecule
2. Creating a perturbed version with different atoms at those positions
3. Comparing prior scores at the SAME positions for both real and perturbed
"""

import os
import random
from typing import List, Optional, Tuple

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


def perturb_at_positions(mol: Chem.Mol, positions: List[int]) -> Optional[Chem.Mol]:
    """Create a perturbed molecule by changing atoms at specified positions.

    Args:
        mol: Original molecule
        positions: List of atom indices to modify

    Returns:
        Perturbed molecule or None if sanitization fails
    """
    rw = Chem.RWMol(mol)

    for idx in positions:
        atom = rw.GetAtomWithIdx(idx)
        old_sym = atom.GetSymbol()
        # Choose a different atom type from ATOM_TYPES
        choices = [e for e in ATOM_TYPES if e != old_sym]
        if not choices:
            continue
        new_sym = random.choice(choices)
        z = Chem.GetPeriodicTable().GetAtomicNumber(new_sym)
        atom.SetAtomicNum(z)

    new_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
        return new_mol
    except Exception:
        return None


def compare_at_same_positions(
    mol: Chem.Mol,
    scorer: PriorMicroScorer,
    k: int = 5,
) -> Optional[Tuple[float, float, float]]:
    """Compare real vs perturbed molecule at the same k positions.

    Args:
        mol: Original molecule
        scorer: Prior scorer
        k: Number of positions to compare

    Returns:
        Tuple of (s_real, s_fake, delta) or None if comparison failed
    """
    # Select k candidate positions (prefer C/N/O/S atoms for more meaningful comparison)
    candidates = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() in ("C", "N", "O", "S", "P")
    ]

    if len(candidates) < k:
        return None

    positions = random.sample(candidates, k)

    # Create perturbed molecule at these positions
    mol_fake = perturb_at_positions(mol, positions)
    if mol_fake is None:
        return None

    # Calculate scores at the SAME positions
    try:
        s_real = scorer.log_prior_micro(mol, positions)
        s_fake = scorer.log_prior_micro(mol_fake, positions)
    except Exception:
        return None

    delta = s_real - s_fake
    return (s_real, s_fake, delta)


def main():
    """Main comparison script with fair evaluation."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    drugs_parquet = os.path.join(project_root, "data", "processed", "drugs_std.parquet")
    envfrag_npz = os.path.join(project_root, "data", "processed", "envfrag_table.npz")
    gmlm_ckpt = os.path.join(project_root, "data", "processed", "graph_mlm.pt")

    print("Loading data...")
    df = pd.read_parquet(drugs_parquet)
    smiles_list = df["smiles"].tolist()
    random.shuffle(smiles_list)
    smiles_list = smiles_list[:1000]  # Sample for faster evaluation

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
    deltas = []

    print("\nComparing real drugs vs perturbed molecules at SAME positions...")
    print("Each comparison uses the same k=5 positions for both real and fake\n")

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        result = compare_at_same_positions(mol, scorer, k=5)
        if result is None:
            continue

        s_real, s_fake, delta = result
        real_scores.append(s_real)
        fake_scores.append(s_fake)
        deltas.append(delta)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} molecules, valid pairs: {len(deltas)}")

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Number of valid comparison pairs: {len(deltas)}")

    if deltas:
        print(f"\nMean log_prior_micro (real):      {np.mean(real_scores):>8.3f} +/- {np.std(real_scores):.3f}")
        print(f"Mean log_prior_micro (fake):      {np.mean(fake_scores):>8.3f} +/- {np.std(fake_scores):.3f}")
        print(f"\nMean(delta = real - fake):        {np.mean(deltas):>8.3f}")
        print(f"Std(delta):                       {np.std(deltas):>8.3f}")

        fraction_positive = np.mean(np.array(deltas) > 0)
        print(f"\nFraction(delta > 0):              {fraction_positive:>8.3f} ({fraction_positive*100:.1f}%)")

        print(f"\n{'='*60}")
        print("INTERPRETATION")
        print(f"{'='*60}")
        if np.mean(deltas) > 0 and fraction_positive > 0.5:
            print("SUCCESS: Real drugs have higher prior scores than random perturbations")
            print("The model has learned meaningful position preferences from DrugBank")
        else:
            print("WARNING: Prior scores do not clearly favor real drugs")
            print("Consider adjusting alpha/gamma weights or retraining Graph-MLM")
    else:
        print("Error: Not enough valid molecules to compare")


if __name__ == "__main__":
    main()
