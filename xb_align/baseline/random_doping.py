# xb_align/baseline/random_doping.py
"""Random single-atom doping on molecular scaffolds."""

from typing import List, Optional, Sequence, Tuple
import random

from rdkit import Chem


ALLOWED_SUB_ELEMS = ("N", "O", "S", "P", "F", "Cl", "Br", "I")


def random_single_atom_doping(
    scaffold_smiles: str,
    max_changes: int = 5,
    rng: Optional[random.Random] = None,
) -> Tuple[Optional[Chem.Mol], List[int]]:
    """Perform random single-atom substitutions on a scaffold molecule.

    Only uses simple SetAtomicNum + Sanitize to ensure chemical validity.
    This is a baseline approach that does not use GFlowNet or learned policies.

    Args:
        scaffold_smiles: SMILES string of the NP scaffold
        max_changes: Maximum number of atom substitutions to apply
        rng: Random number generator for reproducibility

    Returns:
        Tuple of:
        - new_mol: Resulting molecule if at least one valid change is made,
                   otherwise None
        - changed_atoms: List of atom indices that were successfully modified
    """
    if rng is None:
        rng = random

    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None:
        return None, []

    rw = Chem.RWMol(mol)
    atom_indices = list(range(rw.GetNumAtoms()))
    rng.shuffle(atom_indices)

    n_changes = rng.randint(1, max_changes)
    changed_atoms: List[int] = []

    pt = Chem.GetPeriodicTable()

    for idx in atom_indices:
        if len(changed_atoms) >= n_changes:
            break

        atom = rw.GetAtomWithIdx(idx)
        current_sym = atom.GetSymbol()

        # Only consider a small set of host atoms
        if current_sym not in ("C", "N", "O", "S", "P"):
            continue

        # Sample a new element different from the current one
        candidates = [e for e in ALLOWED_SUB_ELEMS if e != current_sym]
        if not candidates:
            continue
        new_elem = rng.choice(candidates)

        new_z = pt.GetAtomicNumber(new_elem)
        old_z = pt.GetAtomicNumber(current_sym)

        # Try the substitution
        atom.SetAtomicNum(new_z)

        try:
            candidate = rw.GetMol()
            Chem.SanitizeMol(candidate)
        except Exception:
            # Revert the change
            atom.SetAtomicNum(old_z)
            continue

        # Accept the change
        mol = candidate
        rw = Chem.RWMol(mol)
        changed_atoms.append(idx)

    if not changed_atoms:
        return None, []

    return mol, changed_atoms
