# xb_align/rewards/prior_micro.py
"""Micro-level position prior scoring combining Graph-MLM, EnvFrag, and PAS."""

from typing import List

import torch
from rdkit import Chem

from xb_align.priors.atom_vocab import ATOM2IDX, MASK_TOKEN_IDX, NUM_ATOM_CLASSES
from xb_align.priors.graph_mlm import GraphMLM
from xb_align.priors.envfrag_energy import EnvFragEnergy
from xb_align.priors.pas_energy import PASEnergy


def graph_mlm_nll(
    mol: Chem.Mol,
    changed_atoms: List[int],
    model: GraphMLM,
    device: torch.device,
    temperature: float = 1.0,
) -> float:
    """Compute negative log-likelihood of true atom types at changed positions.

    Args:
        mol: RDKit molecule object
        changed_atoms: List of atom indices that were modified
        model: Trained Graph-MLM model
        device: PyTorch device
        temperature: Temperature for softmax (lower = sharper distribution)

    Returns:
        Negative log-likelihood (lower is better)
    """
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0 or not changed_atoms:
        return 0.0

    # Extract atom types
    atom_types = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        idx = ATOM2IDX.get(sym, ATOM2IDX["C"])
        atom_types.append(idx)

    x = torch.tensor(atom_types, dtype=torch.long, device=device)

    # Extract edges
    edges_src = []
    edges_dst = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges_src.append(i)
        edges_dst.append(j)
        edges_src.append(j)
        edges_dst.append(i)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device)
    batch = torch.zeros(num_atoms, dtype=torch.long, device=device)

    # Mask changed atoms
    x_masked = x.clone()
    for idx in changed_atoms:
        x_masked[idx] = MASK_TOKEN_IDX

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(x_masked, edge_index, batch)  # [num_nodes, num_classes]
        if temperature != 1.0:
            logits = logits / temperature
        logp = torch.log_softmax(logits, dim=-1)
        idx_tensor = torch.tensor(changed_atoms, dtype=torch.long, device=device)
        true_types = x[idx_tensor]
        nll = -logp[idx_tensor, true_types].sum()

    return float(nll.item())


class PriorMicroScorer:
    """Combine Graph-MLM, EnvFrag and PAS into a micro-level position prior.

    This scorer evaluates how well a set of atom modifications aligns with
    the learned distribution of halogen/heteroatom positions in drug molecules.
    """

    def __init__(
        self,
        graph_mlm: GraphMLM,
        envfrag_energy: EnvFragEnergy,
        pas_energy: PASEnergy,
        device: torch.device,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        temperature: float = 1.0,
    ):
        """Initialize PriorMicroScorer.

        Args:
            graph_mlm: Trained Graph-MLM model
            envfrag_energy: EnvFrag energy model
            pas_energy: PAS energy model
            device: PyTorch device
            alpha: Weight for Graph-MLM NLL term
            beta: Weight for PAS term
            gamma: Weight for EnvFrag term
            temperature: Temperature for Graph-MLM softmax
        """
        self.graph_mlm = graph_mlm.to(device)
        self.envfrag_energy = envfrag_energy
        self.pas_energy = pas_energy
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature

    @classmethod
    def from_files(
        cls,
        graph_mlm_path,
        envfrag_path,
        device=None,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        temperature: float = 1.0,
    ):
        """Load PriorMicroScorer from checkpoint files.

        Args:
            graph_mlm_path: Path to graph_mlm.pt checkpoint
            envfrag_path: Path to envfrag_table.npz file
            device: PyTorch device (defaults to CPU)
            alpha: Weight for Graph-MLM NLL term
            beta: Weight for PAS term
            gamma: Weight for EnvFrag term
            temperature: Temperature for Graph-MLM softmax

        Returns:
            Initialized PriorMicroScorer instance
        """
        from pathlib import Path

        if device is None:
            device = torch.device("cpu")

        # Load Graph-MLM
        # Configuration must match training (see train_graph_mlm.py)
        # GraphMLM(num_atom_types=NUM_ATOM_CLASSES, hidden_dim=128)
        graph_mlm = GraphMLM(
            num_atom_types=NUM_ATOM_CLASSES,
            hidden_dim=128,
        )
        graph_mlm.load_state_dict(torch.load(graph_mlm_path, map_location=device))
        graph_mlm.eval()

        # Load EnvFrag energy
        envfrag_energy = EnvFragEnergy.load(envfrag_path)

        # PAS energy (placeholder)
        pas_energy = PASEnergy()

        return cls(
            graph_mlm=graph_mlm,
            envfrag_energy=envfrag_energy,
            pas_energy=pas_energy,
            device=device,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            temperature=temperature,
        )

    def log_prior_micro(self, mol: Chem.Mol, changed_atoms: List[int]) -> float:
        """Compute log-prior score for modified atoms.

        This method evaluates how well a set of atom modifications aligns with
        the learned distribution of halogen/heteroatom positions in drug molecules.

        Args:
            mol: RDKit molecule object with modifications
            changed_atoms: List of atom indices that were modified from the original scaffold.
                For single-atom substitutions: indices of atoms whose elements were changed
                For fragment attachments: indices of scaffold atoms where fragments were attached
                These indices should correspond to positions in the current molecule structure.

        Returns:
            Combined log-prior score (higher is better). This score combines:
            - Graph-MLM negative log-likelihood (weighted by -alpha)
            - EnvFrag log-probability (weighted by gamma)
            - PAS log-score (weighted by beta)

        Notes:
            The changed_atoms indices must be valid for the provided molecule.
            Empty changed_atoms list will return 0.0.
            This scoring is used for:
            - Ranking baseline-generated molecules
            - Computing GFlowNet rewards (future M3)
            - Evaluating position distribution quality
        """
        if not changed_atoms:
            return 0.0

        # Graph-MLM negative log-likelihood (lower is better)
        nll = graph_mlm_nll(
            mol,
            changed_atoms,
            self.graph_mlm,
            device=self.device,
            temperature=self.temperature,
        )

        # EnvFrag and PAS log-scores (higher is better)
        log_env = self.envfrag_energy.log_score(mol, changed_atoms)
        log_pas = self.pas_energy.log_score(mol, changed_atoms)

        # Combined score: negative NLL + weighted log-probs
        return float(-(self.alpha * nll) + self.beta * log_pas + self.gamma * log_env)
