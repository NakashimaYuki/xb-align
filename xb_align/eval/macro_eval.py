# xb_align/eval/macro_eval.py
"""Macro alignment evaluation and visualization."""

from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # Ensure non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem

from xb_align.priors.macro_align import (
    MacroAlignReference,
    histogram_from_mols,
    count_position_descriptors,
    ALLOWED_ELEMS,
)
from xb_align.core.env_featurizer import SimpleEnvFeaturizer
from xb_align.baseline.generator import BaselineSample


def compute_baseline_macro_hist(
    samples: List[BaselineSample],
    ref: MacroAlignReference,
    env_featurizer: SimpleEnvFeaturizer,
    debug: bool = False,
) -> np.ndarray:
    """Compute the macro histogram for baseline samples using ALL allowed atoms.

    This function builds a position distribution histogram from baseline
    samples by counting ALL atoms of allowed types (F, Cl, Br, I, N, O, S, P),
    not just the changed atoms. This is necessary because NP scaffold environments
    may be very different from DrugBank, causing OOV issues with changed_atoms only.

    Args:
        samples: List of BaselineSample objects
        ref: MacroAlignReference with keys and mapping
        env_featurizer: Object to compute env_id for atoms
        debug: If True, enable debugging output

    Returns:
        Normalized histogram array of shape [K]
    """
    mols = []
    total_allowed_atoms = 0
    for s in samples:
        mol = Chem.MolFromSmiles(s.generated_smiles)
        if mol is None:
            continue
        mols.append(mol)

        # Count allowed atoms for debugging
        if debug:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() in ALLOWED_ELEMS:
                    total_allowed_atoms += 1

    if debug:
        print(f">>> compute_baseline_macro_hist: processing {len(mols)} molecules")
        print(f"    Total allowed atoms across all molecules: {total_allowed_atoms}")
        if total_allowed_atoms == 0:
            print("    WARNING: NO allowed atoms found in generated molecules!")

    hist = histogram_from_mols(
        mols=mols,
        env_featurizer=env_featurizer,
        ref=ref,
        allowed_elems=ALLOWED_ELEMS,
    )
    return hist


def count_baseline_descriptors(
    samples: List[BaselineSample],
    env_featurizer: SimpleEnvFeaturizer,
    debug: bool = False,
):
    """Count position descriptors for baseline samples without reference vocabulary.

    This function counts all (env_id, element) pairs in baseline samples,
    regardless of whether they appear in a reference distribution. This is
    essential for building union support spaces when comparing distributions
    with non-overlapping vocabularies.

    Args:
        samples: List of BaselineSample objects
        env_featurizer: Object to compute env_id for atoms
        debug: If True, enable debugging output

    Returns:
        Dictionary mapping PositionDescriptor to count
    """
    from typing import Dict
    from xb_align.priors.position_descriptor import PositionDescriptor

    mols = []
    for s in samples:
        mol = Chem.MolFromSmiles(s.generated_smiles)
        if mol is not None:
            mols.append(mol)

    counts = count_position_descriptors(
        mols=mols,
        env_featurizer=env_featurizer,
        allowed_elems=ALLOWED_ELEMS,
    )

    if debug:
        print(f">>> count_baseline_descriptors: processed {len(mols)} molecules")
        print(f"    Unique position descriptors: {len(counts)}")
        print(f"    Total atom counts: {sum(counts.values())}")
        if len(counts) > 0:
            # Show sample descriptors
            sample_descs = list(counts.items())[:5]
            print(f"    Sample descriptors:")
            for desc, count in sample_descs:
                print(f"      {desc.env_id}:{desc.elem} -> {count}")

    return counts


def plot_macro_hist_comparison(
    ref: MacroAlignReference,
    gen_hist: np.ndarray,
    out_path: Path,
    top_k: int = 50,
):
    """Plot side-by-side bar plots for reference vs generated histograms.

    Args:
        ref: MacroAlignReference with keys and frequencies
        gen_hist: Generated histogram of shape [K]
        out_path: Path to save the figure (PNG/SVG)
        top_k: Only plot the top_k bins sorted by reference frequency
    """
    ref_freqs = ref.freqs
    K = ref_freqs.shape[0]

    if gen_hist.shape[0] != K:
        raise ValueError("gen_hist shape does not match reference histogram")

    # Sort by reference frequency descending
    indices = np.argsort(ref_freqs)[::-1]
    if top_k < K:
        indices = indices[:top_k]

    x = np.arange(len(indices))
    ref_values = ref_freqs[indices]
    gen_values = gen_hist[indices]

    labels = [
        f"{ref.keys[i].env_id}:{ref.keys[i].elem}" for i in indices
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(indices) * 0.15), 4))
    width = 0.4

    ax.bar(x - width / 2, ref_values, width, label="DrugBank reference")
    ax.bar(x + width / 2, gen_values, width, label="Baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
