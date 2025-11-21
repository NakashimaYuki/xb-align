#!/usr/bin/env python
# xb_align/scripts/analyze_macro_distributions.py
"""Unified script for analyzing and comparing macro position distributions.

This script provides a centralized way to compare position distributions from
multiple sources (DrugBank, Baseline, GFlowNet, etc.) using EMD, MMD, and L2 metrics.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem

from xb_align.priors.macro_align import (
    load_macro_reference,
    histogram_from_changed_atoms,
    compute_macro_metrics,
    MacroAlignReference,
)
from xb_align.core.env_featurizer import SimpleEnvFeaturizer
from xb_align.baseline.generator import BaselineSample


def load_baseline_samples(csv_path: Path) -> List[BaselineSample]:
    """Load baseline samples from CSV file.

    Args:
        csv_path: Path to baseline_top_samples.csv

    Returns:
        List of BaselineSample objects
    """
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        changed_atoms_str = row["changed_atoms"]
        changed_atoms = [int(x) for x in changed_atoms_str.split(";") if x]
        samples.append(
            BaselineSample(
                scaffold_id=int(row["scaffold_id"]),
                scaffold_smiles=row["scaffold_smiles"],
                generated_smiles=row["generated_smiles"],
                changed_atoms=changed_atoms,
            )
        )
    return samples


def compute_distribution_from_baseline(
    samples: List[BaselineSample],
    ref: MacroAlignReference,
    env_featurizer: SimpleEnvFeaturizer,
) -> np.ndarray:
    """Compute position distribution histogram from baseline samples."""
    mol_changed = []
    for s in samples:
        mol = Chem.MolFromSmiles(s.generated_smiles)
        if mol is None:
            continue
        mol_changed.append((mol, s.changed_atoms))

    hist = histogram_from_changed_atoms(
        samples=mol_changed,
        env_featurizer=env_featurizer,
        ref=ref,
    )
    return hist


def plot_multi_distribution_comparison(
    ref: MacroAlignReference,
    distributions: Dict[str, np.ndarray],
    out_path: Path,
    top_k: int = 50,
):
    """Plot comparison of multiple distributions.

    Args:
        ref: MacroAlignReference with keys and reference distribution
        distributions: Dict mapping source names to histograms
        out_path: Path to save the plot
        top_k: Number of top bins to show
    """
    # Sort by reference frequency
    indices = np.argsort(ref.freqs)[::-1]
    if top_k < len(ref.freqs):
        indices = indices[:top_k]

    x = np.arange(len(indices))
    labels = [f"{ref.keys[i].env_id}:{ref.keys[i].elem}" for i in indices]

    # Setup plot
    fig, ax = plt.subplots(figsize=(max(10, len(indices) * 0.2), 5))

    # Calculate bar width based on number of distributions
    n_dists = len(distributions) + 1  # +1 for reference
    width = 0.8 / n_dists

    # Plot reference
    offset = -(n_dists - 1) * width / 2
    ax.bar(
        x + offset,
        ref.freqs[indices],
        width,
        label="DrugBank (reference)",
        alpha=0.8,
    )

    # Plot each distribution
    for i, (name, hist) in enumerate(distributions.items(), start=1):
        offset = -(n_dists - 1) * width / 2 + i * width
        ax.bar(
            x + offset,
            hist[indices],
            width,
            label=name,
            alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Position Descriptor (env_id:element)")
    ax.legend()
    ax.set_title("Position Distribution Comparison")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare macro position distributions from multiple sources.",
    )
    parser.add_argument(
        "--halopos-ref",
        type=Path,
        required=True,
        help="Path to drug_halopos_ref.npz",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        action="append",
        help="Path to baseline_top_samples.csv (can specify multiple)",
    )
    parser.add_argument(
        "--baseline-names",
        type=str,
        action="append",
        help="Names for each baseline (must match number of --baseline-csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to save analysis results",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Macro Distribution Analysis")
    print("=" * 70)

    # Load reference
    print("\n[1/4] Loading reference distribution...")
    ref = load_macro_reference(args.halopos_ref)
    print(f"      Loaded {len(ref.keys)} position descriptors")

    # Load baseline distributions
    print("\n[2/4] Loading baseline distributions...")
    env_featurizer = SimpleEnvFeaturizer()
    distributions = {}

    if args.baseline_csv:
        if args.baseline_names and len(args.baseline_names) != len(args.baseline_csv):
            raise ValueError("Number of baseline names must match number of baseline CSVs")

        names = args.baseline_names or [f"Baseline {i+1}" for i in range(len(args.baseline_csv))]

        for csv_path, name in zip(args.baseline_csv, names):
            print(f"      Loading {name} from {csv_path}...")
            samples = load_baseline_samples(csv_path)
            hist = compute_distribution_from_baseline(samples, ref, env_featurizer)
            distributions[name] = hist
            print(f"      Loaded {len(samples)} samples")

    # Compute metrics
    print("\n[3/4] Computing metrics...")
    results = []
    for name, hist in distributions.items():
        metrics = compute_macro_metrics(hist, ref)
        results.append({
            "Source": name,
            "EMD": f"{metrics.emd:.6f}",
            "MMD2": f"{metrics.mmd2:.6f}",
            "L2": f"{metrics.l2:.6f}",
        })
        print(f"      {name}:")
        print(f"        EMD   : {metrics.emd:.6f}")
        print(f"        MMD^2 : {metrics.mmd2:.6f}")
        print(f"        L2    : {metrics.l2:.6f}")

    # Save metrics table
    if results:
        df_results = pd.DataFrame(results)
        results_path = out_dir / "comparison_metrics.csv"
        df_results.to_csv(results_path, index=False)
        print(f"\n      Saved metrics table to: {results_path}")

    # Generate plots
    print("\n[4/4] Generating plots...")
    if distributions:
        plot_path = out_dir / "distribution_comparison.png"
        plot_multi_distribution_comparison(ref, distributions, plot_path, top_k=50)
        print(f"      Saved comparison plot to: {plot_path}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
