#!/usr/bin/env python
# xb_align/scripts/run_macro_baseline.py
"""Run baseline generation and macro alignment evaluation.

This script generates baseline molecules using random doping on NP scaffolds,
ranks them by prior score, and evaluates their macro position distribution
alignment against DrugBank.
"""

import argparse
from pathlib import Path

import pandas as pd

from xb_align.priors.macro_align import (
    load_macro_reference,
    compute_macro_metrics,
    compute_macro_metrics_union,
    debug_cost_matrix,
)
from xb_align.core.env_featurizer import SimpleEnvFeaturizer
from xb_align.baseline.generator import load_np_scaffolds, sample_baseline
from xb_align.baseline.scoring import BaselinePriorRanker
from xb_align.rewards.prior_micro import PriorMicroScorer
from xb_align.eval.macro_eval import (
    compute_baseline_macro_hist,
    count_baseline_descriptors,
    plot_macro_hist_comparison,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline generation and macro alignment evaluation.",
    )
    parser.add_argument(
        "--np-scaffolds",
        type=Path,
        required=True,
        help="Path to np_scaffolds.parquet",
    )
    parser.add_argument(
        "--halopos-ref",
        type=Path,
        required=True,
        help="Path to drug_halopos_ref.npz",
    )
    parser.add_argument(
        "--graph-mlm",
        type=Path,
        required=True,
        help="Path to trained Graph-MLM checkpoint (graph_mlm.pt)",
    )
    parser.add_argument(
        "--envfrag-table",
        type=Path,
        required=True,
        help="Path to envfrag_table.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to save results and plots.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of baseline molecules to generate.",
    )
    parser.add_argument(
        "--max-changes",
        type=int,
        default=5,
        help="Maximum number of substitutions per molecule.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("M2 Baseline Generation and Macro Alignment Evaluation")
    print("=" * 60)

    # 1) Load reference distribution
    print("\n[1/7] Loading reference distribution...")
    ref = load_macro_reference(args.halopos_ref)
    print(f"      Loaded {len(ref.keys)} position descriptors")

    # DIAGNOSTIC: Check cost matrix
    debug_cost_matrix(ref.cost_matrix)

    # 2) Load NP scaffolds and sample baseline molecules
    print("\n[2/7] Loading NP scaffolds...")
    scaffolds = load_np_scaffolds(args.np_scaffolds)
    print(f"      Loaded {len(scaffolds)} scaffolds")

    print(f"\n[3/7] Sampling {args.n_samples} baseline molecules...")
    baseline_samples = sample_baseline(
        scaffolds=scaffolds,
        n_samples=args.n_samples,
        max_changes=args.max_changes,
        seed=args.seed,
    )
    print(f"      Generated {len(baseline_samples)} valid baseline molecules")

    # 3) Build prior scorer and rank baseline samples
    print("\n[4/7] Loading prior scorer...")
    env_featurizer = SimpleEnvFeaturizer()
    prior_scorer = PriorMicroScorer.from_files(
        graph_mlm_path=args.graph_mlm,
        envfrag_path=args.envfrag_table,
    )
    print("      Prior scorer loaded successfully")

    print("\n[5/7] Ranking baseline samples by prior score...")
    ranker = BaselinePriorRanker(prior_scorer=prior_scorer)
    ranked = ranker.rank_samples(baseline_samples)
    print(f"      Ranked {len(ranked)} samples")

    # 4) Keep top-N as the baseline library
    top_k = min(2000, len(ranked))
    top_samples = ranked[:top_k]
    print(f"      Keeping top {top_k} samples for evaluation")

    # 5) Compute macro histogram for baseline (for visualization on ref keys)
    print("\n[6/7] Computing macro alignment metrics...")

    # NEW: Count baseline descriptors without vocabulary restriction
    print("   [6a] Counting position descriptors in baseline samples...")
    baseline_counts = count_baseline_descriptors(
        samples=top_samples,
        env_featurizer=env_featurizer,
        debug=True,
    )

    # NEW: Compute metrics on union support (this fixes EMD=0 issue)
    print("\n   [6b] Computing metrics on union support...")
    metrics = compute_macro_metrics_union(
        baseline_counts=baseline_counts,
        ref=ref,
        epsilon=0.1,
        n_sinkhorn_iters=200,
        env_mismatch_cost=2.0,
        elem_mismatch_cost=1.0,
        kernel_bandwidth=1.0,
        debug=True,
    )

    print("\nMacro alignment metrics (baseline vs DrugBank, union support):")
    print(f"  Sinkhorn EMD : {metrics.emd:.12e}")
    print(f"  MMD^2        : {metrics.mmd2:.6f}")
    print(f"  L2 distance  : {metrics.l2:.6f}")

    # Also compute histogram on ref keys for visualization comparison
    print("\n   [6c] Computing histogram on ref keys for visualization...")
    baseline_hist = compute_baseline_macro_hist(
        samples=top_samples,
        ref=ref,
        env_featurizer=env_featurizer,
        debug=False,
    )

    # DIAGNOSTIC: Check histogram on ref support
    import numpy as np
    print("\n>>> Histogram diagnostics (on ref support only):")
    print(f"    ref.sum        = {ref.freqs.sum():.6f}")
    print(f"    gen.sum        = {baseline_hist.sum():.6f}")
    print(f"    L1 diff        = {np.abs(ref.freqs - baseline_hist).sum():.6f}")
    print(f"    L2 diff        = {np.linalg.norm(ref.freqs - baseline_hist):.6f}")
    if baseline_hist.sum() == 0:
        print("    WARNING: Histogram on ref support is all zeros (100% OOV)")
        print("    This is why we use union support for metrics calculation!")

    # 6) Save plots
    print("\n[7/7] Saving results...")
    plot_path = out_dir / "macro_hist_baseline_vs_drugbank.png"
    plot_macro_hist_comparison(
        ref=ref,
        gen_hist=baseline_hist,
        out_path=plot_path,
        top_k=50,
    )
    print(f"      Saved histogram plot to: {plot_path}")

    # 7) Save CSV of top baseline samples
    df = pd.DataFrame(
        {
            "scaffold_id": [s.scaffold_id for s in top_samples],
            "scaffold_smiles": [s.scaffold_smiles for s in top_samples],
            "generated_smiles": [s.generated_smiles for s in top_samples],
            "changed_atoms": [";".join(map(str, s.changed_atoms)) for s in top_samples],
        }
    )
    csv_path = out_dir / "baseline_top_samples.csv"
    df.to_csv(csv_path, index=False)
    print(f"      Saved top samples to: {csv_path}")

    # Save metrics to text file
    metrics_path = out_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("Macro Alignment Metrics (Baseline vs DrugBank)\n")
        f.write("=" * 50 + "\n")
        f.write("Metrics computed on union support space:\n")
        f.write("(Union = ref keys + baseline-only keys)\n")
        f.write("\n")
        f.write(f"Sinkhorn EMD  : {metrics.emd:.12e}\n")
        f.write(f"MMD^2         : {metrics.mmd2:.6f}\n")
        f.write(f"L2 distance   : {metrics.l2:.6f}\n")
        f.write(f"\nTotal samples generated: {len(baseline_samples)}\n")
        f.write(f"Top samples kept: {top_k}\n")
        f.write(f"\nBaseline unique position descriptors: {len(baseline_counts)}\n")
        f.write(f"Reference position descriptors: {len(ref.keys)}\n")

        # Calculate overlap statistics
        ref_keys_set = set(ref.keys)
        shared = sum(1 for k in baseline_counts.keys() if k in ref_keys_set)
        baseline_only = len(baseline_counts) - shared
        f.write(f"Shared descriptors: {shared}\n")
        f.write(f"Baseline-only descriptors: {baseline_only}\n")
        f.write(f"OOV rate: {baseline_only / len(baseline_counts) * 100:.2f}%\n")
    print(f"      Saved metrics to: {metrics_path}")

    print("\n" + "=" * 60)
    print("Baseline evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
