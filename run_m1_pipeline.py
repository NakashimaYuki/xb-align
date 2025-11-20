#!/usr/bin/env python
"""Quick-start script to run the complete M1 pipeline.

This script runs all M1 steps in sequence:
1. Extract NP scaffolds
2. Standardize drug data
3. Build position reference distribution
4. Build Env x Frag table
5. Train Graph-MLM model
6. Compare prior scores (real vs random)
"""

import os
import sys

def run_step(step_name, module_path):
    """Run a pipeline step."""
    print("\n" + "="*80)
    print(f"Step: {step_name}")
    print("="*80)

    import importlib
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, 'main'):
            module.main()
        print(f"[SUCCESS] {step_name} completed")
    except Exception as e:
        print(f"[ERROR] {step_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

def main():
    """Run complete M1 pipeline."""
    print("XB-Align M1 Pipeline")
    print("="*80)
    print("This will run all M1 steps to prepare data and train priors.")
    print("Expected time: 5-15 minutes depending on data size and CPU.")
    print("="*80)

    steps = [
        ("1. Extract NP Scaffolds", "xb_align.data.prepare_np_scaffolds"),
        ("2. Standardize Drug Data", "xb_align.data.prepare_drugs"),
        ("3. Build Position Reference", "xb_align.data.build_halopos_stats"),
        ("4. Build Env x Frag Table", "xb_align.data.build_envfrag_table"),
        ("5. Train Graph-MLM Model", "xb_align.priors.train_graph_mlm"),
        ("6. Compare Prior Scores", "xb_align.scripts.compare_prior_on_drugs_vs_random"),
    ]

    for step_name, module_path in steps:
        success = run_step(step_name, module_path)
        if not success:
            print("\n[FAILED] Pipeline stopped due to error")
            return 1

    print("\n" + "="*80)
    print("[SUCCESS] M1 Pipeline completed successfully!")
    print("="*80)
    print("\nGenerated files:")
    print("  - data/processed/np_scaffolds.parquet")
    print("  - data/processed/drugs_std.parquet")
    print("  - data/processed/drug_halopos_ref.npz")
    print("  - data/processed/envfrag_table.npz")
    print("  - data/processed/graph_mlm.pt")
    print("\nNext steps:")
    print("  - Review the comparison results above")
    print("  - Run tests: pytest")
    print("  - Move on to M2: GFlowNet implementation")
    return 0

if __name__ == "__main__":
    sys.exit(main())
