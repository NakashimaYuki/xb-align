# xb_align/priors/macro_align.py
"""Macro-level alignment of halogen/heteroatom position distributions.

This module provides tools for computing and comparing position distributions
across molecule collections using optimal transport and kernel-based metrics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from rdkit import Chem

from xb_align.priors.position_descriptor import PositionDescriptor
from xb_align.core.env_featurizer import SimpleEnvFeaturizer


ALLOWED_ELEMS = ("F", "Cl", "Br", "I", "N", "O", "S", "P")


@dataclass
class MacroAlignReference:
    """Reference distribution over (env_id, element) descriptors.

    Attributes:
        keys: List of PositionDescriptor objects
        freqs: Normalized frequency array of shape [K], sum to 1
        key_to_index: Mapping from PositionDescriptor to array index
        cost_matrix: Cost matrix for Sinkhorn-EMD of shape [K, K]
        kernel_matrix: Kernel matrix for MMD of shape [K, K]
    """
    keys: List[PositionDescriptor]
    freqs: np.ndarray
    key_to_index: Dict[PositionDescriptor, int]
    cost_matrix: np.ndarray
    kernel_matrix: np.ndarray


@dataclass
class MacroAlignMetrics:
    """Macro alignment metrics between distributions.

    Attributes:
        emd: Sinkhorn-regularized Earth Mover's Distance
        mmd2: Squared Maximum Mean Discrepancy
        l2: Simple L2 distance between histograms
    """
    emd: float
    mmd2: float
    l2: float


def _build_cost_matrix(
    keys: Sequence[PositionDescriptor],
    env_mismatch_cost: float = 2.0,
    elem_mismatch_cost: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """Build a simple cost matrix C[i, j] between position descriptors.

    Cost is computed as:
    - 0 if env_id and element are exactly the same
    - elem_mismatch_cost if env_id is the same but element differs
    - env_mismatch_cost + elem_mismatch_cost if both differ

    Args:
        keys: Sequence of PositionDescriptor objects
        env_mismatch_cost: Base cost when env_ids differ
        elem_mismatch_cost: Base cost when elements differ
        normalize: If True, rescale cost matrix to [0, 1]

    Returns:
        Cost matrix of shape [K, K]
    """
    K = len(keys)
    C = np.zeros((K, K), dtype=np.float64)

    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                continue
            cost = 0.0
            if ki.env_id != kj.env_id:
                cost += env_mismatch_cost
            if ki.elem != kj.elem:
                cost += elem_mismatch_cost

            # DEFENSIVE: Ensure off-diagonal elements are strictly positive
            if cost <= 0:
                # At least one mismatch should occur for off-diagonal
                cost = min(env_mismatch_cost, elem_mismatch_cost)
                if cost <= 0:
                    cost = 1.0  # Fallback minimum cost

            C[i, j] = cost

    # DEFENSIVE: Check before normalization
    max_val = C.max()
    if max_val <= 0:
        raise ValueError(
            f"Cost matrix has non-positive max value: {max_val}. "
            f"Check mismatch costs: env={env_mismatch_cost}, elem={elem_mismatch_cost}"
        )

    if normalize:
        C = C / max_val

    return C


def debug_cost_matrix(C: np.ndarray) -> None:
    """Print diagnostic information about a cost matrix.

    This function helps diagnose issues with cost matrices by printing:
    - Matrix shape and basic statistics (min, max)
    - Off-diagonal statistics (to check if costs are non-zero)
    - Count of zero elements

    Args:
        C: Cost matrix of shape [K, K]
    """
    print(">>> Cost matrix diagnostics:")
    print(f"    shape        = {C.shape}")
    print(f"    min          = {C.min():.6f}")
    print(f"    max          = {C.max():.6f}")

    # Check off-diagonal elements
    off_diag = C[~np.eye(C.shape[0], dtype=bool)]
    if len(off_diag) > 0:
        print(f"    off-diag min = {off_diag.min():.6f}")
        print(f"    off-diag max = {off_diag.max():.6f}")
        print(f"    off-diag avg = {off_diag.mean():.6f}")

    print(f"    zeros total  = {(C == 0).sum()} / {C.size}")
    print(f"    expected diag zeros = {C.shape[0]}")


def _build_rbf_kernel(C: np.ndarray, bandwidth: float = 1.0) -> np.ndarray:
    """Build an RBF kernel matrix from the cost matrix.

    Kernel is computed as: K_ij = exp(-(C_ij / bw)^2)

    Args:
        C: Cost matrix of shape [K, K]
        bandwidth: Bandwidth parameter for RBF kernel

    Returns:
        Kernel matrix of shape [K, K]
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    K = np.exp(-(C / bandwidth) ** 2)
    return K


def load_macro_reference(
    path: Path,
    env_mismatch_cost: float = 2.0,
    elem_mismatch_cost: float = 1.0,
    normalize_cost: bool = True,
    kernel_bandwidth: float = 1.0,
) -> MacroAlignReference:
    """Load the drug halogen/heteroatom position reference distribution.

    Builds cost and kernel matrices for computing alignment metrics.

    Args:
        path: Path to drug_halopos_ref.npz file
        env_mismatch_cost: Base cost when env_ids differ
        elem_mismatch_cost: Base cost when elements differ
        normalize_cost: If True, rescale cost matrix to [0, 1]
        kernel_bandwidth: Bandwidth for RBF kernel

    Returns:
        MacroAlignReference object containing reference distribution and matrices

    Raises:
        ValueError: If env_featurizer version mismatch is detected
    """
    data = np.load(path, allow_pickle=True)
    keys_arr = data["keys"]
    freqs = data["freqs"].astype(np.float64)
    freqs = freqs / freqs.sum()

    # Version checking: ensure env_featurizer compatibility
    env_version = data.get("env_version", None)
    current_version = SimpleEnvFeaturizer.version()
    if env_version is not None:
        # Cast to string in case it's stored as numpy object
        env_version_str = str(env_version) if not isinstance(env_version, str) else env_version
        if env_version_str != current_version:
            raise ValueError(
                f"Env featurizer version mismatch:\n"
                f"  Reference file: {env_version_str}\n"
                f"  Current code:   {current_version}\n"
                f"Rebuild {path} with current env_featurizer using build_halopos_stats.py"
            )
    else:
        # Warn if no version is found (old format)
        import warnings
        warnings.warn(
            f"No env_featurizer version found in {path}. "
            f"This file may be incompatible with current code (version {current_version}). "
            f"Consider rebuilding with build_halopos_stats.py",
            UserWarning
        )

    keys: List[PositionDescriptor] = list(keys_arr.tolist())
    key_to_index = {k: i for i, k in enumerate(keys)}

    cost_matrix = _build_cost_matrix(
        keys,
        env_mismatch_cost=env_mismatch_cost,
        elem_mismatch_cost=elem_mismatch_cost,
        normalize=normalize_cost,
    )

    # DEFENSIVE: Final sanity check on cost matrix
    if cost_matrix.max() <= 0:
        raise ValueError(
            "Cost matrix has no positive entries after construction. "
            "This should not happen if _build_cost_matrix is correct."
        )

    kernel_matrix = _build_rbf_kernel(cost_matrix, bandwidth=kernel_bandwidth)

    return MacroAlignReference(
        keys=keys,
        freqs=freqs,
        key_to_index=key_to_index,
        cost_matrix=cost_matrix,
        kernel_matrix=kernel_matrix,
    )


def histogram_from_mols(
    mols: Iterable[Chem.Mol],
    env_featurizer: SimpleEnvFeaturizer,
    ref: MacroAlignReference,
    allowed_elems: Sequence[str] = ALLOWED_ELEMS,
) -> np.ndarray:
    """Build a normalized histogram over position descriptors for molecules.

    Scans all atoms in all molecules and counts occurrences of each
    (env_id, element) pair.

    Args:
        mols: Iterable of RDKit molecule objects
        env_featurizer: Object to compute env_id for each atom
        ref: MacroAlignReference with keys and mapping
        allowed_elems: Elements to consider when building histogram

    Returns:
        Histogram array of shape [K], normalized to sum to 1
        Returns all zeros if no counts are present
    """
    counts = np.zeros(len(ref.keys), dtype=np.float64)
    for mol in mols:
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym not in allowed_elems:
                continue
            idx = atom.GetIdx()
            env_id = env_featurizer.encode(mol, idx)
            desc = PositionDescriptor(env_id=env_id, elem=sym)
            hist_idx = ref.key_to_index.get(desc)
            if hist_idx is None:
                # Out-of-vocabulary env-element combo, ignore
                continue
            counts[hist_idx] += 1.0

    if counts.sum() > 0:
        counts = counts / counts.sum()
    return counts


def histogram_from_changed_atoms(
    samples: Iterable[Tuple[Chem.Mol, Sequence[int]]],
    env_featurizer: SimpleEnvFeaturizer,
    ref: MacroAlignReference,
    debug: bool = False,
) -> np.ndarray:
    """Build histogram using only changed atom positions.

    This is useful for evaluating the position distribution of modifications
    made to molecules (e.g., doping or substitutions).

    Args:
        samples: Iterable of (mol, changed_atoms) tuples where changed_atoms
            is a list of atom indices that were modified
        env_featurizer: Object to compute env_id for each atom
        ref: MacroAlignReference with keys and mapping
        debug: If True, print diagnostic information

    Returns:
        Histogram array of shape [K], normalized to sum to 1
        Returns all zeros if no counts are present
    """
    counts = np.zeros(len(ref.keys), dtype=np.float64)
    total_changed = 0
    matched = 0
    oov = 0  # Out of vocabulary

    for mol, changed_atoms in samples:
        if mol is None or not changed_atoms:
            continue
        for idx in changed_atoms:
            total_changed += 1
            if idx < 0 or idx >= mol.GetNumAtoms():
                continue
            atom = mol.GetAtomWithIdx(idx)
            sym = atom.GetSymbol()
            env_id = env_featurizer.encode(mol, idx)
            desc = PositionDescriptor(env_id=env_id, elem=sym)
            hist_idx = ref.key_to_index.get(desc)
            if hist_idx is None:
                oov += 1
                if debug and oov <= 10:  # Print first 10 OOV cases
                    print(f"  OOV: env_id={env_id}, elem={sym}")
                continue
            matched += 1
            counts[hist_idx] += 1.0

    if debug:
        print(f">>> histogram_from_changed_atoms diagnostics:")
        print(f"    total changed atoms = {total_changed}")
        print(f"    matched to ref      = {matched}")
        print(f"    out of vocabulary   = {oov}")
        print(f"    counts sum          = {counts.sum():.6f}")

    if counts.sum() > 0:
        counts = counts / counts.sum()
    return counts


def count_position_descriptors(
    mols: Iterable[Chem.Mol],
    env_featurizer: SimpleEnvFeaturizer,
    allowed_elems: Sequence[str] = ALLOWED_ELEMS,
) -> Dict[PositionDescriptor, int]:
    """Count position descriptors across molecules without reference vocabulary.

    This function scans all atoms in all molecules and counts occurrences of each
    (env_id, element) pair, regardless of whether they appear in a reference distribution.
    This is useful for building union support spaces when comparing distributions that
    may have non-overlapping vocabularies.

    Args:
        mols: Iterable of RDKit molecule objects
        env_featurizer: Object to compute env_id for each atom
        allowed_elems: Elements to consider when counting

    Returns:
        Dictionary mapping PositionDescriptor to count (raw counts, not normalized)
    """
    from collections import Counter
    counter: Counter = Counter()

    for mol in mols:
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym not in allowed_elems:
                continue
            idx = atom.GetIdx()
            env_id = env_featurizer.encode(mol, idx)
            desc = PositionDescriptor(env_id=env_id, elem=sym)
            counter[desc] += 1

    return dict(counter)


def build_union_support(
    ref: MacroAlignReference,
    baseline_counts: Dict[PositionDescriptor, int],
    env_mismatch_cost: float = 2.0,
    elem_mismatch_cost: float = 1.0,
    normalize_cost: bool = True,
    kernel_bandwidth: float = 1.0,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build union support space for comparing two distributions with different vocabularies.

    This function creates a unified vocabulary that is the union of the reference keys
    and the baseline keys. It then constructs histograms, cost matrix, and kernel matrix
    over this union support, enabling meaningful comparison even when vocabularies don't overlap.

    The approach is:
    1. Union keys = ref.keys + (baseline keys not in ref)
    2. Reference histogram: original freqs for ref keys, 0 for baseline-only keys
    3. Baseline histogram: normalized counts over all union keys
    4. Cost/kernel matrices: built over the full union support

    Args:
        ref: MacroAlignReference object with reference distribution
        baseline_counts: Dictionary of baseline position descriptor counts
        env_mismatch_cost: Base cost when env_ids differ
        elem_mismatch_cost: Base cost when elements differ
        normalize_cost: If True, rescale cost matrix to [0, 1]
        kernel_bandwidth: Bandwidth for RBF kernel
        debug: If True, print diagnostic information

    Returns:
        Tuple of (p_all, q_all, C_all, K_all):
            p_all: baseline histogram on union support [K_union]
            q_all: reference histogram on union support [K_union]
            C_all: cost matrix on union support [K_union, K_union]
            K_all: kernel matrix on union support [K_union, K_union]
    """
    # Step 1: Build union keys
    keys_ref = list(ref.keys)
    key_set_ref = set(keys_ref)

    # Find baseline keys not in reference
    extra_keys = [k for k in baseline_counts.keys() if k not in key_set_ref]
    keys_all = keys_ref + extra_keys

    if debug:
        print(">>> Union support diagnostics:")
        print(f"    ref keys         = {len(keys_ref)}")
        print(f"    baseline-only    = {len(extra_keys)}")
        print(f"    union keys       = {len(keys_all)}")
        shared = sum(1 for k in baseline_counts.keys() if k in key_set_ref)
        print(f"    shared keys      = {shared}")

    K = len(keys_all)
    key_to_index_all = {k: i for i, k in enumerate(keys_all)}

    # Step 2: Build reference histogram on union support
    # Original freqs in first len(keys_ref) positions, zeros for extra keys
    q_all = np.zeros(K, dtype=np.float64)
    q_all[:len(keys_ref)] = ref.freqs

    # Step 3: Build baseline histogram on union support
    p_all = np.zeros(K, dtype=np.float64)
    total_baseline = sum(baseline_counts.values())
    if total_baseline == 0:
        raise ValueError("Baseline has no allowed atoms; cannot build histogram.")

    for k, count in baseline_counts.items():
        idx = key_to_index_all[k]
        p_all[idx] = count / total_baseline

    if debug:
        print(f"    p_all sum        = {p_all.sum():.6f}")
        print(f"    q_all sum        = {q_all.sum():.6f}")
        p_mass_in_ref = sum(p_all[i] for i in range(len(keys_ref)))
        q_mass_in_baseline = sum(q_all[key_to_index_all[k]] for k in baseline_counts.keys() if k in key_set_ref)
        print(f"    p mass in ref    = {p_mass_in_ref:.6f}")
        print(f"    q mass in base   = {q_mass_in_baseline:.6f}")

    # Step 4: Build cost and kernel matrices on union support
    C_all = _build_cost_matrix(
        keys_all,
        env_mismatch_cost=env_mismatch_cost,
        elem_mismatch_cost=elem_mismatch_cost,
        normalize=normalize_cost,
    )

    K_all = _build_rbf_kernel(C_all, bandwidth=kernel_bandwidth)

    return p_all, q_all, C_all, K_all


def compute_macro_metrics_union(
    baseline_counts: Dict[PositionDescriptor, int],
    ref: MacroAlignReference,
    epsilon: float = 0.1,
    n_sinkhorn_iters: int = 200,
    env_mismatch_cost: float = 2.0,
    elem_mismatch_cost: float = 1.0,
    kernel_bandwidth: float = 1.0,
    debug: bool = False,
) -> MacroAlignMetrics:
    """Compute macro alignment metrics on union support space.

    This function builds a union support space from the reference and baseline
    vocabularies, then computes EMD, MMD², and L2 metrics. This approach ensures
    that baseline atoms with env_ids not in the reference are still counted,
    avoiding the "all-zero histogram" problem.

    Args:
        baseline_counts: Dictionary of baseline position descriptor counts
        ref: MacroAlignReference object with reference distribution
        epsilon: Sinkhorn regularization parameter
        n_sinkhorn_iters: Maximum Sinkhorn iterations
        env_mismatch_cost: Base cost when env_ids differ
        elem_mismatch_cost: Base cost when elements differ
        kernel_bandwidth: Bandwidth for RBF kernel in MMD
        debug: If True, print diagnostic information and enable Sinkhorn debug

    Returns:
        MacroAlignMetrics with EMD, MMD², and L2 computed on union support
    """
    p_all, q_all, C_all, K_all = build_union_support(
        ref=ref,
        baseline_counts=baseline_counts,
        env_mismatch_cost=env_mismatch_cost,
        elem_mismatch_cost=elem_mismatch_cost,
        normalize_cost=True,
        kernel_bandwidth=kernel_bandwidth,
        debug=debug,
    )

    emd = sinkhorn_emd(p_all, q_all, C_all, epsilon=epsilon, n_iters=n_sinkhorn_iters, debug=debug)
    mmd2_val = mmd2(p_all, q_all, K_all)
    l2 = float(np.linalg.norm(p_all - q_all))

    return MacroAlignMetrics(emd=emd, mmd2=mmd2_val, l2=l2)


def sinkhorn_emd(
    p: np.ndarray,
    q: np.ndarray,
    C: np.ndarray,
    epsilon: float = 0.1,
    n_iters: int = 200,
    tol: float = 1e-9,
    debug: bool = False,
) -> float:
    """Compute entropic-regularized OT distance (Sinkhorn-EMD).

    Computes the optimal transport distance between two discrete distributions
    with entropic regularization for numerical stability.

    Args:
        p: Source distribution of shape [K], sum to 1
        q: Target distribution of shape [K], sum to 1
        C: Cost matrix of shape [K, K]
        epsilon: Entropic regularization strength (larger = smoother/faster)
        n_iters: Maximum number of Sinkhorn iterations
        tol: Tolerance on marginal violations for early stopping
        debug: If True, print diagnostic information during iterations

    Returns:
        Approximated optimal transport cost
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("p and q must be 1D arrays of the same shape")
    if C.shape[0] != C.shape[1] or C.shape[0] != p.shape[0]:
        raise ValueError("C must be square with size matching p and q")

    # Build kernel matrix
    K = np.exp(-C / max(epsilon, 1e-6))

    u = np.ones_like(p)
    v = np.ones_like(q)

    for it in range(n_iters):
        Kv = K.dot(v)
        u = p / (Kv + 1e-12)
        KT_u = K.T.dot(u)
        v = q / (KT_u + 1e-12)

        # Check convergence
        if tol > 0:
            T = u[:, None] * K * v[None, :]
            row_sums = T.sum(axis=1)
            col_sums = T.sum(axis=0)
            err = max(
                np.max(np.abs(row_sums - p)),
                np.max(np.abs(col_sums - q)),
            )

            # DEBUG: Print diagnostics at first and last iterations
            if debug and (it == 0 or it == n_iters - 1 or err < tol):
                diag_sum = np.trace(T)
                off_diag_sum = T.sum() - diag_sum
                print(f">>> [Sinkhorn] iter {it}")
                print(f"    T sum       = {T.sum():.6f}")
                print(f"    T diag sum  = {diag_sum:.6f}")
                print(f"    T off-diag  = {off_diag_sum:.6f}")
                print(f"    err         = {err:.6e}")
                print(f"    EMD (cur)   = {float(np.sum(T * C)):.12e}")

            if err < tol:
                break

    T = u[:, None] * K * v[None, :]
    emd = float(np.sum(T * C))
    return emd


def mmd2(
    p: np.ndarray,
    q: np.ndarray,
    K: np.ndarray,
) -> float:
    """Compute squared Maximum Mean Discrepancy between distributions.

    MMD^2(p, q) = p^T K p + q^T K q - 2 p^T K q

    Args:
        p: First distribution of shape [K], sum to 1
        q: Second distribution of shape [K], sum to 1
        K: Kernel matrix of shape [K, K]

    Returns:
        Squared MMD value (non-negative)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("p and q must be 1D arrays of the same shape")
    if K.shape[0] != K.shape[1] or K.shape[0] != p.shape[0]:
        raise ValueError("K must be square with size matching p and q")

    term_pp = float(p @ K @ p)
    term_qq = float(q @ K @ q)
    term_pq = float(p @ K @ q)
    mmd2_val = term_pp + term_qq - 2.0 * term_pq
    if mmd2_val < 0:
        # Numerical safety
        mmd2_val = 0.0
    return mmd2_val


def compute_macro_metrics(
    gen_hist: np.ndarray,
    ref: MacroAlignReference,
    epsilon: float = 0.1,
    debug: bool = False,
) -> MacroAlignMetrics:
    """Compute macro alignment metrics between generated and reference distributions.

    Args:
        gen_hist: Generated histogram of shape [K]
        ref: MacroAlignReference with reference distribution
        epsilon: Entropic regularization for Sinkhorn
        debug: If True, enable debugging output for Sinkhorn iterations

    Returns:
        MacroAlignMetrics with EMD, MMD2, and L2 distance
    """
    emd = sinkhorn_emd(
        gen_hist, ref.freqs, ref.cost_matrix, epsilon=epsilon, debug=debug
    )
    mmd2_val = mmd2(gen_hist, ref.freqs, ref.kernel_matrix)
    l2 = float(np.linalg.norm(gen_hist - ref.freqs))
    return MacroAlignMetrics(emd=emd, mmd2=mmd2_val, l2=l2)
