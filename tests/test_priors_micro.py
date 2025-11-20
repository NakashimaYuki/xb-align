# tests/test_priors_micro.py
"""Tests for M1 prior micro scorer using real DrugBank artifacts."""
import os
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
import torch

from xb_align.priors.graph_mlm import GraphMLM
from xb_align.priors.envfrag_energy import EnvFragEnergy
from xb_align.priors.pas_energy import PASEnergy
from xb_align.rewards.prior_micro import PriorMicroScorer
from xb_align.priors.atom_vocab import NUM_ATOM_CLASSES


@pytest.fixture
def m1_artifacts_paths():
    """Return paths to M1 artifacts."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        'drugs_parquet': os.path.join(project_root, "data", "processed", "drugs_std.parquet"),
        'envfrag_npz': os.path.join(project_root, "data", "processed", "envfrag_table.npz"),
        'gmlm_ckpt': os.path.join(project_root, "data", "processed", "graph_mlm.pt"),
    }


@pytest.fixture
def check_m1_artifacts(m1_artifacts_paths):
    """Skip test if M1 artifacts are not available."""
    for name, path in m1_artifacts_paths.items():
        if not os.path.exists(path):
            pytest.skip(f"M1 artifact not available: {name} at {path}")


def test_prior_micro_runs_on_single_drug(m1_artifacts_paths, check_m1_artifacts):
    """Test that PriorMicroScorer can score a single drug molecule."""
    # Load drug data
    df = pd.read_parquet(m1_artifacts_paths['drugs_parquet'])
    smi = df["smiles"].iloc[0]
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, "Failed to parse first drug SMILES"

    changed_atoms = [a.GetIdx() for a in mol.GetAtoms()]

    # Load envfrag table
    data = np.load(m1_artifacts_paths['envfrag_npz'], allow_pickle=True)
    keys = data["keys"]
    log_probs = data["log_probs"]
    table = {}
    for k, lp in zip(keys, log_probs):
        env_id, elem = k
        table[(int(env_id), str(elem))] = float(lp)
    envfrag = EnvFragEnergy(table=table, default_logp=-10.0)

    # Load PAS (placeholder)
    pas = PASEnergy()

    # Load Graph-MLM model
    device = torch.device("cpu")
    model = GraphMLM(num_atom_types=NUM_ATOM_CLASSES, hidden_dim=128)
    model.load_state_dict(torch.load(m1_artifacts_paths['gmlm_ckpt'], map_location=device))

    # Create scorer
    scorer = PriorMicroScorer(
        graph_mlm=model,
        envfrag_energy=envfrag,
        pas_energy=pas,
        device=device,
    )

    # Score the molecule
    score = scorer.log_prior_micro(mol, changed_atoms)
    assert isinstance(score, float), "Score should be a float"
    assert not np.isnan(score), "Score should not be NaN"
    assert not np.isinf(score), "Score should not be infinite"


def test_prior_micro_on_multiple_drugs(m1_artifacts_paths, check_m1_artifacts):
    """Test that PriorMicroScorer can score multiple drug molecules."""
    # Load drug data
    df = pd.read_parquet(m1_artifacts_paths['drugs_parquet'])
    smiles_list = df["smiles"].head(10).tolist()

    # Load envfrag table
    data = np.load(m1_artifacts_paths['envfrag_npz'], allow_pickle=True)
    keys = data["keys"]
    log_probs = data["log_probs"]
    table = {}
    for k, lp in zip(keys, log_probs):
        env_id, elem = k
        table[(int(env_id), str(elem))] = float(lp)
    envfrag = EnvFragEnergy(table=table, default_logp=-10.0)

    # Load PAS (placeholder)
    pas = PASEnergy()

    # Load Graph-MLM model
    device = torch.device("cpu")
    model = GraphMLM(num_atom_types=NUM_ATOM_CLASSES, hidden_dim=128)
    model.load_state_dict(torch.load(m1_artifacts_paths['gmlm_ckpt'], map_location=device))

    # Create scorer
    scorer = PriorMicroScorer(
        graph_mlm=model,
        envfrag_energy=envfrag,
        pas_energy=pas,
        device=device,
    )

    # Score all molecules
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        changed_atoms = [a.GetIdx() for a in mol.GetAtoms()]
        score = scorer.log_prior_micro(mol, changed_atoms)
        scores.append(score)

    assert len(scores) > 0, "Should have scored at least one molecule"
    assert all(isinstance(s, float) for s in scores), "All scores should be floats"
    assert all(not np.isnan(s) for s in scores), "No scores should be NaN"
    assert all(not np.isinf(s) for s in scores), "No scores should be infinite"


def test_envfrag_table_structure(m1_artifacts_paths, check_m1_artifacts):
    """Test that envfrag_table.npz has the expected structure."""
    data = np.load(m1_artifacts_paths['envfrag_npz'], allow_pickle=True)

    assert 'keys' in data, "envfrag_table should have 'keys' array"
    assert 'log_probs' in data, "envfrag_table should have 'log_probs' array"

    keys = data['keys']
    log_probs = data['log_probs']

    assert keys.shape[0] == log_probs.shape[0], "keys and log_probs should have same length"
    assert keys.shape[1] == 2, "Each key should be (env_id, elem) pair"
    assert all(lp <= 0 for lp in log_probs), "Log probabilities should be non-positive"


def test_graph_mlm_checkpoint_loads(m1_artifacts_paths, check_m1_artifacts):
    """Test that Graph-MLM checkpoint can be loaded."""
    device = torch.device("cpu")
    model = GraphMLM(num_atom_types=NUM_ATOM_CLASSES, hidden_dim=128)

    # Load checkpoint
    state_dict = torch.load(m1_artifacts_paths['gmlm_ckpt'], map_location=device)
    model.load_state_dict(state_dict)

    # Verify model is in eval mode
    model.eval()

    # Verify model has expected structure
    assert hasattr(model, 'atom_embedding'), "Model should have atom_embedding"
    assert hasattr(model, 'conv1'), "Model should have conv1"
    assert hasattr(model, 'conv2'), "Model should have conv2"
    assert hasattr(model, 'out_head'), "Model should have out_head"


def test_drugs_std_parquet_structure(m1_artifacts_paths, check_m1_artifacts):
    """Test that drugs_std.parquet has the expected structure."""
    df = pd.read_parquet(m1_artifacts_paths['drugs_parquet'])

    required_columns = ['drug_id', 'name', 'smiles', 'scaffold_smiles',
                       'has_halogen', 'has_hetero', 'mw', 'qed']

    for col in required_columns:
        assert col in df.columns, f"drugs_std.parquet should have '{col}' column"

    assert len(df) > 1000, "Should have at least 1000 drugs"
    assert df['has_halogen'].sum() > 0, "Should have some drugs with halogens"
    assert df['has_hetero'].sum() > 0, "Should have some drugs with heteroatoms"
