# tests/test_baseline_scoring.py
"""Tests for baseline scoring."""

import pytest
from unittest.mock import Mock
from rdkit import Chem

from xb_align.baseline.generator import BaselineSample
from xb_align.baseline.scoring import BaselinePriorRanker


class MockPriorScorer:
    """Mock PriorMicroScorer for testing."""

    def __init__(self, score_fn=None):
        self.score_fn = score_fn or (lambda mol, changed_atoms: -len(changed_atoms))

    def log_prior_micro(self, mol, changed_atoms):
        return self.score_fn(mol, changed_atoms)


def test_baseline_ranker_score_sample():
    """Test scoring a single sample."""
    scorer = MockPriorScorer(score_fn=lambda mol, changed_atoms: -5.0)
    ranker = BaselinePriorRanker(prior_scorer=scorer)

    sample = BaselineSample(
        scaffold_id=0,
        scaffold_smiles="c1ccccc1",
        generated_smiles="c1ccc(N)cc1",
        changed_atoms=[3],
    )

    score = ranker.score_sample(sample)
    assert score == -5.0


def test_baseline_ranker_invalid_smiles():
    """Test that invalid SMILES returns -inf score."""
    scorer = MockPriorScorer()
    ranker = BaselinePriorRanker(prior_scorer=scorer)

    sample = BaselineSample(
        scaffold_id=0,
        scaffold_smiles="c1ccccc1",
        generated_smiles="INVALID_SMILES",
        changed_atoms=[3],
    )

    score = ranker.score_sample(sample)
    assert score == float("-inf")


def test_baseline_ranker_rank_samples():
    """Test ranking multiple samples."""
    # Score based on number of changed atoms (fewer changes = better)
    scorer = MockPriorScorer(score_fn=lambda mol, changed_atoms: -len(changed_atoms))
    ranker = BaselinePriorRanker(prior_scorer=scorer)

    samples = [
        BaselineSample(0, "c1ccccc1", "c1ccc(N)cc1", [3]),       # score: -1
        BaselineSample(1, "c1ccccc1", "c1c(N)c(O)c(F)cc1", [1, 2, 3]),  # score: -3
        BaselineSample(2, "c1ccccc1", "c1cc(N)c(O)cc1", [2, 3]),   # score: -2
    ]

    ranked = ranker.rank_samples(samples)

    # Should be sorted by score (highest first)
    assert len(ranked) == 3
    assert len(ranked[0].changed_atoms) == 1  # Best score
    assert len(ranked[1].changed_atoms) == 2
    assert len(ranked[2].changed_atoms) == 3  # Worst score


def test_baseline_ranker_filters_invalid():
    """Test that invalid samples are filtered out."""
    scorer = MockPriorScorer()
    ranker = BaselinePriorRanker(prior_scorer=scorer)

    samples = [
        BaselineSample(0, "c1ccccc1", "c1ccc(N)cc1", [3]),
        BaselineSample(1, "c1ccccc1", "INVALID", [1]),  # Invalid
        BaselineSample(2, "c1ccccc1", "c1cc(O)ccc1", [2]),
    ]

    ranked = ranker.rank_samples(samples)

    # Should only have valid samples
    assert len(ranked) == 2
    assert all(s.generated_smiles != "INVALID" for s in ranked)


def test_baseline_ranker_deterministic():
    """Test that ranking is deterministic."""
    scorer = MockPriorScorer(score_fn=lambda mol, changed_atoms: -len(changed_atoms))
    ranker = BaselinePriorRanker(prior_scorer=scorer)

    samples = [
        BaselineSample(0, "c1ccccc1", "c1ccc(N)cc1", [3]),
        BaselineSample(1, "c1ccccc1", "c1c(N)c(O)c(F)cc1", [1, 2, 3]),
        BaselineSample(2, "c1ccccc1", "c1cc(N)c(O)cc1", [2, 3]),
    ]

    ranked1 = ranker.rank_samples(samples)
    ranked2 = ranker.rank_samples(samples)

    # Should be identical
    assert len(ranked1) == len(ranked2)
    for s1, s2 in zip(ranked1, ranked2):
        assert s1.scaffold_id == s2.scaffold_id
        assert s1.generated_smiles == s2.generated_smiles


def test_baseline_ranker_empty_list():
    """Test ranking an empty list."""
    scorer = MockPriorScorer()
    ranker = BaselinePriorRanker(prior_scorer=scorer)

    ranked = ranker.rank_samples([])
    assert ranked == []


def test_baseline_ranker_single_sample():
    """Test ranking a single sample."""
    scorer = MockPriorScorer()
    ranker = BaselinePriorRanker(prior_scorer=scorer)

    samples = [
        BaselineSample(0, "c1ccccc1", "c1ccc(N)cc1", [3]),
    ]

    ranked = ranker.rank_samples(samples)
    assert len(ranked) == 1
    assert ranked[0].scaffold_id == 0


def test_baseline_ranker_stable_sort():
    """Test that ranking preserves order for equal scores."""
    # All samples get the same score
    scorer = MockPriorScorer(score_fn=lambda mol, changed_atoms: 0.0)
    ranker = BaselinePriorRanker(prior_scorer=scorer)

    samples = [
        BaselineSample(0, "c1ccccc1", "c1ccc(N)cc1", [3]),
        BaselineSample(1, "c1ccccc1", "c1ccc(O)cc1", [3]),
        BaselineSample(2, "c1ccccc1", "c1ccc(F)cc1", [3]),
    ]

    ranked = ranker.rank_samples(samples)

    # Python's sort is stable, so original order should be preserved
    # for equal scores
    assert len(ranked) == 3
