# xb_align/baseline/scoring.py
"""Prior-based scoring and ranking for baseline samples."""

from typing import List

from rdkit import Chem

from xb_align.baseline.generator import BaselineSample
from xb_align.rewards.prior_micro import PriorMicroScorer


class BaselinePriorRanker:
    """Simple wrapper that ranks baseline samples using PriorMicroScorer.

    This class provides a convenient interface for scoring and ranking
    baseline-generated molecules based on their position prior scores.
    """

    def __init__(self, prior_scorer: PriorMicroScorer):
        """Initialize BaselinePriorRanker.

        Args:
            prior_scorer: Trained PriorMicroScorer instance
        """
        self.prior_scorer = prior_scorer

    def score_sample(self, sample: BaselineSample) -> float:
        """Compute prior score for a single baseline sample.

        Args:
            sample: BaselineSample object to score

        Returns:
            Log-prior score (higher is better)
            Returns -inf if molecule is invalid
        """
        mol = Chem.MolFromSmiles(sample.generated_smiles)
        if mol is None:
            return float("-inf")

        # changed_atoms are defined on the scaffold atom indices
        # Since we only do element substitution, the indices remain consistent
        log_prior = self.prior_scorer.log_prior_micro(
            mol=mol,
            changed_atoms=sample.changed_atoms,
        )
        return float(log_prior)

    def rank_samples(self, samples: List[BaselineSample]) -> List[BaselineSample]:
        """Rank baseline samples by prior score in descending order.

        Args:
            samples: List of BaselineSample objects to rank

        Returns:
            List of BaselineSample objects sorted by score (highest first)
            Samples with invalid scores (-inf) are filtered out
        """
        scored = [(self.score_sample(s), s) for s in samples]
        scored = [item for item in scored if item[0] != float("-inf")]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored]
