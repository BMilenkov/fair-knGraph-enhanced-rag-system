"""Unit tests for statistical utility functions."""

from fair_kg_rag.evaluation.statistical_tests import (
    bootstrap_confidence_interval,
    cohens_d,
    permutation_test,
)


def test_bootstrap_confidence_interval_bounds() -> None:
    low, high = bootstrap_confidence_interval([0.1, 0.2, 0.3], num_samples=200, seed=7)
    assert low <= high


def test_permutation_test_range() -> None:
    p = permutation_test([0.1, 0.2, 0.3], [0.7, 0.8, 0.9], num_permutations=300, seed=1)
    assert 0.0 <= p <= 1.0


def test_cohens_d_nonzero_for_separated_samples() -> None:
    d = cohens_d([1.0, 1.1, 1.2], [2.0, 2.1, 2.2])
    assert d < 0
