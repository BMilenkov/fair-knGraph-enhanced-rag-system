"""Statistical significance testing: bootstrap CI and permutation tests."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        scores: List of per-sample scores.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        Tuple of (mean, ci_lower, ci_upper).
    """
    if not scores:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    arr = np.array(scores)
    means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))

    means = sorted(means)
    alpha = 1.0 - confidence
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1.0 - alpha / 2) * n_bootstrap) - 1

    return float(np.mean(arr)), means[lower_idx], means[upper_idx]


def paired_permutation_test(
    scores_a: list[float],
    scores_b: list[float],
    n_permutations: int = 1000,
    seed: int = 42,
) -> float:
    """Paired permutation test for comparing two systems.

    Tests H0: system A and system B have the same expected performance.

    Args:
        scores_a: Per-sample scores for system A.
        scores_b: Per-sample scores for system B.
        n_permutations: Number of permutation iterations.
        seed: Random seed.

    Returns:
        p-value (probability of observing the difference under H0).
    """
    if len(scores_a) != len(scores_b) or not scores_a:
        return 1.0

    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    observed_diff = abs(float(np.mean(a) - np.mean(b)))

    count_extreme = 0
    for _ in range(n_permutations):
        mask = rng.randint(0, 2, size=len(a)).astype(bool)
        perm_a = np.where(mask, a, b)
        perm_b = np.where(mask, b, a)
        perm_diff = abs(float(np.mean(perm_a) - np.mean(perm_b)))
        if perm_diff >= observed_diff:
            count_extreme += 1

    return count_extreme / n_permutations
