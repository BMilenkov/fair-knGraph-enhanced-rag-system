"""Retrieval quality metrics: MRR, Recall@k, Precision@k."""

from __future__ import annotations

from collections import defaultdict


def mrr_at_k(
    predictions: list[dict],
    ground_truths: list[dict],
    k: int = 5,
) -> float:
    """Compute Mean Reciprocal Rank at k.

    Args:
        predictions: List of {"id", "retrieved_chunk_ids": [...]}.
        ground_truths: List of {"id", "supporting_chunk_ids": [...]}.
        k: Cutoff rank.

    Returns:
        MRR@k score.
    """
    gt_map = {g["id"]: set(g.get("supporting_chunk_ids", [])) for g in ground_truths}

    rr_scores: list[float] = []
    for pred in predictions:
        gold = gt_map.get(pred["id"], set())
        if not gold:
            continue
        retrieved = pred.get("retrieved_chunk_ids", [])[:k]
        rr = 0.0
        for rank, cid in enumerate(retrieved, 1):
            if cid in gold:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)

    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0


def recall_at_k(
    predictions: list[dict],
    ground_truths: list[dict],
    k: int = 5,
) -> float:
    """Compute Recall at k.

    Args:
        predictions: List of {"id", "retrieved_chunk_ids": [...]}.
        ground_truths: List of {"id", "supporting_chunk_ids": [...]}.
        k: Cutoff rank.

    Returns:
        Recall@k score.
    """
    gt_map = {g["id"]: set(g.get("supporting_chunk_ids", [])) for g in ground_truths}

    recalls: list[float] = []
    for pred in predictions:
        gold = gt_map.get(pred["id"], set())
        if not gold:
            continue
        retrieved = set(pred.get("retrieved_chunk_ids", [])[:k])
        recalls.append(len(retrieved & gold) / len(gold))

    return sum(recalls) / len(recalls) if recalls else 0.0


def precision_at_k(
    predictions: list[dict],
    ground_truths: list[dict],
    k: int = 5,
) -> float:
    """Compute Precision at k.

    Args:
        predictions: List of {"id", "retrieved_chunk_ids": [...]}.
        ground_truths: List of {"id", "supporting_chunk_ids": [...]}.
        k: Cutoff rank.

    Returns:
        Precision@k score.
    """
    gt_map = {g["id"]: set(g.get("supporting_chunk_ids", [])) for g in ground_truths}

    precisions: list[float] = []
    for pred in predictions:
        gold = gt_map.get(pred["id"], set())
        if not gold:
            continue
        retrieved = pred.get("retrieved_chunk_ids", [])[:k]
        if not retrieved:
            precisions.append(0.0)
            continue
        precisions.append(len(set(retrieved) & gold) / len(retrieved))

    return sum(precisions) / len(precisions) if precisions else 0.0


def compute_retrieval_metrics(
    predictions: list[dict],
    ground_truths: list[dict],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute all retrieval metrics at various k values.

    Args:
        predictions: Predictions with retrieved_chunk_ids.
        ground_truths: Ground truths with supporting_chunk_ids.
        k_values: List of k cutoffs (default: [1, 3, 5, 10]).

    Returns:
        Dict mapping metric names to values.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results: dict[str, float] = {}
    for k in k_values:
        results[f"mrr@{k}"] = mrr_at_k(predictions, ground_truths, k)
        results[f"recall@{k}"] = recall_at_k(predictions, ground_truths, k)
        results[f"precision@{k}"] = precision_at_k(predictions, ground_truths, k)

    return results
