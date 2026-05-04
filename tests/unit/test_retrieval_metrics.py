"""Unit tests for retrieval metrics."""

from fair_kg_rag.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_reciprocal_rank_returns_inverse_rank() -> None:
    assert reciprocal_rank(["c1", "c2", "c3"], {"c2"}) == 0.5


def test_recall_and_precision_at_k() -> None:
    retrieved = ["c1", "c2", "c3", "c4"]
    gold = {"c2", "c4"}

    assert recall_at_k(retrieved, gold, 2) == 0.5
    assert precision_at_k(retrieved, gold, 2) == 0.5


def test_compute_retrieval_metrics_with_groups() -> None:
    predictions = [
        {"id": "q1", "retrieved_chunk_ids": ["c1", "c2"], "group": "a"},
        {"id": "q2", "retrieved_chunk_ids": ["c3", "c4"], "group": "b"},
    ]
    ground_truths = [
        {"id": "q1", "supporting_chunk_ids": ["c2"]},
        {"id": "q2", "supporting_chunk_ids": ["c3"]},
    ]

    metrics = compute_retrieval_metrics(
        predictions=predictions,
        ground_truths=ground_truths,
        k_values=[1, 2],
        group_attr="group",
    )

    assert "mrr" in metrics
    assert metrics["num_evaluated"] == 2.0
    assert "recall@1" in metrics
    assert "precision@2" in metrics
    assert "group_mrr" in metrics
