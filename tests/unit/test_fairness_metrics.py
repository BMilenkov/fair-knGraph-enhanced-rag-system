"""Unit tests for fairness metrics."""

from fair_kg_rag.evaluation.fairness_metrics import (
    compute_fairness_metrics,
    demographic_parity_difference,
    disparate_impact_ratio,
    equalized_odds_difference,
    retrieval_group_distribution,
)


def test_demographic_parity_and_disparate_impact() -> None:
    rates = {"g1": 0.8, "g2": 0.4}
    assert demographic_parity_difference(rates) == 0.4
    assert disparate_impact_ratio(rates) == 0.5


def test_equalized_odds_difference_nonzero() -> None:
    records = [
        {"group": "a", "gold": 1, "pred": 1},
        {"group": "a", "gold": 0, "pred": 0},
        {"group": "b", "gold": 1, "pred": 0},
        {"group": "b", "gold": 0, "pred": 1},
    ]

    score = equalized_odds_difference(records, "group", "gold", "pred")
    assert score > 0


def test_retrieval_group_distribution() -> None:
    chunk_demo = {
        "c1": {"gender": "female"},
        "c2": {"gender": "male"},
        "c3": {"gender": "female"},
    }

    dist = retrieval_group_distribution(["c1", "c2", "c3"], chunk_demo, "gender")
    assert dist["female"] == 2 / 3
    assert dist["male"] == 1 / 3


def test_compute_fairness_metrics() -> None:
    predictions = [
        {
            "gender": "female",
            "is_correct": 1.0,
            "gold_positive": True,
            "pred_positive": True,
        },
        {
            "gender": "male",
            "is_correct": 0.0,
            "gold_positive": True,
            "pred_positive": False,
        },
    ]

    metrics = compute_fairness_metrics(predictions, group_attrs=["gender"])
    assert "gender" in metrics
    assert "demographic_parity_diff" in metrics["gender"]
