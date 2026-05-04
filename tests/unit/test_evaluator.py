"""Unit tests for unified evaluator."""

from fair_kg_rag.evaluation.evaluator import Evaluator


def test_evaluator_outputs_accuracy_fairness_and_stats() -> None:
    cfg = {
        "evaluation": {
            "fairness": {
                "enabled": True,
                "demographic_attributes": ["gender", "geography"],
            },
            "retrieval": {"k_values": [1, 3]},
            "statistics": {
                "bootstrap_samples": 200,
                "confidence_level": 0.95,
                "random_seed": 42,
            },
        }
    }

    predictions = [
        {
            "id": "q1",
            "answer": "Paris",
            "gender": "female",
            "geo_group": "european",
        },
        {
            "id": "q2",
            "answer": "Berlin",
            "gender": "male",
            "geo_group": "european",
        },
    ]
    ground_truths = [
        {
            "id": "q1",
            "answer": "Paris",
            "supporting_chunk_ids": ["c1"],
            "supporting_titles": ["Paris"],
        },
        {
            "id": "q2",
            "answer": "Munich",
            "supporting_chunk_ids": ["c2"],
            "supporting_titles": ["Munich"],
        },
    ]
    retrieval_predictions = [
        {"id": "q1", "retrieved_chunk_ids": ["c1", "c3"]},
        {"id": "q2", "retrieved_chunk_ids": ["c3", "c2"]},
    ]

    result = Evaluator(cfg=cfg).evaluate(
        predictions=predictions,
        ground_truths=ground_truths,
        retrieval_predictions=retrieval_predictions,
    )

    assert "accuracy" in result.metrics
    assert "retrieval" in result.metrics
    assert "fairness" in result.metrics
    assert "statistics" in result.metrics
    assert len(result.per_question) == 2
