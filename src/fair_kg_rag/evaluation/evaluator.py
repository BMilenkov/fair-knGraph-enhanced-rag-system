"""Unified evaluator running accuracy, retrieval, fairness, and statistical tests."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from fair_kg_rag.evaluation.accuracy_metrics import (
    answer_f1,
    compute_accuracy_metrics,
    exact_match,
)
from fair_kg_rag.evaluation.fairness_metrics import compute_fairness_metrics
from fair_kg_rag.evaluation.retrieval_metrics import compute_retrieval_metrics
from fair_kg_rag.evaluation.statistical_tests import bootstrap_ci

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Aggregated evaluation results.

    Attributes:
        metrics: Nested dict of {category: {metric_name: value}}.
        per_question: Per-question scores for detailed analysis.
    """

    metrics: dict[str, Any] = field(default_factory=dict)
    per_question: list[dict] = field(default_factory=list)


class Evaluator:
    """Unified evaluator for accuracy, retrieval quality, and fairness.

    Args:
        cfg: Evaluation configuration dict or DictConfig.
    """

    def __init__(self, cfg: dict | Any = None) -> None:
        self.cfg = cfg or {}

    def evaluate(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
        retrieval_predictions: list[dict] | None = None,
    ) -> EvaluationResult:
        """Run all configured evaluation metrics.

        Args:
            predictions: List of {"id", "answer", optional demographic keys}.
            ground_truths: List of {"id", "answer", "supporting_titles", ...}.
            retrieval_predictions: Optional list of {"id", "retrieved_chunk_ids"}.

        Returns:
            EvaluationResult with all metrics and per-question scores.
        """
        result = EvaluationResult()
        eval_cfg = self.cfg.get("evaluation", {})

        # Accuracy metrics
        result.metrics["accuracy"] = compute_accuracy_metrics(
            predictions, ground_truths
        )

        # Retrieval metrics
        if retrieval_predictions:
            k_values = eval_cfg.get("retrieval", {}).get("k_values", [1, 3, 5, 10])
            result.metrics["retrieval"] = compute_retrieval_metrics(
                retrieval_predictions, ground_truths, k_values=k_values
            )
        else:
            result.metrics["retrieval"] = {}

        # Fairness metrics
        fairness_cfg = eval_cfg.get("fairness", {})
        if fairness_cfg.get("enabled", False):
            demo_attrs = fairness_cfg.get(
                "demographic_attributes", ["gender", "geo_group"]
            )
            result.metrics["fairness"] = compute_fairness_metrics(
                predictions, ground_truths, demographic_attributes=demo_attrs
            )
        else:
            result.metrics["fairness"] = {}

        # Statistical significance (bootstrap CI on answer F1)
        stats_cfg = eval_cfg.get("statistics", {})
        gt_map = {g["id"]: g for g in ground_truths}
        f1_scores = [
            answer_f1(p["answer"], gt_map[p["id"]]["answer"])
            for p in predictions
            if p["id"] in gt_map
        ]

        if f1_scores:
            mean, ci_lo, ci_hi = bootstrap_ci(
                f1_scores,
                n_bootstrap=stats_cfg.get("bootstrap_samples", 1000),
                confidence=stats_cfg.get("confidence_level", 0.95),
                seed=stats_cfg.get("random_seed", 42),
            )
            result.metrics["statistics"] = {
                "f1_mean": mean,
                "f1_ci_lower": ci_lo,
                "f1_ci_upper": ci_hi,
            }
        else:
            result.metrics["statistics"] = {}

        # Per-question breakdown
        for pred in predictions:
            qid = pred["id"]
            gt = gt_map.get(qid)
            if gt is None:
                continue
            result.per_question.append({
                "id": qid,
                "em": exact_match(pred["answer"], gt["answer"]),
                "f1": answer_f1(pred["answer"], gt["answer"]),
                "predicted": pred["answer"],
                "gold": gt["answer"],
            })

        logger.info(
            "Evaluation complete: %d questions, EM=%.4f, F1=%.4f",
            len(result.per_question),
            result.metrics["accuracy"].get("exact_match", 0),
            result.metrics["accuracy"].get("answer_f1", 0),
        )

        return result
