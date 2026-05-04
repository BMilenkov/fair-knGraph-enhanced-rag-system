"""Accuracy metrics for multi-hop QA evaluation (HotpotQA-style)."""

from __future__ import annotations

from fair_kg_rag.utils.text_utils import compute_token_f1, normalize_text


def exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match after normalization.

    Args:
        prediction: Predicted answer.
        ground_truth: Gold answer.

    Returns:
        1.0 if match, 0.0 otherwise.
    """
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def answer_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth.

    Args:
        prediction: Predicted answer.
        ground_truth: Gold answer.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    return compute_token_f1(prediction, ground_truth)


def support_f1(
    predicted_titles: set[str],
    gold_titles: set[str],
) -> float:
    """Compute F1 for supporting passage identification.

    Args:
        predicted_titles: Set of predicted supporting paragraph titles.
        gold_titles: Set of gold supporting paragraph titles.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    if not predicted_titles and not gold_titles:
        return 1.0
    if not predicted_titles or not gold_titles:
        return 0.0

    common = predicted_titles & gold_titles
    precision = len(common) / len(predicted_titles)
    recall = len(common) / len(gold_titles)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_accuracy_metrics(
    predictions: list[dict],
    ground_truths: list[dict],
) -> dict[str, float]:
    """Compute all accuracy metrics over a dataset.

    Args:
        predictions: List of {"id": ..., "answer": ...}.
        ground_truths: List of {"id": ..., "answer": ..., "supporting_titles": [...]}.

    Returns:
        Dictionary of aggregated metrics.
    """
    gt_map = {g["id"]: g for g in ground_truths}

    em_scores: list[float] = []
    f1_scores: list[float] = []
    sp_f1_scores: list[float] = []

    for pred in predictions:
        qid = pred["id"]
        if qid not in gt_map:
            continue

        gt = gt_map[qid]
        em_scores.append(exact_match(pred["answer"], gt["answer"]))
        f1_scores.append(answer_f1(pred["answer"], gt["answer"]))

        if "supporting_titles" in pred and "supporting_titles" in gt:
            sp_f1_scores.append(
                support_f1(
                    set(pred["supporting_titles"]),
                    set(gt["supporting_titles"]),
                )
            )

    n = len(em_scores) or 1
    results = {
        "exact_match": sum(em_scores) / n,
        "answer_f1": sum(f1_scores) / n,
        "num_evaluated": len(em_scores),
    }

    if sp_f1_scores:
        results["support_f1"] = sum(sp_f1_scores) / len(sp_f1_scores)

    return results
