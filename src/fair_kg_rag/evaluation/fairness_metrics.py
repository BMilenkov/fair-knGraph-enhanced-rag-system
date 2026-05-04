"""Fairness metrics adapted from the RAG Fairness paper (COLING 2025).

Measures demographic disparity across gender and geographic groups
in both retrieval and generation stages.
"""

from __future__ import annotations

from collections import Counter, defaultdict


def group_disparity(
    predictions: list[dict],
    ground_truths: list[dict],
    demographic_key: str = "gender",
    metric_fn: str = "em",
) -> dict[str, float]:
    """Compute group disparity (GD) across demographic groups.

    GD = Perf(protected) - Perf(non_protected).
    Values near 0 indicate fairness; positive favours protected group.

    Args:
        predictions: List of {"id", "answer", demographic_key: group_label}.
        ground_truths: List of {"id", "answer"}.
        demographic_key: Attribute to group by ("gender" or "geo_group").
        metric_fn: Metric function name ("em" or "f1").

    Returns:
        Dict with per-group accuracy and disparity.
    """
    from fair_kg_rag.evaluation.accuracy_metrics import answer_f1, exact_match

    gt_map = {g["id"]: g for g in ground_truths}
    score_fn = exact_match if metric_fn == "em" else answer_f1

    group_scores: dict[str, list[float]] = defaultdict(list)

    for pred in predictions:
        qid = pred["id"]
        gt = gt_map.get(qid)
        if gt is None:
            continue
        group = pred.get(demographic_key, "unknown")
        group_scores[group].append(score_fn(pred["answer"], gt["answer"]))

    result: dict[str, float] = {}
    for group, scores in group_scores.items():
        result[f"{group}_accuracy"] = sum(scores) / len(scores) if scores else 0.0
        result[f"{group}_count"] = len(scores)

    groups = sorted(group_scores.keys())
    if len(groups) >= 2:
        accs = [result[f"{g}_accuracy"] for g in groups]
        result["max_disparity"] = max(accs) - min(accs)

    return result


def demographic_parity(
    predictions: list[dict],
    ground_truths: list[dict],
    demographic_key: str = "gender",
) -> float:
    """Compute demographic parity: |P(correct|A) - P(correct|B)|.

    Args:
        predictions: Predictions with demographic labels.
        ground_truths: Gold answers.
        demographic_key: Attribute to group by.

    Returns:
        Absolute difference in accuracy between groups.
    """
    gd = group_disparity(predictions, ground_truths, demographic_key, "em")
    return gd.get("max_disparity", 0.0)


def equalized_odds(
    predictions: list[dict],
    ground_truths: list[dict],
    demographic_key: str = "gender",
) -> dict[str, float]:
    """Compute equalized odds across demographic groups.

    Measures whether accuracy disparity exists conditional on true answer.

    Args:
        predictions: Predictions with demographic labels.
        ground_truths: Gold answers.
        demographic_key: Attribute to group by.

    Returns:
        Dict with TPR and FPR gaps per group.
    """
    from fair_kg_rag.evaluation.accuracy_metrics import exact_match

    gt_map = {g["id"]: g for g in ground_truths}

    group_tp: dict[str, int] = Counter()
    group_fp: dict[str, int] = Counter()
    group_total_pos: dict[str, int] = Counter()
    group_total_neg: dict[str, int] = Counter()

    for pred in predictions:
        gt = gt_map.get(pred["id"])
        if gt is None:
            continue
        group = pred.get(demographic_key, "unknown")
        correct = exact_match(pred["answer"], gt["answer"]) > 0.5

        if correct:
            group_tp[group] += 1
            group_total_pos[group] += 1
        else:
            group_fp[group] += 1
            group_total_neg[group] += 1

    result: dict[str, float] = {}
    groups = sorted(set(group_tp.keys()) | set(group_fp.keys()))

    tpr_values = []
    for group in groups:
        total_pos = group_total_pos.get(group, 0) + group_total_neg.get(group, 0)
        tpr = group_tp.get(group, 0) / total_pos if total_pos > 0 else 0.0
        result[f"{group}_tpr"] = tpr
        tpr_values.append(tpr)

    if len(tpr_values) >= 2:
        result["tpr_gap"] = max(tpr_values) - min(tpr_values)

    return result


def retrieval_fairness(
    predictions: list[dict],
    ground_truths: list[dict],
    demographic_key: str = "gender",
    k: int = 5,
) -> dict[str, float]:
    """Compute retrieval fairness: MRR gap across demographic groups.

    Args:
        predictions: List with "retrieved_chunk_ids" and demographic labels.
        ground_truths: List with "supporting_chunk_ids".
        demographic_key: Attribute to group by.
        k: Cutoff for MRR computation.

    Returns:
        Dict with per-group MRR and MRR gap.
    """
    gt_map = {g["id"]: g for g in ground_truths}

    group_mrr: dict[str, list[float]] = defaultdict(list)

    for pred in predictions:
        gt = gt_map.get(pred["id"])
        if gt is None:
            continue

        group = pred.get(demographic_key, "unknown")
        retrieved = pred.get("retrieved_chunk_ids", [])[:k]
        gold_set = set(gt.get("supporting_chunk_ids", []))

        rr = 0.0
        for rank, cid in enumerate(retrieved, 1):
            if cid in gold_set:
                rr = 1.0 / rank
                break
        group_mrr[group].append(rr)

    result: dict[str, float] = {}
    mrr_values = []
    for group, scores in group_mrr.items():
        mean_mrr = sum(scores) / len(scores) if scores else 0.0
        result[f"{group}_mrr@{k}"] = mean_mrr
        mrr_values.append(mean_mrr)

    if len(mrr_values) >= 2:
        result[f"mrr@{k}_gap"] = max(mrr_values) - min(mrr_values)

    return result


def compute_fairness_metrics(
    predictions: list[dict],
    ground_truths: list[dict],
    demographic_attributes: list[str] | None = None,
) -> dict[str, dict]:
    """Compute all fairness metrics across all demographic attributes.

    Args:
        predictions: Predictions with demographic labels.
        ground_truths: Gold answers.
        demographic_attributes: Attributes to evaluate (default: gender, geo_group).

    Returns:
        Nested dict: {attribute: {metric_name: value}}.
    """
    if demographic_attributes is None:
        demographic_attributes = ["gender", "geo_group"]

    results: dict[str, dict] = {}
    for attr in demographic_attributes:
        attr_results: dict[str, float] = {}
        attr_results.update(group_disparity(predictions, ground_truths, attr))
        attr_results["demographic_parity"] = demographic_parity(
            predictions, ground_truths, attr
        )
        attr_results.update(equalized_odds(predictions, ground_truths, attr))
        attr_results.update(retrieval_fairness(predictions, ground_truths, attr))
        results[attr] = attr_results

    return results
