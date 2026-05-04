"""Compute accuracy + fairness metrics on generated predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fair_kg_rag.data.dataset_loader import load_dataset
from fair_kg_rag.evaluation.evaluator import Evaluator
from fair_kg_rag.utils.config import load_config
from fair_kg_rag.utils.io_utils import read_json, write_json
from fair_kg_rag.utils.logging_utils import setup_logging

logger = setup_logging("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation.yaml"))
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_dir = Path(cfg.get("paths", {}).get("raw_data", "data/raw"))
    predictions_dir = Path(cfg.get("paths", {}).get("predictions", "outputs/predictions"))
    metrics_dir = Path(cfg.get("paths", {}).get("metrics", "outputs/metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    pred_path = predictions_dir / f"{args.split}_predictions.json"
    if not pred_path.exists():
        logger.error(f"Predictions not found: {pred_path}")
        return
    predictions = read_json(pred_path)

    # Load ground truths
    records = load_dataset(raw_dir / f"{args.split}.json")
    ground_truths = [
        {
            "id": r.id,
            "answer": r.answer,
            "supporting_titles": list(r.supporting_titles),
            "supporting_chunk_ids": [
                f"doc_{t.replace(' ', '_')[:80]}_0"
                for t in r.supporting_titles
            ],
        }
        for r in records
    ]

    # Load demographics if available
    demo_path = Path(cfg.get("paths", {}).get("processed_data", "data/processed"))
    demo_file = demo_path / "entity_demographics.json"
    if demo_file.exists():
        demographics = read_json(demo_file)
        demo_map = {d["qid"]: d for d in demographics}

        # Annotate predictions with demographics from their associated entities
        record_map = {r.id: r for r in records}
        for pred in predictions:
            record = record_map.get(pred["id"])
            if record:
                for qid in record.wikidata_ids:
                    demo = demo_map.get(qid, {})
                    if demo.get("gender"):
                        pred.setdefault("gender", demo["gender"])
                    if demo.get("geo_group"):
                        pred.setdefault("geo_group", demo["geo_group"])

    # Run evaluation
    evaluator = Evaluator(cfg=cfg)
    result = evaluator.evaluate(
        predictions=predictions,
        ground_truths=ground_truths,
        retrieval_predictions=predictions,
    )

    # Save metrics
    output_path = metrics_dir / f"{args.split}_metrics.json"
    write_json(result.metrics, output_path)
    logger.info(f"Metrics saved to {output_path}")

    # Print summary
    acc = result.metrics.get("accuracy", {})
    logger.info(
        "Results: EM=%.4f, F1=%.4f",
        acc.get("exact_match", 0),
        acc.get("answer_f1", 0),
    )

    fairness = result.metrics.get("fairness", {})
    for attr, metrics in fairness.items():
        dp = metrics.get("demographic_parity", 0)
        logger.info(f"  {attr} demographic parity: {dp:.4f}")


if __name__ == "__main__":
    main()
