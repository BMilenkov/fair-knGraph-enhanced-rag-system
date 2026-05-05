"""Fetch demographic attributes for all entities in the dataset from Wikidata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fair_kg_rag.data.dataset_loader import load_dataset
from fair_kg_rag.data.wikidata_demographics import (
    demographics_to_dicts,
    fetch_demographics_batch,
)
from fair_kg_rag.utils.config import load_config
from fair_kg_rag.utils.io_utils import write_json
from fair_kg_rag.utils.logging_utils import setup_logging

logger = setup_logging("fetch_demographics")


def main() -> None:
    """Extract all Wikidata QIDs from the dataset and fetch demographics."""
    parser = argparse.ArgumentParser(description="Fetch entity demographics from Wikidata")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/wikidata_demographics.yaml"),
    )
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_dir = Path(cfg.get("paths", {}).get("raw_data", "data/raw"))
    data_path = raw_dir / f"{args.split}.json"

    logger.info(f"Loading dataset from {data_path}")
    records = load_dataset(data_path)

    all_qids: set[str] = set()
    for record in records:
        all_qids.update(record.wikidata_ids)

    logger.info(f"Found {len(all_qids)} unique Wikidata entities")

    logger.info("Fetching demographics from Wikidata SPARQL...")
    demographics = fetch_demographics_batch(list(all_qids))

    gendered = sum(1 for d in demographics.values() if d.gender is not None)
    with_country = sum(1 for d in demographics.values() if d.country is not None)

    logger.info(
        f"Fetched demographics for {len(demographics)} entities: "
        f"{gendered} with gender, {with_country} with country"
    )

    output_path = Path(
        cfg.get("demographics", {}).get(
            "output_file", "data/processed/entity_demographics.json"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(demographics_to_dicts(demographics), output_path)
    logger.info(f"Saved demographics to {output_path}")


if __name__ == "__main__":
    main()
