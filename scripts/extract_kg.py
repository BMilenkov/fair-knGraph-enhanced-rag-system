"""Extract and normalize triplets, then persist KG to Neo4j."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fair_kg_rag.data.corpus_manager import CorpusManager
from fair_kg_rag.data.dataset_loader import load_dataset
from fair_kg_rag.kg.entity_linker import EntityLinker
from fair_kg_rag.kg.kg_store import connect_kg, save_kg
from fair_kg_rag.kg.triplet_extractor import Triplet, extract_triplets_from_evidence, triplets_to_dicts
from fair_kg_rag.utils.config import load_config
from fair_kg_rag.utils.io_utils import write_json
from fair_kg_rag.utils.logging_utils import setup_logging

logger = setup_logging("extract_kg")


def _load_or_process_chunks(
    raw_dir: Path,
    processed_dir: Path,
    split: str,
    max_tokens: int,
) -> tuple[list, dict[str, list[str]]]:
    manager = CorpusManager(raw_dir=raw_dir, processed_dir=processed_dir, max_tokens=max_tokens)

    try:
        chunks = manager.load_processed(split)
        mapping = manager._question_to_chunks or {}
        if mapping:
            return chunks, mapping
    except FileNotFoundError:
        pass

    chunks, mapping = manager.process_split(split)
    return chunks, mapping


def run_extract_kg(config_path: Path, split: str = "dev") -> dict:
    """Run KG extraction and persistence pipeline."""
    cfg = load_config(config_path)

    raw_dir = Path(cfg.get("paths", {}).get("raw_data", "data/raw"))
    processed_dir = Path(cfg.get("paths", {}).get("processed_data", "data/processed"))
    kg_dir = Path(cfg.get("paths", {}).get("kg_data", "data/kg"))

    chunk_cfg = cfg.get("chunking", {})
    max_tokens = int(chunk_cfg.get("max_tokens", 512))

    logger.info("Loading/processing chunks for split=%s", split)
    _, question_to_chunks = _load_or_process_chunks(raw_dir, processed_dir, split, max_tokens)

    dataset_path = raw_dir / f"{split}.json"
    records = load_dataset(dataset_path)

    logger.info("Extracting evidence-based triplets from %s records", len(records))
    triplets: list[Triplet] = []

    for record in records:
        chunk_ids = question_to_chunks.get(record.id, [])
        evidence_tuples = [(ev.subject, ev.relation, ev.obj) for ev in record.evidences]
        # Link evidence triples to all associated chunks
        for cid in chunk_ids:
            triplets.extend(extract_triplets_from_evidence(evidence_tuples, chunk_id=cid))
        if not chunk_ids:
            triplets.extend(extract_triplets_from_evidence(evidence_tuples, chunk_id=""))

    logger.info("Raw triplets extracted: %s", len(triplets))

    linking_cfg = cfg.get("kg_extraction", {}).get("entity_linking", {})
    linker = EntityLinker(
        overlap_threshold=float(linking_cfg.get("overlap_threshold", 0.90)),
        ngram_size=int(linking_cfg.get("ngram_size", 3)),
    )
    linker.build_from_triplets(triplets)
    normalized_triplets = linker.normalize_triplets(triplets)

    neo4j_cfg = cfg.get("neo4j", {})
    kg = connect_kg(
        uri=neo4j_cfg.get("uri"),
        user=neo4j_cfg.get("user"),
        password=neo4j_cfg.get("password"),
        database=neo4j_cfg.get("database", "neo4j"),
        clear_on_init=bool(neo4j_cfg.get("clear_on_start", False)),
    )

    batch_size = int(cfg.get("kg_extraction", {}).get("neo4j_batch_size", 2000))
    for i in range(0, len(normalized_triplets), batch_size):
        kg.add_triplets(normalized_triplets[i : i + batch_size])

    kg_name = f"{split}_kg"
    save_kg(kg, output_dir=kg_dir, name=kg_name)

    if bool(cfg.get("kg_extraction", {}).get("save_intermediate", True)):
        triplets_path = kg_dir / f"{kg_name}_triplets.json"
        write_json(triplets_to_dicts(normalized_triplets), triplets_path)
        logger.info("Saved normalized triplets to %s", triplets_path)

    summary = kg.summary()
    logger.info("KG extraction complete: %s", summary)
    kg.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract KG and persist to Neo4j")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/kg_extraction.yaml"),
        help="Path to KG extraction config",
    )
    parser.add_argument("--split", default="dev", help="Dataset split")
    args = parser.parse_args()

    run_extract_kg(config_path=args.config, split=args.split)


if __name__ == "__main__":
    main()
