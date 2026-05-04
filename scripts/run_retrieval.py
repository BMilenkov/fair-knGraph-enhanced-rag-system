"""Execute the retrieval pipeline and save results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fair_kg_rag.data.corpus_manager import CorpusManager
from fair_kg_rag.data.dataset_loader import load_dataset
from fair_kg_rag.kg.kg_store import load_kg
from fair_kg_rag.retrieval.retrieval_pipeline import RetrievalPipeline
from fair_kg_rag.utils.config import load_config
from fair_kg_rag.utils.io_utils import write_json
from fair_kg_rag.utils.logging_utils import setup_logging

logger = setup_logging("run_retrieval")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval.yaml"))
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_dir = Path(cfg.get("paths", {}).get("raw_data", "data/raw"))
    processed_dir = Path(cfg.get("paths", {}).get("processed_data", "data/processed"))
    output_dir = Path(cfg.get("paths", {}).get("predictions", "outputs/predictions"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load chunks
    manager = CorpusManager(raw_dir=raw_dir, processed_dir=processed_dir)
    chunks = manager.load_processed(args.split)
    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    chunk_texts = dict(zip(chunk_ids, texts))

    # Connect to Neo4j KG
    neo4j_cfg = cfg.get("neo4j", {})
    kg = None
    if cfg.get("retrieval", {}).get("kg_expansion", {}).get("enabled", False):
        kg = load_kg(
            uri=neo4j_cfg.get("uri"),
            user=neo4j_cfg.get("user"),
            password=neo4j_cfg.get("password"),
            database=neo4j_cfg.get("database", "neo4j"),
        )

    # Setup pipeline (inject split for index path resolution)
    cfg["_split"] = args.split
    pipeline = RetrievalPipeline(cfg=cfg, kg=kg, chunk_texts=chunk_texts)
    pipeline.setup(chunk_ids, texts)

    # Load questions
    records = load_dataset(raw_dir / f"{args.split}.json")
    max_samples = cfg.get("dataset", {}).get("max_samples")
    if max_samples:
        records = records[:int(max_samples)]

    # Run retrieval
    results = []
    for record in tqdm(records, desc="Retrieving"):
        ret = pipeline.retrieve(record.question)
        results.append({
            "id": record.id,
            "question": record.question,
            "retrieved_chunk_ids": [cid for cid, _ in ret.retrieved_chunks],
            "retrieved_scores": [s for _, s in ret.retrieved_chunks],
            "context": ret.context,
            "metadata": ret.metadata,
        })

    output_path = output_dir / f"{args.split}_retrieval.json"
    write_json(results, output_path)
    logger.info(f"Saved retrieval results: {output_path} ({len(results)} queries)")

    if kg is not None:
        kg.close()


if __name__ == "__main__":
    main()
