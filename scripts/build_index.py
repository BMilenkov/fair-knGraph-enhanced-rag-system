"""Build FAISS and BM25 indices from processed chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fair_kg_rag.data.corpus_manager import CorpusManager
from fair_kg_rag.retrieval.semantic_retriever import SemanticRetriever
from fair_kg_rag.retrieval.sparse_retriever import SparseRetriever
from fair_kg_rag.utils.config import load_config
from fair_kg_rag.utils.logging_utils import setup_logging

logger = setup_logging("build_index")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval indices")
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval.yaml"))
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_dir = Path(cfg.get("paths", {}).get("raw_data", "data/raw"))
    processed_dir = Path(cfg.get("paths", {}).get("processed_data", "data/processed"))
    index_dir = Path(cfg.get("paths", {}).get("index_data", "data/indices"))

    manager = CorpusManager(raw_dir=raw_dir, processed_dir=processed_dir)
    try:
        chunks = manager.load_processed(args.split)
    except FileNotFoundError:
        logger.info("Processing split first...")
        chunks, _ = manager.process_split(args.split)

    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]

    # Dense index
    dense_cfg = cfg.get("retrieval", {}).get("dense", {})
    model_name = dense_cfg.get("model_name", "BAAI/bge-base-en-v1.5")
    index_path = index_dir / f"{args.split}_faiss.index"

    logger.info(f"Building dense index with {model_name}...")
    semantic = SemanticRetriever(
        model_name=model_name,
        index_path=index_path,
        device=cfg.get("device", "cuda"),
    )
    semantic.build_index(chunk_ids, texts)

    # Sparse index (BM25 is in-memory, just verify it works)
    sparse_cfg = cfg.get("retrieval", {}).get("sparse", {})
    logger.info("Building BM25 index...")
    sparse = SparseRetriever(
        k1=sparse_cfg.get("k1", 1.5), b=sparse_cfg.get("b", 0.75)
    )
    sparse.build_index(chunk_ids, texts)

    logger.info(f"Indices built: {len(chunk_ids)} chunks indexed")


if __name__ == "__main__":
    main()
