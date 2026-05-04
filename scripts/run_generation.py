"""Generate answers from retrieved contexts using a local LLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fair_kg_rag.generation.generator import Generator
from fair_kg_rag.generation.llm_backend import LLMBackend
from fair_kg_rag.utils.config import load_config
from fair_kg_rag.utils.io_utils import read_json, write_json
from fair_kg_rag.utils.logging_utils import setup_logging

logger = setup_logging("run_generation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answers from contexts")
    parser.add_argument("--config", type=Path, default=Path("configs/generation.yaml"))
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()

    cfg = load_config(args.config)

    predictions_dir = Path(cfg.get("paths", {}).get("predictions", "outputs/predictions"))
    retrieval_path = predictions_dir / f"{args.split}_retrieval.json"

    if not retrieval_path.exists():
        logger.error(f"Retrieval results not found: {retrieval_path}")
        logger.error("Run 'python scripts/run_retrieval.py' first.")
        return

    retrieval_results = read_json(retrieval_path)
    logger.info(f"Loaded {len(retrieval_results)} retrieval results")

    gen_cfg = cfg.get("generation", {})
    llm = LLMBackend(
        model_name=gen_cfg.get("model_name", "mistralai/Mistral-7B-Instruct-v0.3"),
        device=cfg.get("device", "cuda"),
        max_new_tokens=gen_cfg.get("max_new_tokens", 128),
        temperature=gen_cfg.get("temperature", 0.1),
        use_4bit=gen_cfg.get("use_4bit", True),
    )

    generator = Generator(
        llm=llm,
        include_evidence=gen_cfg.get("include_evidence_triples", False),
        max_new_tokens=gen_cfg.get("max_new_tokens", 128),
    )

    predictions = []
    for item in tqdm(retrieval_results, desc="Generating"):
        result = generator.generate(
            question=item["question"],
            context=item.get("context", ""),
        )
        predictions.append({
            "id": item["id"],
            "question": item["question"],
            "answer": result.answer,
            "raw_response": result.raw_response,
            "retrieved_chunk_ids": item.get("retrieved_chunk_ids", []),
        })

    output_path = predictions_dir / f"{args.split}_predictions.json"
    write_json(predictions, output_path)
    logger.info(f"Saved predictions: {output_path} ({len(predictions)} answers)")


if __name__ == "__main__":
    main()
