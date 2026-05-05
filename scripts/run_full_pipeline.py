"""Fair KG-Enhanced RAG — Full Pipeline Runner.

Runs the end-to-end pipeline with smart caching, versioned artifacts,
structured logging, and stage-level skip/resume support.

Pipeline stages:
  1. PREPROCESS  — Parse 2WikiMultiHopQA, chunk paragraphs, extract evidence triples
  2. KG_BUILD    — Normalize entities, persist triples to Neo4j
  3. DEMOGRAPHICS — Fetch Wikidata demographics for entity_ids
  4. INDEX        — Build FAISS dense + BM25 sparse indices (once, then reuse)
  5. RETRIEVE     — Run retrieval pipeline (semantic → KG expand → MST filter → DFS organize)
  6. GENERATE     — Generate answers via local LLM
  7. EVALUATE     — Compute accuracy, retrieval, and fairness metrics

Usage:
    python scripts/run_full_pipeline.py --config configs/experiments/kg2rag_fair.yaml
    python scripts/run_full_pipeline.py --config configs/experiments/kg2rag_fair.yaml --start-from RETRIEVE
    python scripts/run_full_pipeline.py --config configs/experiments/kg2rag_fair.yaml --only EVALUATE
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fair_kg_rag import __version__
from fair_kg_rag.utils.config import load_config, parse_cli_overrides
from fair_kg_rag.utils.io_utils import read_json, write_json
from fair_kg_rag.utils.logging_utils import setup_logging
from fair_kg_rag.utils.seed import set_global_seed

# ---------------------------------------------------------------------------
# Stage enum (ordered)
# ---------------------------------------------------------------------------

class Stage(IntEnum):
    PREPROCESS = 1
    KG_BUILD = 2
    DEMOGRAPHICS = 3
    INDEX = 4
    RETRIEVE = 5
    GENERATE = 6
    EVALUATE = 7


STAGE_NAMES = {s.name: s for s in Stage}

# ---------------------------------------------------------------------------
# Artifact manifest (for caching)
# ---------------------------------------------------------------------------

MANIFEST_NAME = "pipeline_manifest.json"


@dataclass
class StageResult:
    stage: str
    status: str  # "completed" | "skipped" | "failed"
    duration_s: float = 0.0
    artifacts: dict[str, str] = field(default_factory=dict)  # path → sha256[:16]
    message: str = ""


def _sha256_file(path: Path) -> str:
    """Return first 16 hex chars of SHA-256 for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Individual stage runners
# ---------------------------------------------------------------------------

def run_preprocess(cfg: dict, split: str, logger) -> StageResult:
    """Stage 1 — Parse dataset and chunk paragraphs."""
    from fair_kg_rag.data.corpus_manager import CorpusManager

    raw_dir = Path(cfg["paths"]["raw_data"])
    processed_dir = Path(cfg["paths"]["processed_data"])
    max_tokens = int(cfg.get("chunking", {}).get("max_tokens", 512))

    chunks_path = processed_dir / f"{split}_chunks.json"
    if chunks_path.exists():
        logger.info("Chunks already exist at %s — reusing", chunks_path)
        return StageResult(stage="PREPROCESS", status="skipped",
                           message=f"Reused {chunks_path}")

    manager = CorpusManager(raw_dir=raw_dir, processed_dir=processed_dir,
                            max_tokens=max_tokens)
    chunks, q2c = manager.process_split(split)

    artifacts = {str(chunks_path): _sha256_file(chunks_path)}
    return StageResult(
        stage="PREPROCESS", status="completed",
        artifacts=artifacts,
        message=f"{len(chunks)} chunks from {split} split",
    )


def run_kg_build(cfg: dict, split: str, logger) -> StageResult:
    """Stage 2 — Extract evidence triples → normalize → persist to Neo4j."""
    from fair_kg_rag.data.dataset_loader import load_dataset
    from fair_kg_rag.data.corpus_manager import CorpusManager
    from fair_kg_rag.kg.entity_linker import EntityLinker
    from fair_kg_rag.kg.kg_store import connect_kg, save_kg
    from fair_kg_rag.kg.triplet_extractor import (
        extract_triplets_from_evidence, triplets_to_dicts, Triplet,
    )

    raw_dir = Path(cfg["paths"]["raw_data"])
    processed_dir = Path(cfg["paths"]["processed_data"])
    kg_dir = Path(cfg["paths"]["kg_data"])
    kg_dir.mkdir(parents=True, exist_ok=True)

    # Check if Neo4j already has data — skip if populated
    neo4j_cfg = cfg.get("neo4j", {})
    if not neo4j_cfg.get("clear_on_start", False):
        try:
            kg_check = connect_kg(
                uri=neo4j_cfg.get("uri"),
                user=neo4j_cfg.get("user"),
                password=neo4j_cfg.get("password"),
                database=neo4j_cfg.get("database", "neo4j"),
                clear_on_init=False,
            )
            existing = kg_check.num_entities
            if existing > 0:
                summary = kg_check.summary()
                kg_check.close()
                logger.info(
                    "Neo4j already has %d entities — reusing existing KG", existing
                )
                return StageResult(
                    stage="KG_BUILD", status="skipped",
                    message=(
                        f"Reused Neo4j KG: {summary['num_entities']} entities, "
                        f"{summary['num_relations']} relations"
                    ),
                )
            kg_check.close()
        except Exception:
            logger.debug("Could not check Neo4j, will rebuild KG")

    # Load chunks mapping for source_chunk_id linkage
    manager = CorpusManager(raw_dir=raw_dir, processed_dir=processed_dir)
    manager.load_processed(split)
    q2c = manager._question_to_chunks or {}

    records = load_dataset(raw_dir / f"{split}.json")
    logger.info("Extracting ground-truth evidence triples from %d records", len(records))

    triplets: list[Triplet] = []
    for record in records:
        chunk_ids = q2c.get(record.id, [])
        # Link each evidence triple to ALL supporting chunks of that question
        evidence_tuples = [(ev.subject, ev.relation, ev.obj) for ev in record.evidences]
        for cid in chunk_ids:
            triplets.extend(extract_triplets_from_evidence(evidence_tuples, chunk_id=cid))
        if not chunk_ids:
            triplets.extend(extract_triplets_from_evidence(evidence_tuples, chunk_id=""))

    logger.info("Raw evidence triples: %d", len(triplets))

    # Entity normalization
    link_cfg = cfg.get("kg_extraction", {}).get("entity_linking", {})
    linker = EntityLinker(
        overlap_threshold=float(link_cfg.get("overlap_threshold", 0.90)),
        ngram_size=int(link_cfg.get("ngram_size", 3)),
    )
    linker.build_from_triplets(triplets)
    normalized = linker.normalize_triplets(triplets)
    logger.info("After normalization: %d triples, %d entity clusters",
                len(normalized), linker.num_clusters)

    # Persist to Neo4j
    neo4j_cfg = cfg.get("neo4j", {})
    kg = connect_kg(
        uri=neo4j_cfg.get("uri"),
        user=neo4j_cfg.get("user"),
        password=neo4j_cfg.get("password"),
        database=neo4j_cfg.get("database", "neo4j"),
        clear_on_init=bool(neo4j_cfg.get("clear_on_start", False)),
    )

    batch_size = int(cfg.get("kg_extraction", {}).get("neo4j_batch_size", 2000))
    for i in range(0, len(normalized), batch_size):
        kg.add_triplets(normalized[i : i + batch_size])

    summary = kg.summary()
    save_kg(kg, output_dir=kg_dir, name=f"{split}_kg")

    # Save normalized triples for inspection
    triples_path = kg_dir / f"{split}_kg_triplets.json"
    write_json(triplets_to_dicts(normalized), triples_path)

    kg.close()
    return StageResult(
        stage="KG_BUILD", status="completed",
        artifacts={str(triples_path): _sha256_file(triples_path)},
        message=f"{summary['num_entities']} entities, {summary['num_relations']} relations",
    )


def run_demographics(cfg: dict, split: str, logger) -> StageResult:
    """Stage 3 — Fetch Wikidata demographics for entity_ids."""
    from fair_kg_rag.data.dataset_loader import load_dataset
    from fair_kg_rag.data.wikidata_demographics import (
        demographics_to_dicts, fetch_demographics_batch,
    )

    processed_dir = Path(cfg["paths"]["processed_data"])
    output_path = processed_dir / "entity_demographics.json"

    if output_path.exists():
        logger.info("Demographics already cached at %s — reusing", output_path)
        return StageResult(stage="DEMOGRAPHICS", status="skipped",
                           message=f"Reused {output_path}")

    raw_dir = Path(cfg["paths"]["raw_data"])
    records = load_dataset(raw_dir / f"{split}.json")

    all_qids: set[str] = set()
    for r in records:
        all_qids.update(r.wikidata_ids)

    logger.info("Fetching demographics for %d Wikidata entities", len(all_qids))
    demographics = fetch_demographics_batch(list(all_qids))

    gendered = sum(1 for d in demographics.values() if d.gender is not None)
    logger.info("Fetched %d entities (%d with gender)", len(demographics), gendered)

    processed_dir.mkdir(parents=True, exist_ok=True)
    write_json(demographics_to_dicts(demographics), output_path)

    return StageResult(
        stage="DEMOGRAPHICS", status="completed",
        artifacts={str(output_path): _sha256_file(output_path)},
        message=f"{len(demographics)} entities, {gendered} with gender",
    )


def run_index(cfg: dict, split: str, logger) -> StageResult:
    """Stage 4 — Build FAISS + BM25 indices (skip if index file exists)."""
    from fair_kg_rag.data.corpus_manager import CorpusManager
    from fair_kg_rag.retrieval.semantic_retriever import SemanticRetriever

    processed_dir = Path(cfg["paths"]["processed_data"])
    index_dir = Path(cfg["paths"]["index_data"])
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = index_dir / f"{split}_faiss.index"
    ids_path = faiss_path.with_suffix(".ids.npy")

    if faiss_path.exists() and ids_path.exists():
        logger.info("FAISS index already exists at %s — reusing", faiss_path)
        return StageResult(stage="INDEX", status="skipped",
                           message=f"Reused {faiss_path}")

    raw_dir = Path(cfg["paths"]["raw_data"])
    manager = CorpusManager(raw_dir=raw_dir, processed_dir=processed_dir)
    chunks = manager.load_processed(split)
    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]

    dense_cfg = cfg.get("retrieval", {}).get("dense", {})
    model_name = dense_cfg.get("model_name", "BAAI/bge-base-en-v1.5")

    logger.info("Building FAISS index with %s (%d chunks)", model_name, len(chunks))
    retriever = SemanticRetriever(
        model_name=model_name,
        index_path=faiss_path,
        device=cfg.get("device", "cuda"),
    )
    retriever.build_index(chunk_ids, texts)

    return StageResult(
        stage="INDEX", status="completed",
        artifacts={str(faiss_path): _sha256_file(faiss_path)},
        message=f"Indexed {len(chunks)} chunks with {model_name}",
    )


def run_retrieve(cfg: dict, split: str, logger) -> StageResult:
    """Stage 5 — Run the full retrieval pipeline."""
    from tqdm import tqdm
    from fair_kg_rag.data.corpus_manager import CorpusManager
    from fair_kg_rag.data.dataset_loader import load_dataset
    from fair_kg_rag.kg.kg_store import load_kg
    from fair_kg_rag.retrieval.retrieval_pipeline import RetrievalPipeline

    raw_dir = Path(cfg["paths"]["raw_data"])
    processed_dir = Path(cfg["paths"]["processed_data"])
    output_dir = Path(cfg["paths"]["predictions"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load chunks
    manager = CorpusManager(raw_dir=raw_dir, processed_dir=processed_dir)
    chunks = manager.load_processed(split)
    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    chunk_texts = dict(zip(chunk_ids, texts))

    # Load demographics for fair expansion
    chunk_demographics: dict[str, dict] = {}
    demo_path = processed_dir / "entity_demographics.json"
    if demo_path.exists():
        demo_data = read_json(demo_path)
        demo_by_qid = {d["qid"]: d for d in demo_data}
        records_for_demo = load_dataset(raw_dir / f"{split}.json")
        # Map chunk → demographic via question → entity_ids → demographics
        for record in records_for_demo:
            q_chunks = manager.get_chunks_for_question(record.id)
            gender = None
            geo = None
            for qid in record.wikidata_ids:
                d = demo_by_qid.get(qid, {})
                if d.get("gender"):
                    gender = gender or d["gender"]
                if d.get("geo_group"):
                    geo = geo or d["geo_group"]
            for c in q_chunks:
                chunk_demographics[c.chunk_id] = {"gender": gender, "geo_group": geo}

    # Connect Neo4j KG if KG expansion is enabled
    kg = None
    if cfg.get("retrieval", {}).get("kg_expansion", {}).get("enabled", False):
        neo4j_cfg = cfg.get("neo4j", {})
        kg = load_kg(
            uri=neo4j_cfg.get("uri"), user=neo4j_cfg.get("user"),
            password=neo4j_cfg.get("password"),
            database=neo4j_cfg.get("database", "neo4j"),
        )

    # Setup and run pipeline (inject split for index path resolution)
    cfg_with_split = dict(cfg)
    cfg_with_split["_split"] = split
    pipeline = RetrievalPipeline(
        cfg=cfg_with_split, kg=kg, chunk_texts=chunk_texts,
        chunk_demographics=chunk_demographics,
    )
    pipeline.setup(chunk_ids, texts)

    records = load_dataset(raw_dir / f"{split}.json")
    max_samples = cfg.get("dataset", {}).get("max_samples")
    if max_samples:
        records = records[: int(max_samples)]

    results = []
    for record in tqdm(records, desc="Retrieving", unit="q"):
        ret = pipeline.retrieve(record.question)
        results.append({
            "id": record.id,
            "question": record.question,
            "answer": record.answer,
            "type": record.question_type,
            "entity_ids": record.entity_ids,
            "retrieved_chunk_ids": [cid for cid, _ in ret.retrieved_chunks],
            "retrieved_scores": [round(s, 6) for _, s in ret.retrieved_chunks],
            "context": ret.context,
            "supporting_titles": list(record.supporting_titles),
            "metadata": ret.metadata,
        })

    output_path = output_dir / f"{split}_retrieval.json"
    write_json(results, output_path)

    if kg is not None:
        kg.close()

    return StageResult(
        stage="RETRIEVE", status="completed",
        artifacts={str(output_path): _sha256_file(output_path)},
        message=f"{len(results)} queries processed",
    )


def run_generate(cfg: dict, split: str, logger) -> StageResult:
    """Stage 6 — Generate answers from retrieved contexts."""
    from tqdm import tqdm
    from fair_kg_rag.generation.generator import Generator
    from fair_kg_rag.generation.llm_backend import LLMBackend

    pred_dir = Path(cfg["paths"]["predictions"])
    retrieval_path = pred_dir / f"{split}_retrieval.json"

    if not retrieval_path.exists():
        return StageResult(stage="GENERATE", status="failed",
                           message=f"Missing {retrieval_path} — run RETRIEVE first")

    retrieval_results = read_json(retrieval_path)
    logger.info("Loaded %d retrieval results", len(retrieval_results))

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
    for item in tqdm(retrieval_results, desc="Generating", unit="q"):
        result = generator.generate(
            question=item["question"], context=item.get("context", ""),
        )
        predictions.append({
            "id": item["id"],
            "question": item["question"],
            "answer": result.answer,
            "raw_response": result.raw_response,
            "gold_answer": item.get("answer", ""),
            "type": item.get("type", ""),
            "entity_ids": item.get("entity_ids", ""),
            "retrieved_chunk_ids": item.get("retrieved_chunk_ids", []),
            "supporting_titles": item.get("supporting_titles", []),
        })

    output_path = pred_dir / f"{split}_predictions.json"
    write_json(predictions, output_path)

    return StageResult(
        stage="GENERATE", status="completed",
        artifacts={str(output_path): _sha256_file(output_path)},
        message=f"{len(predictions)} answers generated",
    )


def run_evaluate(cfg: dict, split: str, logger) -> StageResult:
    """Stage 7 — Compute accuracy + fairness metrics."""
    from fair_kg_rag.data.dataset_loader import load_dataset
    from fair_kg_rag.evaluation.evaluator import Evaluator

    raw_dir = Path(cfg["paths"]["raw_data"])
    pred_dir = Path(cfg["paths"]["predictions"])
    metrics_dir = Path(cfg["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pred_path = pred_dir / f"{split}_predictions.json"
    if not pred_path.exists():
        return StageResult(stage="EVALUATE", status="failed",
                           message=f"Missing {pred_path} — run GENERATE first")

    predictions = read_json(pred_path)
    records = load_dataset(raw_dir / f"{split}.json")

    # Build ground truths using the dataset's native format
    ground_truths = []
    for r in records:
        ground_truths.append({
            "id": r.id,
            "answer": r.answer,
            "supporting_titles": list(r.supporting_titles),
            "supporting_chunk_ids": [
                f"doc_{t.replace(' ', '_')[:80]}_0" for t in r.supporting_titles
            ],
        })

    # Annotate predictions with demographics
    processed_dir = Path(cfg["paths"]["processed_data"])
    demo_path = processed_dir / "entity_demographics.json"
    if demo_path.exists():
        demo_data = read_json(demo_path)
        demo_by_qid = {d["qid"]: d for d in demo_data}
        record_map = {r.id: r for r in records}

        for pred in predictions:
            record = record_map.get(pred["id"])
            if not record:
                continue
            for qid in record.wikidata_ids:
                d = demo_by_qid.get(qid, {})
                if d.get("gender"):
                    pred.setdefault("gender", d["gender"])
                if d.get("geo_group"):
                    pred.setdefault("geo_group", d["geo_group"])

    evaluator = Evaluator(cfg=cfg)
    result = evaluator.evaluate(
        predictions=predictions,
        ground_truths=ground_truths,
        retrieval_predictions=predictions,
    )

    # Save full metrics
    exp_name = cfg.get("experiment", {}).get("name", "default")
    metrics_path = metrics_dir / f"{split}_{exp_name}_metrics.json"
    write_json(result.metrics, metrics_path)

    # Save per-question details
    details_path = metrics_dir / f"{split}_{exp_name}_per_question.json"
    write_json(result.per_question, details_path)

    # Print summary
    acc = result.metrics.get("accuracy", {})
    summary_lines = [
        f"EM={acc.get('exact_match', 0):.4f}",
        f"F1={acc.get('answer_f1', 0):.4f}",
        f"N={acc.get('num_evaluated', 0)}",
    ]
    fairness = result.metrics.get("fairness", {})
    for attr, m in fairness.items():
        dp = m.get("demographic_parity", 0)
        summary_lines.append(f"{attr}_parity={dp:.4f}")

    return StageResult(
        stage="EVALUATE", status="completed",
        artifacts={str(metrics_path): _sha256_file(metrics_path)},
        message=" | ".join(summary_lines),
    )


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

STAGE_RUNNERS = {
    Stage.PREPROCESS: run_preprocess,
    Stage.KG_BUILD: run_kg_build,
    Stage.DEMOGRAPHICS: run_demographics,
    Stage.INDEX: run_index,
    Stage.RETRIEVE: run_retrieve,
    Stage.GENERATE: run_generate,
    Stage.EVALUATE: run_evaluate,
}


def _load_manifest(output_dir: Path) -> dict:
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        return read_json(manifest_path)
    return {}


def _save_manifest(output_dir: Path, manifest: dict) -> None:
    write_json(manifest, output_dir / MANIFEST_NAME)


def run_pipeline(
    cfg: dict,
    split: str,
    start_from: Stage | None = None,
    only: Stage | None = None,
    logger=None,
) -> dict[str, StageResult]:
    """Run the full (or partial) pipeline.

    Args:
        cfg: Merged configuration dict.
        split: Dataset split.
        start_from: Skip all stages before this one.
        only: Run only this single stage.
        logger: Logger instance.

    Returns:
        Dict mapping stage name to StageResult.
    """
    output_dir = Path(cfg["paths"]["output_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(output_dir)

    exp_name = cfg.get("experiment", {}).get("name", "default")
    exp_desc = cfg.get("experiment", {}).get("description", "")

    # Run header
    logger.info("=" * 72)
    logger.info("Fair KG-Enhanced RAG Pipeline v%s", __version__)
    logger.info("Experiment : %s", exp_name)
    logger.info("Description: %s", exp_desc)
    logger.info("Split      : %s", split)
    logger.info("Device     : %s", cfg.get("device", "cpu"))
    logger.info("Python     : %s", platform.python_version())
    logger.info("Platform   : %s", platform.platform())
    logger.info("Timestamp  : %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 72)

    results: dict[str, StageResult] = {}
    pipeline_start = time.perf_counter()

    for stage in Stage:
        if only is not None and stage != only:
            continue
        if start_from is not None and stage < start_from:
            logger.info("[%s] skipped (--start-from %s)", stage.name, start_from.name)
            continue

        runner = STAGE_RUNNERS[stage]
        logger.info("")
        logger.info("─" * 72)
        logger.info("  Stage %d/%d: %s", stage.value, len(Stage), stage.name)
        logger.info("─" * 72)

        t0 = time.perf_counter()
        try:
            sr = runner(cfg, split, logger)
        except Exception as exc:
            logger.exception("Stage %s FAILED", stage.name)
            sr = StageResult(stage=stage.name, status="failed", message=str(exc))

        sr.duration_s = round(time.perf_counter() - t0, 2)
        results[stage.name] = sr

        status_icon = {"completed": "✓", "skipped": "»", "failed": "✗"}.get(sr.status, "?")
        logger.info(
            "  %s %s  [%s]  %.1fs  — %s",
            status_icon, stage.name, sr.status, sr.duration_s, sr.message,
        )

        # Update manifest
        manifest.setdefault("stages", {})[stage.name] = {
            "status": sr.status,
            "duration_s": sr.duration_s,
            "artifacts": sr.artifacts,
            "message": sr.message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _save_manifest(output_dir, manifest)

        if sr.status == "failed":
            logger.error("Pipeline halted at stage %s", stage.name)
            break

    total = round(time.perf_counter() - pipeline_start, 2)

    # Final summary
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PIPELINE SUMMARY — %s — %.1fs total", exp_name, total)
    logger.info("=" * 72)
    for name, sr in results.items():
        icon = {"completed": "✓", "skipped": "»", "failed": "✗"}.get(sr.status, "?")
        logger.info("  %s %-14s %8.1fs  %s", icon, name, sr.duration_s, sr.message)
    logger.info("=" * 72)

    # Save run record
    manifest["last_run"] = {
        "experiment": exp_name,
        "split": split,
        "version": __version__,
        "total_duration_s": total,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stages": {
            k: {"status": v.status, "duration_s": v.duration_s, "message": v.message}
            for k, v in results.items()
        },
    }
    _save_manifest(output_dir, manifest)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fair KG-Enhanced RAG — Full Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment config YAML")
    parser.add_argument("--split", default="dev", help="Dataset split (default: dev)")
    parser.add_argument(
        "--start-from",
        choices=[s.name for s in Stage],
        default=None,
        help="Skip stages before this one",
    )
    parser.add_argument(
        "--only",
        choices=[s.name for s in Stage],
        default=None,
        help="Run only this single stage",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=parse_cli_overrides())
    set_global_seed(cfg.get("seed", 42))

    log_dir = Path(cfg.get("paths", {}).get("logs", "outputs/logs"))
    exp_name = cfg.get("experiment", {}).get("name", "default")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{exp_name}_{args.split}_{ts}.log"

    logger = setup_logging(
        name="pipeline",
        level=cfg.get("logging", {}).get("level", "INFO"),
        log_file=log_file if cfg.get("logging", {}).get("log_to_file", True) else None,
    )

    start_from = STAGE_NAMES.get(args.start_from) if args.start_from else None
    only = STAGE_NAMES.get(args.only) if args.only else None

    run_pipeline(cfg, args.split, start_from=start_from, only=only, logger=logger)


if __name__ == "__main__":
    main()
