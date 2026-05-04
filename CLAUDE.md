# Fair KG-Enhanced RAG System

## Project Overview
Combines KG²RAG (knowledge graph-guided retrieval) with fairness evaluation to build a Fair
KG-Enhanced RAG system for multi-hop QA with demographic fairness.
Dataset: 2WikiMultiHopQA. Reference papers: KG²RAG (NAACL 2025), RAG Fairness (COLING 2025).

## Tech Stack
- Python 3.10+, pip, venv
- LLM backend: HuggingFace transformers (local models, e.g., Llama-3-8B, Mistral-7B)
- KG storage: **Neo4j** (graph DB) + NetworkX (subgraph algorithms)
- Retrieval: FAISS (dense), rank_bm25 (sparse), sentence-transformers
- Demographics: Wikidata SPARQL via requests
- Config: OmegaConf YAML files
- Testing: pytest
- Linting: ruff (100-char line limit)

## Conventions
- Google-style docstrings, type hints everywhere
- 100-character line width
- Config-driven design: no hardcoded paths or model names
- Conventional commits: feat:, fix:, docs:, refactor:, test:, chore:

## Key Commands
```bash
# Setup
python -m venv .venv && source .venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Neo4j: must be running at bolt://localhost:7687
# Set NEO4J_PASSWORD env var or add to configs/base.yaml

# Data
python scripts/download_data.py
python scripts/fetch_demographics.py

# Full pipeline (recommended — smart caching, versioned, with logs)
python scripts/run_full_pipeline.py --config configs/experiments/kg2rag_fair.yaml
python scripts/run_full_pipeline.py --config configs/experiments/kg2rag_fair.yaml --start-from RETRIEVE
python scripts/run_full_pipeline.py --config configs/experiments/kg2rag_fair.yaml --only EVALUATE

# Named experiment shorthand
python scripts/run_experiment.py --experiment kg2rag_fair
python scripts/run_experiment.py --experiment baseline_naive --only EVALUATE

# Individual stages (for development)
python scripts/extract_kg.py --config configs/kg_extraction.yaml
python scripts/build_index.py --config configs/retrieval.yaml
python scripts/run_retrieval.py --config configs/retrieval.yaml
python scripts/run_generation.py --config configs/generation.yaml
python scripts/evaluate.py --config configs/evaluation.yaml

# Tests
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/ -v --tb=short
```

## Pipeline Stages
The full pipeline (`run_full_pipeline.py`) runs 7 stages with smart caching:

| # | Stage       | Description                                    | Cached? |
|---|-------------|------------------------------------------------|---------|
| 1 | PREPROCESS  | Parse 2WikiMultiHopQA, chunk paragraphs        | Yes     |
| 2 | KG_BUILD    | Evidence triples → normalize → Neo4j           | Neo4j   |
| 3 | DEMOGRAPHICS| Fetch Wikidata gender/nationality for entities | Yes     |
| 4 | INDEX       | Build FAISS dense + BM25 indices               | Yes     |
| 5 | RETRIEVE    | Semantic → KG expand → MST filter → DFS order  | No      |
| 6 | GENERATE    | Local LLM answer generation                    | No      |
| 7 | EVALUATE    | Accuracy + fairness + statistical metrics       | No      |

Stages 1-4 are built once and reused. Re-running only rebuilds stages 5-7.

## Dataset Format (2WikiMultiHopQA)
Each sample has: `_id`, `question`, `answer`, `supporting_facts` (title + sent_id),
`context` (title + sentences), `evidences` (subject, relation, object triples from Wikidata),
`type` (comparison|inference|compositional|bridge_comparison), `entity_ids` (Wikidata QIDs).

Key insight: we use `evidences` directly as KG triples (no LLM extraction needed)
and `entity_ids` for ground-truth demographics from Wikidata.

## Data Layout
- `data/raw/` — 2WikiMultiHopQA downloads (gitignored)
- `data/processed/` — Chunked data + demographics (gitignored)
- `data/kg/` — Triplet snapshots + Neo4j metadata (gitignored)
- `data/indices/` — FAISS vector indices (gitignored)
- `outputs/` — Predictions, metrics, logs, manifests (gitignored)

## Module Organization
- `src/fair_kg_rag/data/` — Dataset loading, preprocessing, Wikidata demographics
- `src/fair_kg_rag/kg/` — Triplet extraction, Neo4j KG builder, entity linking
- `src/fair_kg_rag/retrieval/` — Dense/sparse retrieval, KG expansion, fairness-aware expansion
- `src/fair_kg_rag/generation/` — HuggingFace LLM backend, prompts, answer generation
- `src/fair_kg_rag/evaluation/` — Accuracy, fairness, retrieval metrics, statistical tests
- `src/fair_kg_rag/utils/` — Config, logging, IO, text processing, seeds
