# Fair KG-Enhanced RAG System

## Project Overview
Combines KG²RAG (knowledge graph-guided retrieval) with fairness evaluation to build a Fair
KG-Enhanced RAG system for multi-hop QA with demographic fairness.
Dataset: 2WikiMultiHopQA. Reference papers: KG²RAG (NAACL 2025), RAG Fairness (COLING 2025).

## Tech Stack
- Python 3.10+, pip
- LLM backend: HuggingFace transformers (Mistral-7B, 4-bit quantized via bitsandbytes)
- KG storage: **Neo4j AuraDB** (free tier, cloud) + NetworkX (subgraph algorithms)
- Retrieval: FAISS (dense), rank_bm25 (sparse), sentence-transformers (BGE)
- Demographics: Wikidata SPARQL via requests
- Config: OmegaConf YAML with env var interpolation
- Runtime: **Google Colab** (T4 GPU) recommended

## Running on Google Colab
1. Open `notebooks/00_colab_setup.ipynb` in Colab
2. Set runtime to T4 GPU
3. Set Neo4j AuraDB credentials (NEO4J_URI, NEO4J_PASSWORD)
4. Run cells in order

## Key Commands
```bash
# Data
python scripts/download_data.py
python scripts/fetch_demographics.py

# Full pipeline (smart caching — stages 1-4 built once, 5-7 re-run per experiment)
python scripts/run_full_pipeline.py --config configs/experiments/kg2rag_fair.yaml
python scripts/run_full_pipeline.py --config configs/experiments/kg2rag_fair.yaml --start-from RETRIEVE
python scripts/run_full_pipeline.py --config configs/experiments/kg2rag_fair.yaml --only EVALUATE

# Named experiment shorthand
python scripts/run_experiment.py --experiment kg2rag_fair
python scripts/run_experiment.py --experiment baseline_naive --only EVALUATE

# Quick test (100 samples)
python scripts/run_full_pipeline.py --config configs/experiments/kg2rag_fair.yaml dataset.max_samples=100
```

## Neo4j AuraDB Setup
1. Create free instance at https://neo4j.io/cloud/aura-free/
2. Set env vars before running pipeline:
   ```bash
   export NEO4J_URI="neo4j+s://xxxxxxxx.databases.neo4j.io"
   export NEO4J_PASSWORD="your-password"
   ```
3. AuraDB uses `neo4j+s://` protocol (encrypted); local Neo4j uses `bolt://`

## Pipeline Stages
| # | Stage       | Description                                    | Cached? |
|---|-------------|------------------------------------------------|---------|
| 1 | PREPROCESS  | Parse 2WikiMultiHopQA, chunk paragraphs        | Yes     |
| 2 | KG_BUILD    | Evidence triples → normalize → Neo4j AuraDB    | Neo4j   |
| 3 | DEMOGRAPHICS| Fetch Wikidata gender/nationality for entities | Yes     |
| 4 | INDEX       | Build FAISS dense + BM25 indices               | Yes     |
| 5 | RETRIEVE    | Semantic → KG expand → MST filter → DFS order  | No      |
| 6 | GENERATE    | Local LLM answer generation (Mistral-7B 4-bit) | No      |
| 7 | EVALUATE    | Accuracy + fairness + statistical metrics       | No      |

## Dataset Format (2WikiMultiHopQA)
Each sample has: `_id`, `question`, `answer`, `supporting_facts` (title + sent_id),
`context` (title + sentences), `evidences` (subject, relation, object triples from Wikidata),
`type` (comparison|inference|compositional|bridge_comparison), `entity_ids` (Wikidata QIDs).

Key insight: we use `evidences` directly as KG triples (no LLM extraction needed)
and `entity_ids` for ground-truth demographics from Wikidata.

## Conventions
- Google-style docstrings, type hints everywhere
- 100-character line width
- Config-driven design: no hardcoded paths or model names

## Module Organization
- `src/fair_kg_rag/data/` — Dataset loading, preprocessing, Wikidata demographics
- `src/fair_kg_rag/kg/` — Triplet extraction, Neo4j KG builder, entity linking
- `src/fair_kg_rag/retrieval/` — Dense/sparse retrieval, KG expansion, fairness-aware expansion
- `src/fair_kg_rag/generation/` — HuggingFace LLM backend, prompts, answer generation
- `src/fair_kg_rag/evaluation/` — Accuracy, fairness, retrieval metrics, statistical tests
- `src/fair_kg_rag/utils/` — Config, logging, IO, text processing, seeds
