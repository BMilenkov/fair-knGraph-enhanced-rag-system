# Fair KG-Enhanced RAG System

> **v0.2.0** | Knowledge Graph-Guided Retrieval-Augmented Generation with Demographic Fairness

A research system that combines **KG²RAG** (knowledge graph-guided retrieval) with **fairness-aware expansion** for multi-hop question answering. Built on Neo4j, evaluated on 2WikiMultiHopQA.

## Research Context

| Paper | Venue | Role |
|-------|-------|------|
| [KG²RAG](https://aclanthology.org/2025.naacl-long.449/) | NAACL 2025 | KG-guided retrieval: BFS expansion + MST filtering + DFS context |
| [RAG Fairness](https://aclanthology.org/2025.coling-main.669/) | COLING 2025 | Fairness evaluation framework across gender/geography |

**Novel contribution:** Fairness-aware BFS traversal that prevents demographic bias from compounding across multi-hop reasoning steps.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        PIPELINE v0.2.0                           │
├──────────┬───────────┬────────────┬───────┬────────┬────────────┤
│ 1.PREPROCESS │ 2.KG_BUILD │ 3.DEMOGRAPHICS │ 4.INDEX │ 5.RETRIEVE │ 6.GENERATE │ 7.EVALUATE │
└──────────┴───────────┴────────────┴───────┴────────┴────────────┘
     │            │            │           │          │          │          │
   Parse &     Evidence    Wikidata     FAISS +   Semantic →  Local    EM, F1,
   chunk       triples →   SPARQL →    BM25      KG expand → LLM →   Fairness,
   dataset     Neo4j KG    demographics indices   MST filter  answer   Bootstrap CI
                                                  → DFS org
                                                  → rerank
```

**Retrieval Pipeline Detail (Stage 5):**
```
Query → Dense Retrieval (FAISS) → Seed chunks
      → KG Expansion (BFS over Neo4j) → Expanded chunks
      → MST Subgraph Filtering → Filtered chunks
      → DFS Context Organization → Ordered context
      → Cross-encoder Reranking → Final context → LLM
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Knowledge Graph | **Neo4j** (Cypher queries, MERGE/UNWIND batching) |
| Dense Retrieval | FAISS IndexFlatIP + sentence-transformers (BGE) |
| Sparse Retrieval | BM25 (rank_bm25) |
| Graph Algorithms | NetworkX (MST, DFS on Neo4j-extracted subgraphs) |
| Reranking | FlagEmbedding cross-encoder |
| LLM Backend | HuggingFace transformers (Mistral-7B, 4-bit quantized) |
| Demographics | Wikidata SPARQL (P21 gender, P27 nationality) |
| Configuration | OmegaConf YAML with env var interpolation |
| Dataset | [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) (COLING 2020) |

## Quick Start

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

### 2. Neo4j Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
```

### 3. Download Dataset

```bash
python scripts/download_data.py
```

### 4. Run an Experiment

```bash
# Full pipeline (all 7 stages)
python scripts/run_full_pipeline.py --config configs/experiments/kg2rag_fair.yaml

# Or use the experiment shorthand
python scripts/run_experiment.py --experiment kg2rag_fair

# Resume from a specific stage
python scripts/run_full_pipeline.py --config configs/experiments/kg2rag_fair.yaml --start-from RETRIEVE

# Run only evaluation
python scripts/run_full_pipeline.py --config configs/experiments/kg2rag_fair.yaml --only EVALUATE
```

## Pipeline Stages

| # | Stage | Description | Caching |
|---|-------|-------------|---------|
| 1 | `PREPROCESS` | Parse 2WikiMultiHopQA, chunk paragraphs, extract evidence triples | Skipped if `{split}_chunks.json` exists |
| 2 | `KG_BUILD` | Normalize entities, persist triples to Neo4j via UNWIND batching | Rebuilds each run (Neo4j is source of truth) |
| 3 | `DEMOGRAPHICS` | Fetch Wikidata demographics (gender, nationality, geo group) | Skipped if `entity_demographics.json` exists |
| 4 | `INDEX` | Build FAISS dense + BM25 sparse indices | Skipped if `{split}_faiss.index` exists |
| 5 | `RETRIEVE` | Semantic retrieval → KG expansion → MST filter → DFS organize → rerank | Always runs (experiment-dependent) |
| 6 | `GENERATE` | Generate answers via local LLM from organized contexts | Always runs |
| 7 | `EVALUATE` | Compute EM, F1, retrieval metrics, fairness metrics, bootstrap CI | Always runs |

Stages 1-4 are cached (built once, reused across experiments). Stages 5-7 run per experiment.

## Experiments

| Experiment | Config | Description |
|-----------|--------|-------------|
| `baseline_naive` | No KG, no reranker | Standard dense retrieval baseline |
| `kg2rag_standard` | KG expansion + MST + DFS | KG²RAG replication |
| **`kg2rag_fair`** | Fair expansion + MST + DFS | **Our contribution: fairness-aware BFS** |
| `ablation_no_mst` | KG expansion, no MST | Ablation: skip graph filtering |
| `ablation_no_fairness` | Standard expansion only | Ablation: disable fairness balancing |

Run all experiments:
```bash
make exp-all SPLIT=dev
```

## Fairness-Aware KG Expansion

The key insight: in multi-hop reasoning, **bias compounds across hops**. If hop 1 retrieves gender-skewed context, hop 2 expands from a biased seed set, amplifying the skew.

Three strategies implemented in `fair_kg_expander.py`:

| Strategy | Approach | When to use |
|----------|----------|-------------|
| `post_hoc` | Standard BFS, then resample to balance demographics | Quick baseline |
| `in_traversal` | Adjust entity scores during BFS based on running distribution | **Default — most novel** |
| `constraint` | Hard cap: no group exceeds `max_group_ratio` in final context | Strict fairness requirement |

## Project Structure

```
src/fair_kg_rag/
├── data/               # Dataset loading, chunking, Wikidata demographics
├── kg/                 # Triplet extraction, Neo4j KG builder, entity linking
├── retrieval/          # Dense/sparse retrieval, KG expansion, MST filter, DFS, reranker
├── generation/         # LLM backend, prompt templates, response parsing
├── evaluation/         # Accuracy, fairness, retrieval metrics, statistical tests
└── utils/              # Config, logging, IO, text processing, seeds

scripts/                # CLI pipeline runners (download, extract, build, retrieve, generate, evaluate)
configs/                # YAML configs (base + per-stage + experiments)
tests/                  # Unit + integration tests
notebooks/              # Analysis notebooks (data exploration → case studies)
```

## Makefile Commands

```bash
make help               # Show all available commands
make setup              # Install dependencies
make lint               # Run ruff linter
make test               # Run all tests
make pipeline           # Run full pipeline (CONFIG=... SPLIT=...)
make experiment         # Run named experiment (EXPERIMENT=... SPLIT=...)
make exp-all            # Run ALL experiments
make clean              # Remove generated outputs
```

## Configuration

All behavior is config-driven via OmegaConf YAML files. No hardcoded paths or model names.

```
configs/
├── base.yaml                     # Shared defaults (paths, seed, device, Neo4j)
├── kg_extraction.yaml            # Evidence extraction + entity linking
├── retrieval.yaml                # Dense/sparse/KG expansion/MST/DFS/reranker
├── generation.yaml               # LLM model, temperature, quantization
├── evaluation.yaml               # Metrics, fairness flags, bootstrap params
├── wikidata_demographics.yaml    # SPARQL query config
└── experiments/                  # Per-experiment overrides
```

Override any config value via CLI:
```bash
python scripts/run_full_pipeline.py --config configs/experiments/kg2rag_fair.yaml \
    retrieval.kg_expansion.max_hops=3 generation.temperature=0.2
```

## Evaluation Metrics

**Accuracy:** Exact Match (EM), Answer F1, Support F1

**Retrieval:** MRR@k, Recall@k, Precision@k

**Fairness:** Demographic Parity, Equalized Odds, Retrieval Fairness (per gender/geography group)

**Statistical:** Bootstrap confidence intervals, paired permutation tests

## Testing

```bash
pytest tests/unit -v          # Unit tests (no Neo4j required)
pytest tests/integration -v   # Integration tests
pytest tests -v --tb=short    # All tests
```

## References

1. **KG²RAG**: Guan et al. "Knowledge Graph-Guided Retrieval Augmented Generation" (NAACL 2025)
2. **RAG Fairness**: Wang et al. "Evaluating Fairness in RAG" (COLING 2025)
3. **2WikiMultiHopQA**: Ho et al. "Constructing A Multi-Hop QA Dataset for Comprehensive Evaluation" (COLING 2020)
