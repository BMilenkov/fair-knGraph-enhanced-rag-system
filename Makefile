# Fair KG-Enhanced RAG System — Makefile
# ========================================
# Common commands for development and experimentation.

PYTHON     ?= python
SPLIT      ?= dev
CONFIG     ?= configs/experiments/kg2rag_fair.yaml
EXPERIMENT ?= kg2rag_fair

.PHONY: help setup lint test test-unit test-integration \
        download preprocess extract-kg index demographics \
        retrieve generate evaluate pipeline experiment clean

# ────────────────────────────────────────
# Help
# ────────────────────────────────────────
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ────────────────────────────────────────
# Setup
# ────────────────────────────────────────
setup:  ## Install all dependencies
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# ────────────────────────────────────────
# Quality
# ────────────────────────────────────────
lint:  ## Run ruff linter on src, tests, scripts
	ruff check src tests scripts

lint-fix:  ## Auto-fix lint issues
	ruff check --fix src tests scripts

test: test-unit test-integration  ## Run all tests

test-unit:  ## Run unit tests
	$(PYTHON) -m pytest tests/unit -v --tb=short

test-integration:  ## Run integration tests
	$(PYTHON) -m pytest tests/integration -v --tb=short

# ────────────────────────────────────────
# Data Pipeline (individual stages)
# ────────────────────────────────────────
download:  ## Download 2WikiMultiHopQA dataset
	$(PYTHON) scripts/download_data.py

demographics:  ## Fetch Wikidata demographics for entity_ids
	$(PYTHON) scripts/fetch_demographics.py --config configs/wikidata_demographics.yaml --split $(SPLIT)

extract-kg:  ## Extract & persist KG triples to Neo4j
	$(PYTHON) scripts/extract_kg.py --config configs/kg_extraction.yaml --split $(SPLIT)

index:  ## Build FAISS + BM25 retrieval indices
	$(PYTHON) scripts/build_index.py --config configs/retrieval.yaml --split $(SPLIT)

retrieve:  ## Run retrieval pipeline
	$(PYTHON) scripts/run_retrieval.py --config configs/retrieval.yaml --split $(SPLIT)

generate:  ## Generate answers from retrieved contexts
	$(PYTHON) scripts/run_generation.py --config configs/generation.yaml --split $(SPLIT)

evaluate:  ## Compute accuracy + fairness metrics
	$(PYTHON) scripts/evaluate.py --config configs/evaluation.yaml --split $(SPLIT)

# ────────────────────────────────────────
# Full Pipeline & Experiments
# ────────────────────────────────────────
pipeline:  ## Run full end-to-end pipeline (CONFIG=... SPLIT=...)
	$(PYTHON) scripts/run_full_pipeline.py --config $(CONFIG) --split $(SPLIT)

experiment:  ## Run named experiment (EXPERIMENT=... SPLIT=...)
	$(PYTHON) scripts/run_experiment.py --experiment $(EXPERIMENT) --split $(SPLIT)

# Predefined experiments
exp-baseline:  ## Run baseline naive RAG
	$(PYTHON) scripts/run_experiment.py --experiment baseline_naive --split $(SPLIT)

exp-standard:  ## Run standard KG²RAG
	$(PYTHON) scripts/run_experiment.py --experiment kg2rag_standard --split $(SPLIT)

exp-fair:  ## Run fairness-aware KG²RAG (our contribution)
	$(PYTHON) scripts/run_experiment.py --experiment kg2rag_fair --split $(SPLIT)

exp-ablation-mst:  ## Ablation: skip MST filtering
	$(PYTHON) scripts/run_experiment.py --experiment ablation_no_mst --split $(SPLIT)

exp-ablation-fair:  ## Ablation: disable fairness balancing
	$(PYTHON) scripts/run_experiment.py --experiment ablation_no_fairness --split $(SPLIT)

exp-all:  ## Run ALL experiments sequentially
	@echo "=== Running all experiments on split=$(SPLIT) ==="
	$(MAKE) exp-baseline
	$(MAKE) exp-standard
	$(MAKE) exp-fair
	$(MAKE) exp-ablation-mst
	$(MAKE) exp-ablation-fair
	@echo "=== All experiments complete ==="

# ────────────────────────────────────────
# Cleanup
# ────────────────────────────────────────
clean:  ## Remove generated outputs (predictions, metrics, logs)
	rm -rf outputs/predictions/* outputs/metrics/* outputs/logs/*

clean-indices:  ## Remove built indices (forces rebuild)
	rm -rf data/indices/*

clean-all: clean clean-indices  ## Remove all generated artifacts
	rm -rf data/processed/* data/kg/*
