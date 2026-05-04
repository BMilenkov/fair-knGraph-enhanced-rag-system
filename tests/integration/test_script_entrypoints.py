"""Integration smoke tests for CLI script entrypoints."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def _has_dependency(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def test_script_help_entrypoints() -> None:
    required = [
        "omegaconf",
        "requests",
        "neo4j",
        "rank_bm25",
        "sentence_transformers",
        "faiss",
        "transformers",
        "torch",
    ]
    missing = [name for name in required if not _has_dependency(name)]
    if missing:
        pytest.skip(f"Missing runtime dependencies for script smoke test: {missing}")

    root = Path(__file__).resolve().parents[2]

    scripts = [
        "extract_kg.py",
        "build_index.py",
        "run_retrieval.py",
        "run_generation.py",
        "evaluate.py",
        "run_experiment.py",
        "run_full_pipeline.py",
        "fetch_demographics.py",
    ]

    for script_name in scripts:
        script_path = root / "scripts" / script_name
        completed = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=str(root),
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, (
            f"{script_name} --help failed: stdout={completed.stdout} stderr={completed.stderr}"
        )
