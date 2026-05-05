"""Utility modules for config, I/O, text processing, logging, and seeds."""

from fair_kg_rag.utils.config import load_config
from fair_kg_rag.utils.io_utils import read_json, write_json
from fair_kg_rag.utils.logging_utils import get_logger, setup_logging
from fair_kg_rag.utils.seed import set_global_seed
from fair_kg_rag.utils.text_utils import ngram_overlap, normalize_text

__all__ = [
    "get_logger",
    "load_config",
    "ngram_overlap",
    "normalize_text",
    "read_json",
    "set_global_seed",
    "setup_logging",
    "write_json",
]
