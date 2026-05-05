"""Data loading, preprocessing, and demographic annotation modules."""

from fair_kg_rag.data.corpus_manager import CorpusManager
from fair_kg_rag.data.dataset_loader import QARecord, load_dataset
from fair_kg_rag.data.preprocessor import Chunk, preprocess_dataset

__all__ = [
    "Chunk",
    "CorpusManager",
    "QARecord",
    "load_dataset",
    "preprocess_dataset",
]
