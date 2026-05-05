"""Corpus management: load raw data, chunk, persist, and access."""

from __future__ import annotations

import logging
from pathlib import Path

from fair_kg_rag.data.dataset_loader import load_dataset
from fair_kg_rag.data.preprocessor import Chunk, chunks_to_dicts, preprocess_dataset
from fair_kg_rag.utils.io_utils import read_json, write_json

logger = logging.getLogger(__name__)


class CorpusManager:
    """Manages chunking and persistence of the document corpus.

    Args:
        raw_dir: Directory containing raw 2WikiMultiHopQA JSON files.
        processed_dir: Directory for processed chunks.
        max_tokens: Max words per chunk.
    """

    def __init__(
        self,
        raw_dir: str | Path,
        processed_dir: str | Path,
        max_tokens: int = 512,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.max_tokens = max_tokens
        self._chunks: list[Chunk] = []
        self._chunk_map: dict[str, Chunk] = {}
        self._question_to_chunks: dict[str, list[str]] = {}

    def process_split(self, split: str = "dev") -> tuple[list[Chunk], dict[str, list[str]]]:
        """Load raw dataset, chunk paragraphs, save to disk."""
        raw_path = self.raw_dir / f"{split}.json"
        if not raw_path.exists():
            raise FileNotFoundError(f"Dataset not found: {raw_path}")

        records = load_dataset(raw_path)
        logger.info("Loaded %d records from %s", len(records), raw_path)

        chunks, q2c = preprocess_dataset(records, self.max_tokens)
        logger.info("Created %d chunks", len(chunks))

        # Persist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        write_json(chunks_to_dicts(chunks), self.processed_dir / f"{split}_chunks.json")
        write_json(q2c, self.processed_dir / f"{split}_question_chunks.json")

        self._chunks = chunks
        self._chunk_map = {c.chunk_id: c for c in chunks}
        self._question_to_chunks = q2c
        return chunks, q2c

    def load_processed(self, split: str = "dev") -> list[Chunk]:
        """Load previously processed chunks from disk."""
        chunks_path = self.processed_dir / f"{split}_chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(f"No processed chunks: {chunks_path}")

        raw = read_json(chunks_path)
        self._chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                source_title=c["source_title"],
                text=c["text"],
                question_ids=c.get("question_ids", []),
                is_supporting=c.get("is_supporting", False),
            )
            for c in raw
        ]
        self._chunk_map = {c.chunk_id: c for c in self._chunks}

        mapping_path = self.processed_dir / f"{split}_question_chunks.json"
        if mapping_path.exists():
            self._question_to_chunks = read_json(mapping_path)

        return self._chunks

    def get_chunks_for_question(self, question_id: str) -> list[Chunk]:
        """Get all chunks associated with a question."""
        chunk_ids = self._question_to_chunks.get(question_id, [])
        return [self._chunk_map[cid] for cid in chunk_ids if cid in self._chunk_map]

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)
