"""Corpus lifecycle management: load, chunk, persist, and iterate."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from fair_kg_rag.data.dataset_loader import QARecord, load_dataset
from fair_kg_rag.data.preprocessor import (
    Chunk,
    chunks_to_dicts,
    preprocess_dataset,
)
from fair_kg_rag.utils.io_utils import read_json, write_json

logger = logging.getLogger(__name__)


class CorpusManager:
    """Manages the document corpus lifecycle for the RAG pipeline.

    Handles loading raw data, chunking, persisting processed chunks,
    and providing efficient access during retrieval.

    Args:
        raw_dir: Directory containing raw 2WikiMultiHopQA JSON files.
        processed_dir: Directory for storing processed chunks.
        max_tokens: ~ Maximum word count per chunk.
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
        self._chunks: list[Chunk] | None = None
        self._chunk_index: dict[str, Chunk] | None = None
        self._question_to_chunks: dict[str, list[str]] | None = None

    def process_split(self, split: str = "dev") -> tuple[list[Chunk], dict[str, list[str]]]:
        """Process a dataset split: load, chunk, and persist.

        Args:
            split: Dataset split name (train, dev, test).

        Returns:
            Tuple of (chunks, question_to_chunk_ids mapping).
        """
        raw_path = self.raw_dir / f"{split}.json"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {raw_path}. "
                f"Run 'python scripts/download_data.py' first."
            )

        logger.info(f"Loading {split} split from {raw_path}")
        records = load_dataset(raw_path)
        logger.info(f"Loaded {len(records)} records")

        logger.info("Chunking paragraphs...")
        chunks, question_to_chunks = preprocess_dataset(records, self.max_tokens)
        logger.info(f"Created {len(chunks)} chunks")

        # Persist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = self.processed_dir / f"{split}_chunks.json"
        mapping_path = self.processed_dir / f"{split}_question_chunks.json"

        write_json(chunks_to_dicts(chunks), chunks_path)
        write_json(question_to_chunks, mapping_path)
        logger.info(f"Saved processed data to {self.processed_dir}")

        self._chunks = chunks
        self._question_to_chunks = question_to_chunks
        self._build_index()

        return chunks, question_to_chunks

    def load_processed(self, split: str = "dev") -> list[Chunk]:
        """Load previously processed chunks from disk.

        Args:
            split: Dataset split name.

        Returns:
            List of Chunk objects.
        """
        chunks_path = self.processed_dir / f"{split}_chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Processed chunks not found: {chunks_path}. "
                f"Run process_split('{split}') first."
            )

        raw_chunks = read_json(chunks_path)
        self._chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                source_title=c["source_title"],
                text=c["text"],
                question_ids=c.get("question_ids", []),
                is_supporting=c.get("is_supporting", False),
                sentence_indices=c.get("sentence_indices", []),
            )
            for c in raw_chunks
        ]

        mapping_path = self.processed_dir / f"{split}_question_chunks.json"
        if mapping_path.exists():
            self._question_to_chunks = read_json(mapping_path)

        self._build_index()
        return self._chunks

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a single chunk by ID.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            Chunk object or None if not found.
        """
        if self._chunk_index is None:
            return None
        return self._chunk_index.get(chunk_id)

    def get_chunks_for_question(self, question_id: str) -> list[Chunk]:
        """Get all chunks associated with a question.

        Args:
            question_id: The question identifier.

        Returns:
            List of associated Chunk objects.
        """
        if self._question_to_chunks is None or self._chunk_index is None:
            return []
        chunk_ids = self._question_to_chunks.get(question_id, [])
        return [
            self._chunk_index[cid]
            for cid in chunk_ids
            if cid in self._chunk_index
        ]

    def iter_chunks(self, batch_size: int = 100) -> Iterator[list[Chunk]]:
        """Iterate over chunks in batches.

        Args:
            batch_size: Number of chunks per batch.

        Yields:
            Batches of Chunk objects.
        """
        if self._chunks is None:
            return
        for i in range(0, len(self._chunks), batch_size):
            yield self._chunks[i : i + batch_size]

    @property
    def num_chunks(self) -> int:
        """Return the total number of chunks."""
        return len(self._chunks) if self._chunks else 0

    def _build_index(self) -> None:
        """Build the chunk_id -> Chunk lookup index."""
        if self._chunks is not None:
            self._chunk_index = {c.chunk_id: c for c in self._chunks}
