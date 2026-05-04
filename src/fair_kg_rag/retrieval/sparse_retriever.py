"""BM25-based sparse retrieval."""

from __future__ import annotations

import logging

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class SparseRetriever:
    """BM25 sparse retrieval over text chunks.

    Args:
        k1: BM25 term frequency saturation parameter.
        b: BM25 document length normalization parameter.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []

    def build_index(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Build BM25 index from chunk texts.

        Args:
            chunk_ids: List of chunk identifiers.
            texts: Corresponding chunk texts.
        """
        self._chunk_ids = chunk_ids
        tokenized = [text.lower().split() for text in texts]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        logger.info(f"Built BM25 index: {len(chunk_ids)} documents")

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: Query text.
            top_k: Number of results to return.

        Returns:
            List of (chunk_id, bm25_score) tuples, sorted descending.
        """
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]
        return [
            (self._chunk_ids[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]
