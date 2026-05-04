"""Cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker using BGE or similar models.

    Args:
        model_name: Cross-encoder model name.
        device: Device for inference.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._reranker = None

    def _load_model(self) -> None:
        """Lazy-load the reranker model."""
        if self._reranker is None:
            from FlagEmbedding import FlagReranker
            logger.info(f"Loading reranker: {self.model_name}")
            self._reranker = FlagReranker(
                self.model_name, use_fp16=(self.device == "cuda")
            )

    def rerank(
        self,
        query: str,
        chunk_ids: list[str],
        chunk_texts: dict[str, str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Rerank chunks using cross-encoder scoring.

        Args:
            query: Query text.
            chunk_ids: List of candidate chunk IDs.
            chunk_texts: Mapping from chunk_id to text.
            top_k: Number of top results to return.

        Returns:
            Reranked list of (chunk_id, score) tuples.
        """
        self._load_model()

        pairs = []
        valid_ids = []
        for cid in chunk_ids:
            if cid in chunk_texts:
                pairs.append([query, chunk_texts[cid]])
                valid_ids.append(cid)

        if not pairs:
            return []

        scores = self._reranker.compute_score(pairs)
        if isinstance(scores, float):
            scores = [scores]

        scored = list(zip(valid_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
