"""Cross-encoder reranking for retrieved chunks.

Uses transformers AutoModelForSequenceClassification directly instead of
FlagEmbedding's FlagReranker wrapper to avoid tokenizer compatibility issues
(XLMRobertaTokenizer.prepare_for_model removed in newer transformers).
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker using BGE or similar models.

    Args:
        model_name: Cross-encoder model name.
        device: Device for inference.
        batch_size: Batch size for scoring pairs.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cuda",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the cross-encoder model and tokenizer."""
        if self._model is None:
            logger.info(f"Loading reranker: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.to(self.device)
            self._model.eval()
            if self.device == "cuda":
                self._model.half()

    @torch.no_grad()
    def _compute_scores(self, pairs: list[list[str]]) -> list[float]:
        """Score query-passage pairs in batches.

        Args:
            pairs: List of [query, passage] pairs.

        Returns:
            List of relevance scores.
        """
        all_scores: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            inputs = self._tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            logits = self._model(**inputs).logits.squeeze(-1)
            all_scores.extend(logits.float().cpu().tolist())
        return all_scores

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

        scores = self._compute_scores(pairs)

        scored = list(zip(valid_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
