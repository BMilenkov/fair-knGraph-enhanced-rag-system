"""Dense vector similarity retrieval using FAISS and sentence-transformers."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Dense retrieval using sentence embeddings and FAISS.

    Args:
        model_name: Sentence-transformer model name.
        index_path: Path to save/load the FAISS index.
        device: Device for embedding computation.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        index_path: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.index_path = Path(index_path) if index_path else None
        self.device = device
        self._model = None
        self._index = None
        self._chunk_ids: list[str] = []

    def _load_model(self) -> None:
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def build_index(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Build a FAISS index from chunk texts.

        Args:
            chunk_ids: List of chunk identifiers.
            texts: Corresponding chunk texts.
        """
        import faiss

        self._load_model()
        self._chunk_ids = chunk_ids

        logger.info(f"Encoding {len(texts)} chunks...")
        embeddings = self._model.encode(
            texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        logger.info(f"Built FAISS index: {self._index.ntotal} vectors, dim={dim}")

        if self.index_path:
            self._save_index()

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve the most similar chunks for a query.

        Args:
            query: Query text.
            top_k: Number of results to return.

        Returns:
            List of (chunk_id, similarity_score) tuples, sorted descending.
        """
        if self._index is None:
            self._try_load_index()
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        self._load_model()
        query_emb = self._model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self._index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._chunk_ids):
                results.append((self._chunk_ids[idx], float(score)))
        return results

    def _save_index(self) -> None:
        """Save FAISS index and chunk IDs to disk."""
        import faiss
        if self.index_path and self._index:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_path))
            ids_path = self.index_path.with_suffix(".ids.npy")
            np.save(ids_path, np.array(self._chunk_ids, dtype=object))
            logger.info(f"Saved index to {self.index_path}")

    def _try_load_index(self) -> None:
        """Try to load a previously saved index."""
        import faiss
        if self.index_path and self.index_path.exists():
            logger.info(f"Loading index from {self.index_path}")
            self._index = faiss.read_index(str(self.index_path))
            ids_path = self.index_path.with_suffix(".ids.npy")
            if ids_path.exists():
                self._chunk_ids = list(np.load(ids_path, allow_pickle=True))
