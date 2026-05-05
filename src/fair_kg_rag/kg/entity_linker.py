"""Entity normalization via character n-gram overlap (KG²RAG §3.2).

Entities with 3-gram Jaccard overlap >= 0.90 are merged into one canonical form.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from fair_kg_rag.kg.triplet_extractor import Triplet
from fair_kg_rag.utils.text_utils import ngram_overlap

logger = logging.getLogger(__name__)


class EntityLinker:
    """Cluster and normalize entity mentions using n-gram overlap.

    Args:
        overlap_threshold: Min Jaccard similarity for merging (default 0.90).
        ngram_size: Character n-gram size (default 3).
    """

    def __init__(self, overlap_threshold: float = 0.90, ngram_size: int = 3) -> None:
        self.overlap_threshold = overlap_threshold
        self.ngram_size = ngram_size
        self._canonical: dict[str, str] = {}

    def build_from_triplets(self, triplets: list[Triplet]) -> None:
        """Cluster entities from triplets by n-gram overlap."""
        # Collect mentions and their frequency
        mention_counts: Counter[str] = Counter()
        for t in triplets:
            mention_counts[t.subject] += 1
            mention_counts[t.obj] += 1

        mentions = sorted(mention_counts.keys())
        logger.info("Clustering %d unique entity mentions", len(mentions))

        # Union-find clustering
        parent: dict[str, str] = {m: m for m in mentions}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, m1 in enumerate(mentions):
            for m2 in mentions[i + 1:]:
                if ngram_overlap(m1, m2, self.ngram_size) >= self.overlap_threshold:
                    rx, ry = find(m1), find(m2)
                    if rx != ry:
                        parent[rx] = ry

        # Build canonical mapping (most frequent mention per cluster)
        clusters: dict[str, list[str]] = defaultdict(list)
        for m in mentions:
            clusters[find(m)].append(m)

        self._canonical = {}
        for members in clusters.values():
            canonical = max(members, key=lambda m: mention_counts[m])
            for member in members:
                self._canonical[member] = canonical

        merged = sum(1 for c in clusters.values() if len(c) > 1)
        logger.info("%d clusters (%d with merged mentions)", len(clusters), merged)

    @property
    def num_clusters(self) -> int:
        return len(set(self._canonical.values()))

    def normalize(self, entity: str) -> str:
        """Get canonical form of an entity mention."""
        return self._canonical.get(entity, entity)

    def normalize_triplets(self, triplets: list[Triplet]) -> list[Triplet]:
        """Normalize all entities in a list of triplets, dropping self-loops."""
        result = []
        for t in triplets:
            s = self.normalize(t.subject)
            o = self.normalize(t.obj)
            if s != o:
                result.append(Triplet(
                    subject=s, relation=t.relation, obj=o,
                    source_chunk_id=t.source_chunk_id, confidence=t.confidence,
                ))
        return result
