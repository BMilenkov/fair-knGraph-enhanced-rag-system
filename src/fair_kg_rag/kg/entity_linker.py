"""Entity normalization and linking via character n-gram overlap.

Following KG²RAG, entities with character 3-gram Jaccard overlap >= 0.90
are considered the same entity and merged into a canonical form.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from fair_kg_rag.kg.triplet_extractor import Triplet
from fair_kg_rag.utils.text_utils import ngram_overlap

logger = logging.getLogger(__name__)


class EntityLinker:
    """Link and normalize entity mentions across chunks.

    Groups entity mentions that exceed the n-gram overlap threshold and
    merges them into canonical forms (most frequent surface form).

    Args:
        overlap_threshold: Minimum n-gram Jaccard similarity for merging.
        ngram_size: Character n-gram size for overlap computation.
    """

    def __init__(
        self,
        overlap_threshold: float = 0.90,
        ngram_size: int = 3,
    ) -> None:
        self.overlap_threshold = overlap_threshold
        self.ngram_size = ngram_size
        self._canonical: dict[str, str] = {}
        self._mention_counts: Counter = Counter()
        self._clusters: dict[str, list[str]] = defaultdict(list)

    def build_from_triplets(self, triplets: list[Triplet]) -> None:
        """Build entity clusters from a list of triplets.

        Collects all entity mentions, clusters them by n-gram overlap,
        and selects canonical forms.

        Args:
            triplets: List of triplets containing entity mentions.
        """
        # Collect all unique mentions
        mentions: set[str] = set()
        for t in triplets:
            mentions.add(t.subject)
            mentions.add(t.obj)
            self._mention_counts[t.subject] += 1
            self._mention_counts[t.obj] += 1

        mention_list = sorted(mentions)
        logger.info(f"Clustering {len(mention_list)} unique entity mentions")

        # Union-find style clustering
        parent: dict[str, str] = {m: m for m in mention_list}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        # Compare all pairs (quadratic but entity lists are typically small)
        for i, m1 in enumerate(mention_list):
            for m2 in mention_list[i + 1 :]:
                overlap = ngram_overlap(m1, m2, self.ngram_size)
                if overlap >= self.overlap_threshold:
                    union(m1, m2)

        # Build clusters
        clusters: dict[str, list[str]] = defaultdict(list)
        for mention in mention_list:
            root = find(mention)
            clusters[root].append(mention)

        # Select canonical form (most frequent mention in each cluster)
        for root, members in clusters.items():
            canonical = max(members, key=lambda m: self._mention_counts[m])
            for member in members:
                self._canonical[member] = canonical
            self._clusters[canonical] = members

        num_merged = sum(1 for c in clusters.values() if len(c) > 1)
        logger.info(
            f"Created {len(clusters)} entity clusters "
            f"({num_merged} clusters with merged mentions)"
        )

    def normalize(self, entity: str) -> str:
        """Get the canonical form of an entity mention.

        Args:
            entity: Entity mention string.

        Returns:
            Canonical entity name.
        """
        return self._canonical.get(entity, entity)

    def normalize_triplet(self, triplet: Triplet) -> Triplet:
        """Normalize entity mentions in a triplet.

        Args:
            triplet: Input triplet.

        Returns:
            New Triplet with normalized entity names.
        """
        return Triplet(
            subject=self.normalize(triplet.subject),
            relation=triplet.relation,
            obj=self.normalize(triplet.obj),
            source_chunk_id=triplet.source_chunk_id,
            confidence=triplet.confidence,
        )

    def normalize_triplets(self, triplets: list[Triplet]) -> list[Triplet]:
        """Normalize all triplets in a list.

        Args:
            triplets: List of triplets.

        Returns:
            List of triplets with normalized entity names.
        """
        normalized = []
        for t in triplets:
            nt = self.normalize_triplet(t)
            # Skip self-loops created by normalization
            if nt.subject != nt.obj:
                normalized.append(nt)
        return normalized

    def get_cluster(self, entity: str) -> list[str]:
        """Get all mentions in the same cluster as an entity.

        Args:
            entity: Entity mention.

        Returns:
            List of all surface forms in the cluster.
        """
        canonical = self.normalize(entity)
        return self._clusters.get(canonical, [entity])

    @property
    def num_clusters(self) -> int:
        """Number of entity clusters."""
        return len(self._clusters)

    @property
    def canonical_entities(self) -> list[str]:
        """List of all canonical entity names."""
        return list(self._clusters.keys())
