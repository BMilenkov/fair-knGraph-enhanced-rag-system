"""BFS expansion over Neo4j knowledge graph from seed entities (KG²RAG core).

Given seed chunks from semantic retrieval, expands through the knowledge
graph stored in Neo4j to discover related chunks sharing entities or facts.
"""

from __future__ import annotations

import logging

from fair_kg_rag.kg.kg_builder import KnowledgeGraph

logger = logging.getLogger(__name__)


class KGExpander:
    """BFS-based graph expansion from seed entities over Neo4j KG.

    Replicates KG²RAG's KG-enhanced chunk retrieval:
    1. Extract entities from seed chunks via Neo4j Chunk→Entity lookups
    2. BFS traverse RELATED edges for n hops
    3. Score propagation with decay per hop
    4. Return expanded (chunk_id, score) pairs

    Args:
        kg: Neo4j-backed KnowledgeGraph instance.
        max_hops: Maximum BFS hops from seed entities.
        score_decay: Score decay factor per hop (0-1).
        max_expanded_chunks: Maximum chunks to return after expansion.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        max_hops: int = 2,
        score_decay: float = 0.5,
        max_expanded_chunks: int = 20,
    ) -> None:
        self.kg = kg
        self.max_hops = max_hops
        self.score_decay = score_decay
        self.max_expanded_chunks = max_expanded_chunks

    def expand(
        self,
        seed_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Expand seed retrieval results through the Neo4j knowledge graph.

        Args:
            seed_results: List of (chunk_id, score) from initial retrieval.

        Returns:
            Expanded (chunk_id, score) list including seed and
            KG-expanded chunks, sorted by score descending.
        """
        entity_scores: dict[str, float] = {}
        chunk_scores: dict[str, float] = {}

        # Bulk-fetch entities for all seed chunks
        seed_chunk_ids = [cid for cid, _ in seed_results]
        seed_score_map = dict(seed_results)
        chunk_entity_map = self.kg.get_entities_for_chunks(seed_chunk_ids)

        for chunk_id, score in seed_results:
            chunk_scores[chunk_id] = score
            for entity in chunk_entity_map.get(chunk_id, []):
                if entity not in entity_scores or score > entity_scores[entity]:
                    entity_scores[entity] = score

        if not entity_scores:
            logger.debug("No entities found in seed chunks, returning seeds only")
            return seed_results

        # BFS expansion via Neo4j
        visited: set[str] = set(entity_scores.keys())
        frontier = dict(entity_scores)

        for hop in range(self.max_hops):
            decay = self.score_decay ** (hop + 1)
            next_frontier: dict[str, float] = {}

            for entity, parent_score in frontier.items():
                neighbors = self.kg.get_neighbors(entity, hops=1)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        neighbor_score = parent_score * decay
                        if (
                            neighbor not in next_frontier
                            or neighbor_score > next_frontier[neighbor]
                        ):
                            next_frontier[neighbor] = neighbor_score

            visited.update(next_frontier.keys())
            frontier = next_frontier

            # Bulk-fetch chunks for newly discovered entities
            if next_frontier:
                entity_chunk_map = self.kg.get_chunks_for_entities(
                    list(next_frontier.keys())
                )
                for entity, score in next_frontier.items():
                    for chunk_id in entity_chunk_map.get(entity, []):
                        if chunk_id not in chunk_scores or score > chunk_scores[chunk_id]:
                            chunk_scores[chunk_id] = score

        sorted_chunks = sorted(
            chunk_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_chunks[: self.max_expanded_chunks]

    def get_expanded_entities(
        self,
        seed_results: list[tuple[str, float]],
    ) -> dict[str, float]:
        """Get the expanded entity set with scores (for graph filtering).

        Args:
            seed_results: List of (chunk_id, score) from initial retrieval.

        Returns:
            Dict mapping entity names to their expansion scores.
        """
        entity_scores: dict[str, float] = {}
        seed_chunk_ids = [cid for cid, _ in seed_results]
        chunk_entity_map = self.kg.get_entities_for_chunks(seed_chunk_ids)

        for chunk_id, score in seed_results:
            for entity in chunk_entity_map.get(chunk_id, []):
                if entity not in entity_scores or score > entity_scores[entity]:
                    entity_scores[entity] = score

        visited: set[str] = set(entity_scores.keys())
        frontier = dict(entity_scores)

        for hop in range(self.max_hops):
            decay = self.score_decay ** (hop + 1)
            next_frontier: dict[str, float] = {}
            for entity, parent_score in frontier.items():
                for neighbor in self.kg.get_neighbors(entity, hops=1):
                    if neighbor not in visited:
                        score = parent_score * decay
                        if neighbor not in next_frontier or score > next_frontier[neighbor]:
                            next_frontier[neighbor] = score
            visited.update(next_frontier.keys())
            entity_scores.update(next_frontier)
            frontier = next_frontier

        return entity_scores
