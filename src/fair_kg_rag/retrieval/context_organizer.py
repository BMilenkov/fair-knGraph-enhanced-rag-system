"""DFS-based context organization (KG²RAG core).

Orders filtered chunks using DFS traversal over the KG subgraph
(fetched from Neo4j) to produce a coherent reading sequence.
"""

from __future__ import annotations

import logging

import networkx as nx

from fair_kg_rag.kg.kg_builder import KnowledgeGraph
from fair_kg_rag.utils.text_utils import truncate_text

logger = logging.getLogger(__name__)


class ContextOrganizer:
    """Organize filtered chunks into coherent context using DFS ordering.

    Args:
        kg: Neo4j-backed KnowledgeGraph instance.
        max_context_tokens: Maximum word tokens in the final context.
        organization: Organization strategy ("dfs" or "relevance").
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        max_context_tokens: int = 2048,
        organization: str = "dfs",
    ) -> None:
        self.kg = kg
        self.max_context_tokens = max_context_tokens
        self.organization = organization

    def organize(
        self,
        filtered_chunks: list[tuple[str, float]],
        chunk_texts: dict[str, str],
    ) -> str:
        """Organize chunks into a single context string.

        Args:
            filtered_chunks: List of (chunk_id, score) tuples.
            chunk_texts: Mapping from chunk_id to text content.

        Returns:
            Organized context string, truncated to max_context_tokens.
        """
        if not filtered_chunks:
            return ""

        if self.organization == "dfs":
            ordered_ids = self._dfs_order(filtered_chunks)
        else:
            ordered_ids = [cid for cid, _ in filtered_chunks]

        context_parts = [
            chunk_texts[cid] for cid in ordered_ids if cid in chunk_texts
        ]
        context = "\n\n".join(context_parts)
        return truncate_text(context, self.max_context_tokens)

    def _dfs_order(self, chunks: list[tuple[str, float]]) -> list[str]:
        """Order chunks using DFS traversal over the entity subgraph.

        Args:
            chunks: List of (chunk_id, score) tuples.

        Returns:
            Ordered list of chunk IDs.
        """
        chunk_ids = [cid for cid, _ in chunks]
        chunk_score_map = dict(chunks)

        # Bulk-fetch entities from Neo4j
        chunk_entity_map = self.kg.get_entities_for_chunks(chunk_ids)

        all_entities: set[str] = set()
        for ents in chunk_entity_map.values():
            all_entities.update(ents)

        if not all_entities:
            return chunk_ids

        # Get subgraph from Neo4j and convert to undirected
        subgraph = self.kg.get_subgraph(all_entities)
        undirected = subgraph.to_undirected()

        # Build entity→chunks reverse map
        entity_to_chunks: dict[str, list[str]] = {}
        for cid, ents in chunk_entity_map.items():
            for entity in ents:
                if entity not in entity_to_chunks:
                    entity_to_chunks[entity] = []
                entity_to_chunks[entity].append(cid)

        # Find start node: entity from highest-scored chunk
        start_entity = None
        best_score = -1.0
        for entity in all_entities:
            if entity in undirected:
                for cid in entity_to_chunks.get(entity, []):
                    if chunk_score_map.get(cid, 0) > best_score:
                        best_score = chunk_score_map.get(cid, 0)
                        start_entity = entity

        if start_entity is None:
            return chunk_ids

        # DFS traversal
        try:
            dfs_order = list(nx.dfs_preorder_nodes(undirected, start_entity))
        except nx.NetworkXError:
            dfs_order = list(all_entities)

        ordered: list[str] = []
        seen: set[str] = set()
        for entity in dfs_order:
            for cid in entity_to_chunks.get(entity, []):
                if cid not in seen and cid in chunk_score_map:
                    ordered.append(cid)
                    seen.add(cid)

        # Add unreached chunks
        for cid in chunk_ids:
            if cid not in seen:
                ordered.append(cid)

        return ordered
