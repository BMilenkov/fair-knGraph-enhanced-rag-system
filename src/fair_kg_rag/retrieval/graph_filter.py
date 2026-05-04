"""MST-based subgraph filtering (KG²RAG core).

Filters the expanded chunk set by building a maximum spanning tree over the
entity subgraph (fetched from Neo4j), retaining only the most informative
connected structure.
"""

from __future__ import annotations

import logging

import networkx as nx

from fair_kg_rag.kg.kg_builder import KnowledgeGraph
from fair_kg_rag.utils.text_utils import ngram_overlap

logger = logging.getLogger(__name__)


class GraphFilter:
    """MST-based filtering of expanded knowledge subgraph.

    Following KG²RAG's post-processing:
    1. Collect entities in candidate chunks via Neo4j
    2. Extract subgraph from Neo4j as NetworkX graph
    3. Add co-occurrence edges for entities matching the query
    4. Compute maximum spanning tree
    5. Retain chunks with entities in the MST

    Args:
        kg: Neo4j-backed KnowledgeGraph instance.
        max_chunks: Maximum chunks to retain after filtering.
        query_match_threshold: N-gram overlap threshold for query-entity matching.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        max_chunks: int = 15,
        query_match_threshold: float = 0.90,
    ) -> None:
        self.kg = kg
        self.max_chunks = max_chunks
        self.query_match_threshold = query_match_threshold

    def filter(
        self,
        expanded_chunks: list[tuple[str, float]],
        query: str,
        entity_scores: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """Filter expanded chunks using MST over the entity subgraph.

        Args:
            expanded_chunks: List of (chunk_id, score) from KG expansion.
            query: Original query text for entity matching.
            entity_scores: Optional entity scores from expansion.

        Returns:
            Filtered list of (chunk_id, score) tuples.
        """
        if not expanded_chunks:
            return []

        chunk_ids = [cid for cid, _ in expanded_chunks]
        chunk_score_map = dict(expanded_chunks)

        # Bulk-fetch entities for all candidate chunks from Neo4j
        chunk_entity_map = self.kg.get_entities_for_chunks(chunk_ids)
        entities_in_chunks: set[str] = set()
        for ents in chunk_entity_map.values():
            entities_in_chunks.update(ents)

        if len(entities_in_chunks) < 2:
            return expanded_chunks[: self.max_chunks]

        # Get subgraph from Neo4j as NetworkX
        subgraph = self.kg.get_subgraph(entities_in_chunks)
        weighted = self._add_weights(subgraph, query, entity_scores)

        if weighted.number_of_edges() == 0:
            return expanded_chunks[: self.max_chunks]

        # Compute MST entities
        mst_entities = self._compute_mst_entities(weighted)

        # Filter chunks to those with entities in the MST
        filtered = []
        for chunk_id, score in expanded_chunks:
            chunk_ents = set(chunk_entity_map.get(chunk_id, []))
            if chunk_ents & mst_entities:
                filtered.append((chunk_id, score))

        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[: self.max_chunks]

    def _add_weights(
        self,
        subgraph: nx.MultiDiGraph,
        query: str,
        entity_scores: dict[str, float] | None,
    ) -> nx.Graph:
        """Build a weighted undirected graph for MST computation.

        Args:
            subgraph: NetworkX subgraph from Neo4j.
            query: Query text.
            entity_scores: Entity expansion scores.

        Returns:
            Weighted undirected graph.
        """
        G = nx.Graph()

        for node in subgraph.nodes:
            G.add_node(node)

        for u, v, data in subgraph.edges(data=True):
            weight = 1.0
            if entity_scores:
                weight = entity_scores.get(u, 0) + entity_scores.get(v, 0)
            if G.has_edge(u, v):
                G[u][v]["weight"] = max(G[u][v]["weight"], weight)
            else:
                G.add_edge(u, v, weight=weight)

        # Add co-occurrence edges for query-matching entities
        query_lower = query.lower()
        query_words = query_lower.split()
        query_entities = [
            e for e in subgraph.nodes
            if any(
                ngram_overlap(e, w, n=3) >= self.query_match_threshold
                for w in query_words
            )
            or e.lower() in query_lower
        ]

        for i, e1 in enumerate(query_entities):
            for e2 in query_entities[i + 1 :]:
                boost = 2.0
                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] += boost
                else:
                    G.add_edge(e1, e2, weight=boost)

        return G

    def _compute_mst_entities(self, graph: nx.Graph) -> set[str]:
        """Compute maximum spanning tree and return its entities.

        Args:
            graph: Weighted undirected graph.

        Returns:
            Set of entity names in the MST.
        """
        mst_entities: set[str] = set()
        for component in nx.connected_components(graph):
            sub = graph.subgraph(component)
            if sub.number_of_edges() == 0:
                mst_entities.update(component)
                continue
            mst = nx.maximum_spanning_tree(sub, weight="weight")
            mst_entities.update(mst.nodes)
        return mst_entities
