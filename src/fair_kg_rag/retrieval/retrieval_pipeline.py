"""Orchestrate the full retrieval pipeline.

Flow: query -> semantic/sparse retrieval -> KG expansion (standard or fair)
     -> MST graph filtering -> DFS context organization -> reranking -> final context

All KG operations go through the Neo4j-backed KnowledgeGraph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

import networkx as nx

from fair_kg_rag.kg.kg_builder import KnowledgeGraph
from fair_kg_rag.retrieval.context_organizer import ContextOrganizer
from fair_kg_rag.retrieval.fair_kg_expander import FairKGExpander
from fair_kg_rag.retrieval.graph_filter import GraphFilter
from fair_kg_rag.retrieval.kg_expander import KGExpander
from fair_kg_rag.retrieval.reranker import Reranker
from fair_kg_rag.retrieval.semantic_retriever import SemanticRetriever
from fair_kg_rag.retrieval.sparse_retriever import SparseRetriever
from fair_kg_rag.utils.text_utils import truncate_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory KG: loads everything from Neo4j once, serves all queries locally
# ---------------------------------------------------------------------------

class _LocalKG:
    """In-memory KG for fast retrieval — no per-query Neo4j calls.

    Loads the full entity graph + chunk↔entity mappings from Neo4j once at init.
    All BFS, neighbor, subgraph, and mapping operations run in-memory.
    """

    def __init__(self, kg: KnowledgeGraph) -> None:
        logger.info("Loading KG into memory for fast retrieval...")
        self._graph: nx.MultiDiGraph = kg.to_networkx()
        c2e, e2c = kg.get_all_chunk_entity_links()
        self._chunk_to_entities: dict[str, list[str]] = c2e
        self._entity_to_chunks: dict[str, list[str]] = e2c
        logger.info(
            "Local KG ready: %d entities, %d edges, %d chunk-entity links",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
            len(self._chunk_to_entities),
        )

    def clear_cache(self) -> None:
        pass  # no-op — everything is pre-loaded

    def get_entities_for_chunks(self, chunk_ids):
        return {cid: self._chunk_to_entities.get(cid, []) for cid in chunk_ids}

    def get_chunks_for_entities(self, entities):
        return {ent: self._entity_to_chunks.get(ent, []) for ent in entities}

    def get_entity_chunks(self, entity: str) -> list[str]:
        return self._entity_to_chunks.get(entity, [])

    def get_neighbors(self, entity: str, hops: int = 1) -> set[str]:
        return self.get_neighbors_bulk([entity], hops).get(entity, set())

    def get_neighbors_bulk(self, entities: list[str], hops: int = 1) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {e: set() for e in entities}
        if hops < 1:
            return result
        for entity in entities:
            if entity not in self._graph:
                continue
            visited = {entity}
            frontier = {entity}
            for _ in range(hops):
                next_frontier: set[str] = set()
                for node in frontier:
                    for nbr in self._graph.successors(node):
                        if nbr not in visited:
                            next_frontier.add(nbr)
                            visited.add(nbr)
                    for nbr in self._graph.predecessors(node):
                        if nbr not in visited:
                            next_frontier.add(nbr)
                            visited.add(nbr)
                frontier = next_frontier
            result[entity] = visited - {entity}
        return result

    def get_subgraph(self, entities: set[str]) -> nx.MultiDiGraph:
        names = sorted({n for n in entities if n and n in self._graph})
        if not names:
            return nx.MultiDiGraph()
        return self._graph.subgraph(names).copy()

    def get_edge_data(self, source: str, target: str) -> list[dict]:
        if not self._graph.has_node(source) or not self._graph.has_node(target):
            return []
        edges = self._graph.get_edge_data(source, target)
        if edges is None:
            return []
        return [dict(data) for _, data in edges.items()]


@dataclass
class RetrievalResult:
    """Result from the retrieval pipeline.

    Attributes:
        context: Organized context string for the LLM.
        retrieved_chunks: Final (chunk_id, score) list.
        seed_chunks: Initial retrieval before expansion.
        expanded_chunks: After KG expansion.
        filtered_chunks: After MST filtering.
        metadata: Additional pipeline metadata.
    """

    context: str = ""
    retrieved_chunks: list[tuple[str, float]] = field(default_factory=list)
    seed_chunks: list[tuple[str, float]] = field(default_factory=list)
    expanded_chunks: list[tuple[str, float]] = field(default_factory=list)
    filtered_chunks: list[tuple[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalPipeline:
    """Full retrieval pipeline orchestrating all stages.

    Args:
        cfg: Retrieval configuration.
        kg: Neo4j-backed KnowledgeGraph (required for KG expansion).
        chunk_texts: Mapping from chunk_id to text content.
        chunk_demographics: Demographics per chunk (for fair expansion).
    """

    def __init__(
        self,
        cfg: DictConfig,
        kg: KnowledgeGraph | None = None,
        chunk_texts: dict[str, str] | None = None,
        chunk_demographics: dict[str, dict] | None = None,
    ) -> None:
        self.cfg = cfg
        self._raw_kg = kg
        self.kg = _LocalKG(kg) if kg is not None else None
        self.chunk_texts = chunk_texts or {}
        self.chunk_demographics = chunk_demographics or {}

        self._semantic: SemanticRetriever | None = None
        self._sparse: SparseRetriever | None = None
        self._expander: KGExpander | FairKGExpander | None = None
        self._filter: GraphFilter | None = None
        self._organizer: ContextOrganizer | None = None
        self._reranker: Reranker | None = None

    def setup(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Initialize retrieval components, reusing cached indices when possible.

        Args:
            chunk_ids: List of chunk identifiers.
            texts: Corresponding chunk texts.
        """
        retrieval_cfg = self.cfg.get("retrieval", {})

        for cid, text in zip(chunk_ids, texts):
            self.chunk_texts[cid] = text

        # Dense retriever — reuse saved FAISS index if available
        dense_cfg = retrieval_cfg.get("dense", {})
        index_dir = self.cfg.get("paths", {}).get("index_data", "data/indices")
        faiss_path = Path(index_dir) / f"{self.cfg.get('_split', 'dev')}_faiss.index"

        self._semantic = SemanticRetriever(
            model_name=dense_cfg.get("model_name", "BAAI/bge-base-en-v1.5"),
            index_path=faiss_path,
            device=self.cfg.get("device", "cuda"),
        )
        # Try loading existing index first; only build if not cached
        if faiss_path.exists() and faiss_path.with_suffix(".ids.npy").exists():
            self._semantic._try_load_index()
            logger.info("Reusing cached FAISS index from %s", faiss_path)
        else:
            self._semantic.build_index(chunk_ids, texts)

        # Sparse retriever — built on demand (currently unused in retrieve())
        sparse_cfg = retrieval_cfg.get("sparse", {})
        self._sparse = SparseRetriever(
            k1=sparse_cfg.get("k1", 1.5), b=sparse_cfg.get("b", 0.75)
        )
        # Skip building BM25 index — not used in the current retrieval flow.
        # Uncomment the line below if hybrid retrieval is added.
        # self._sparse.build_index(chunk_ids, texts)

        # KG expansion
        kg_cfg = retrieval_cfg.get("kg_expansion", {})
        if kg_cfg.get("enabled", False) and self.kg is not None:
            if kg_cfg.get("use_fair_expansion", False):
                self._expander = FairKGExpander(
                    kg=self.kg,
                    chunk_demographics=self.chunk_demographics,
                    max_hops=kg_cfg.get("max_hops", 2),
                    score_decay=kg_cfg.get("score_decay", 0.5),
                    max_expanded_chunks=kg_cfg.get("max_expanded_chunks", 20),
                    fairness_strategy=kg_cfg.get("fairness_strategy", "in_traversal"),
                    balance_factor=kg_cfg.get("balance_factor", 0.8),
                    max_group_ratio=kg_cfg.get("max_group_ratio", 0.7),
                )
                logger.info("Fair KG expansion: %s", kg_cfg.get("fairness_strategy"))
            else:
                self._expander = KGExpander(
                    kg=self.kg,
                    max_hops=kg_cfg.get("max_hops", 2),
                    score_decay=kg_cfg.get("score_decay", 0.5),
                    max_expanded_chunks=kg_cfg.get("max_expanded_chunks", 20),
                )
                logger.info("Standard KG expansion")

        # Graph filter
        filter_cfg = retrieval_cfg.get("graph_filter", {})
        if filter_cfg.get("enabled", False) and self.kg is not None:
            self._filter = GraphFilter(
                kg=self.kg,
                max_chunks=filter_cfg.get("max_chunks_after_filter", 15),
            )

        # Context organizer
        ctx_cfg = retrieval_cfg.get("context", {})
        if self.kg is not None:
            self._organizer = ContextOrganizer(
                kg=self.kg,
                max_context_tokens=ctx_cfg.get("max_context_tokens", 2048),
                organization=ctx_cfg.get("organization", "dfs"),
            )

        # Reranker
        rerank_cfg = retrieval_cfg.get("reranker", {})
        if rerank_cfg.get("enabled", False):
            self._reranker = Reranker(
                model_name=rerank_cfg.get("model_name", "BAAI/bge-reranker-base"),
                device=self.cfg.get("device", "cuda"),
            )

    def retrieve(self, query: str) -> RetrievalResult:
        """Execute the full retrieval pipeline for a query.

        Args:
            query: The query text.

        Returns:
            RetrievalResult with context and intermediate results.
        """
        result = RetrievalResult()
        retrieval_cfg = self.cfg.get("retrieval", {})

        # Clear per-query Neo4j cache
        if self.kg is not None:
            self.kg.clear_cache()

        # Step 1: Seed retrieval
        top_k = retrieval_cfg.get("dense", {}).get("top_k", 10)
        seed = self._semantic.retrieve(query, top_k=top_k)
        result.seed_chunks = seed

        # Step 2: KG expansion
        if self._expander is not None:
            expanded = self._expander.expand(seed)
        else:
            expanded = seed
        result.expanded_chunks = expanded

        # Step 3: Graph filtering
        if self._filter is not None:
            entity_scores = None
            if isinstance(self._expander, KGExpander):
                entity_scores = self._expander.get_expanded_entities(seed)
            filtered = self._filter.filter(expanded, query, entity_scores)
        else:
            filtered = expanded
        result.filtered_chunks = filtered

        # Step 4: Reranking
        if self._reranker is not None:
            rerank_k = retrieval_cfg.get("reranker", {}).get("top_k", 5)
            ids = [cid for cid, _ in filtered]
            reranked = self._reranker.rerank(query, ids, self.chunk_texts, top_k=rerank_k)
            result.retrieved_chunks = reranked
        else:
            result.retrieved_chunks = filtered

        # Step 5: Context organization
        if self._organizer is not None:
            context = self._organizer.organize(result.retrieved_chunks, self.chunk_texts)
        else:
            parts = [
                self.chunk_texts[cid]
                for cid, _ in result.retrieved_chunks
                if cid in self.chunk_texts
            ]
            context = "\n\n".join(parts)
            max_ctx = self.cfg.get("retrieval", {}).get("context", {}).get(
                "max_context_tokens", 2048
            )
            context = truncate_text(context, max_ctx)

        result.context = context
        result.metadata = {
            "num_seed": len(result.seed_chunks),
            "num_expanded": len(result.expanded_chunks),
            "num_filtered": len(result.filtered_chunks),
            "num_final": len(result.retrieved_chunks),
            "context_length": len(context.split()),
        }
        return result
