"""Fairness-aware BFS expansion over Neo4j knowledge graph.

NOVEL CONTRIBUTION: Extends KG²RAG's BFS with demographic-aware traversal.
During multi-hop reasoning, bias compounds across hops — this module prevents
that cascade by adjusting entity scores based on demographic distribution.

Strategies:
  post_hoc:      Standard BFS → resample to balance demographics.
  in_traversal:  Adjust scores during BFS based on running distribution (most novel).
  constraint:    Hard cap on any single group's ratio in final context.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Callable

from fair_kg_rag.kg.kg_builder import KnowledgeGraph

logger = logging.getLogger(__name__)


class FairKGExpander:
    """Fairness-aware BFS expansion over Neo4j knowledge graph.

    Args:
        kg: Neo4j-backed KnowledgeGraph.
        chunk_demographics: {chunk_id: {"gender": ..., "geo_group": ...}}
        max_hops: Max BFS depth.
        score_decay: Score multiplier per hop.
        max_expanded_chunks: Max chunks to return.
        fairness_strategy: "post_hoc", "in_traversal", or "constraint".
        balance_factor: Penalty for over-represented groups (0-1).
        max_group_ratio: Max ratio for constraint strategy.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        chunk_demographics: dict[str, dict],
        max_hops: int = 2,
        score_decay: float = 0.5,
        max_expanded_chunks: int = 20,
        fairness_strategy: str = "in_traversal",
        balance_factor: float = 0.8,
        max_group_ratio: float = 0.7,
        demographic_attrs: list[str] | None = None,
    ) -> None:
        self.kg = kg
        self.chunk_demographics = chunk_demographics
        self.max_hops = max_hops
        self.score_decay = score_decay
        self.max_expanded_chunks = max_expanded_chunks
        self.fairness_strategy = fairness_strategy
        self.balance_factor = balance_factor
        self.max_group_ratio = max_group_ratio
        self.demographic_attrs = demographic_attrs or ["gender", "geo_group"]

    def expand(self, seed_results: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Expand seed results with fairness-aware graph traversal."""
        if self.fairness_strategy == "post_hoc":
            return self._post_hoc(seed_results)
        elif self.fairness_strategy == "in_traversal":
            return self._in_traversal(seed_results)
        elif self.fairness_strategy == "constraint":
            return self._constraint(seed_results)
        raise ValueError(f"Unknown fairness strategy: {self.fairness_strategy}")

    # ------------------------------------------------------------------
    # Shared BFS core
    # ------------------------------------------------------------------

    def _bfs_expand(
        self,
        seed_results: list[tuple[str, float]],
        score_adjust: Callable[[str, float, dict[str, Counter], dict[str, list[str]]], float] | None = None,
    ) -> tuple[dict[str, float], dict[str, Counter]]:
        """Core BFS expansion shared by all strategies.

        Args:
            seed_results: (chunk_id, score) pairs from initial retrieval.
            score_adjust: Optional callback(entity, base_score, demo_counts, ent_chunk_map) → adjusted_score.

        Returns:
            (chunk_scores, demographic_counts)
        """
        entity_scores: dict[str, float] = {}
        chunk_scores: dict[str, float] = {}

        # Init from seeds
        seed_cids = [cid for cid, _ in seed_results]
        chunk_entity_map = self.kg.get_entities_for_chunks(seed_cids)

        for cid, score in seed_results:
            chunk_scores[cid] = score
            for entity in chunk_entity_map.get(cid, []):
                if entity not in entity_scores or score > entity_scores[entity]:
                    entity_scores[entity] = score

        if not entity_scores:
            return chunk_scores, {}

        # Track demographics
        demo_counts = self._count_demographics(list(chunk_scores.keys()))
        visited: set[str] = set(entity_scores.keys())
        frontier = dict(entity_scores)

        # BFS hops
        for hop in range(self.max_hops):
            decay = self.score_decay ** (hop + 1)
            next_frontier: dict[str, float] = {}

            # Bulk-fetch neighbors for entire frontier in one Neo4j call
            all_neighbors = self.kg.get_neighbors_bulk(list(frontier.keys()), hops=1)

            # Collect all new neighbor entities first
            new_neighbors: set[str] = set()
            for entity in frontier:
                for neighbor in all_neighbors.get(entity, set()):
                    if neighbor not in visited:
                        new_neighbors.add(neighbor)

            # Bulk-fetch entity→chunks for all new neighbors (for score_adjust)
            ent_chunk_map: dict[str, list[str]] = {}
            if new_neighbors and score_adjust is not None:
                ent_chunk_map = self.kg.get_chunks_for_entities(list(new_neighbors))

            for entity, parent_score in frontier.items():
                for neighbor in all_neighbors.get(entity, set()):
                    if neighbor in visited:
                        continue
                    base_score = parent_score * decay

                    # Apply fairness adjustment if provided
                    if score_adjust is not None:
                        adjusted = score_adjust(neighbor, base_score, demo_counts, ent_chunk_map)
                    else:
                        adjusted = base_score

                    if neighbor not in next_frontier or adjusted > next_frontier[neighbor]:
                        next_frontier[neighbor] = adjusted

            visited.update(next_frontier.keys())
            frontier = next_frontier

            # Map new entities to chunks (reuse ent_chunk_map if available)
            if next_frontier:
                if not ent_chunk_map:
                    ent_chunk_map = self.kg.get_chunks_for_entities(list(next_frontier.keys()))
                for entity, score in next_frontier.items():
                    for cid in ent_chunk_map.get(entity, []):
                        if cid not in chunk_scores or score > chunk_scores[cid]:
                            chunk_scores[cid] = score
                            self._update_counts(cid, demo_counts)

        return chunk_scores, demo_counts

    # ------------------------------------------------------------------
    # Strategy 1: Post-hoc rebalancing
    # ------------------------------------------------------------------

    def _post_hoc(self, seed_results: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Standard BFS, then resample to balance demographics."""
        chunk_scores, _ = self._bfs_expand(seed_results)
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

        # Group by primary demographic attribute
        primary = self.demographic_attrs[0] if self.demographic_attrs else None
        groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for cid, score in sorted_chunks:
            group = self.chunk_demographics.get(cid, {}).get(primary, "unknown")
            groups[group].append((cid, score))

        # Equal allocation per group
        n_groups = max(len(groups), 1)
        per_group = self.max_expanded_chunks // n_groups
        remainder = self.max_expanded_chunks % n_groups

        result: list[tuple[str, float]] = []
        for i, items in enumerate(groups.values()):
            n = per_group + (1 if i < remainder else 0)
            result.extend(items[:n])

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    # ------------------------------------------------------------------
    # Strategy 2: In-traversal balancing (NOVEL)
    # ------------------------------------------------------------------

    def _in_traversal(self, seed_results: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Adjust entity scores during BFS based on running demographic distribution."""

        def adjust_score(
            entity: str,
            base_score: float,
            demo_counts: dict[str, Counter],
            ent_chunk_map: dict[str, list[str]],
        ) -> float:
            """Penalize entities linked to over-represented demographic groups."""
            chunk_ids = ent_chunk_map.get(entity, [])
            if not chunk_ids:
                return base_score

            weights: list[float] = []
            for cid in chunk_ids:
                demo = self.chunk_demographics.get(cid, {})
                for attr in self.demographic_attrs:
                    group = demo.get(attr)
                    if group is None:
                        continue
                    counts = demo_counts.get(attr, Counter())
                    total = sum(counts.values())
                    if total == 0:
                        weights.append(1.0)
                        continue
                    group_ratio = counts.get(group, 0) / total
                    ideal = 1.0 / max(len(counts), 2)
                    if group_ratio > ideal:
                        weights.append(self.balance_factor)  # penalize
                    else:
                        weights.append(1.0 / self.balance_factor)  # boost

            weight = sum(weights) / len(weights) if weights else 1.0
            return base_score * weight

        chunk_scores, _ = self._bfs_expand(seed_results, score_adjust=adjust_score)
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_chunks[:self.max_expanded_chunks]

    # ------------------------------------------------------------------
    # Strategy 3: Constraint-based
    # ------------------------------------------------------------------

    def _constraint(self, seed_results: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Hard cap: no group exceeds max_group_ratio in final context."""
        chunk_scores, _ = self._bfs_expand(seed_results)
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

        selected: list[tuple[str, float]] = []
        running_counts: dict[str, Counter] = {attr: Counter() for attr in self.demographic_attrs}

        for cid, score in sorted_chunks:
            if len(selected) >= self.max_expanded_chunks:
                break
            # Check if adding this chunk would violate constraints
            violates = False
            demo = self.chunk_demographics.get(cid, {})
            for attr in self.demographic_attrs:
                group = demo.get(attr)
                if group is None:
                    continue
                new_count = running_counts[attr].get(group, 0) + 1
                new_total = len(selected) + 1
                if new_total > 0 and new_count / new_total > self.max_group_ratio:
                    violates = True
                    break

            if not violates:
                selected.append((cid, score))
                self._update_counts(cid, running_counts)

        return selected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_demographics(self, chunk_ids: list[str]) -> dict[str, Counter]:
        counts: dict[str, Counter] = {attr: Counter() for attr in self.demographic_attrs}
        for cid in chunk_ids:
            self._update_counts(cid, counts)
        return counts

    def _update_counts(self, chunk_id: str, counts: dict[str, Counter]) -> None:
        demo = self.chunk_demographics.get(chunk_id, {})
        for attr in self.demographic_attrs:
            group = demo.get(attr)
            if group is not None:
                counts[attr][group] += 1
