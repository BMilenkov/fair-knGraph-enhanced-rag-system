"""Fairness-aware BFS expansion over Neo4j knowledge graph.

NOVEL CONTRIBUTION: Extends KG²RAG's BFS expansion with demographic-aware
graph traversal. During multi-hop reasoning, bias compounds across hops.
This module prevents that cascade using three strategies.

Three fairness strategies:
1. Post-hoc rebalancing: Standard BFS, then resample to balance demographics.
2. In-traversal balancing: Adjust entity scores during BFS based on running
   demographic distribution (most novel).
3. Constraint-based: Hard cap on demographic group ratio in final context.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from fair_kg_rag.kg.kg_builder import KnowledgeGraph

logger = logging.getLogger(__name__)


class FairKGExpander:
    """Fairness-aware BFS expansion over Neo4j knowledge graph.

    Args:
        kg: Neo4j-backed KnowledgeGraph instance.
        chunk_demographics: Mapping of chunk_id to demographic attributes.
            Format: {chunk_id: {"gender": "male"|"female"|None, "geo_group": ...}}
        max_hops: Maximum BFS hops.
        score_decay: Score decay factor per hop.
        max_expanded_chunks: Maximum chunks to return.
        fairness_strategy: One of "post_hoc", "in_traversal", "constraint".
        balance_factor: Score multiplier for demographic balancing (0-1).
        max_group_ratio: Max ratio for any single group (constraint strategy).
        demographic_attrs: Which attributes to balance.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        chunk_demographics: dict[str, dict[str, str | None]],
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

    def expand(
        self,
        seed_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Expand seed results with fairness-aware graph traversal.

        Args:
            seed_results: List of (chunk_id, score) from initial retrieval.

        Returns:
            Fairness-balanced list of (chunk_id, score) tuples.
        """
        if self.fairness_strategy == "post_hoc":
            return self._expand_post_hoc(seed_results)
        elif self.fairness_strategy == "in_traversal":
            return self._expand_in_traversal(seed_results)
        elif self.fairness_strategy == "constraint":
            return self._expand_constraint(seed_results)
        else:
            raise ValueError(f"Unknown fairness strategy: {self.fairness_strategy}")

    # ------------------------------------------------------------------
    # Strategy 1: Post-hoc rebalancing
    # ------------------------------------------------------------------

    def _expand_post_hoc(
        self, seed_results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        expanded = self._standard_bfs(seed_results)
        groups = self._group_by_demographics(expanded)
        return self._resample_balanced(groups, self.max_expanded_chunks)

    # ------------------------------------------------------------------
    # Strategy 2: In-traversal balancing (NOVEL)
    # ------------------------------------------------------------------

    def _expand_in_traversal(
        self, seed_results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        entity_scores: dict[str, float] = {}
        chunk_scores: dict[str, float] = {}

        seed_cids = [cid for cid, _ in seed_results]
        chunk_entity_map = self.kg.get_entities_for_chunks(seed_cids)

        for chunk_id, score in seed_results:
            chunk_scores[chunk_id] = score
            for entity in chunk_entity_map.get(chunk_id, []):
                if entity not in entity_scores or score > entity_scores[entity]:
                    entity_scores[entity] = score

        if not entity_scores:
            return seed_results

        demographic_counts = self._count_demographics(list(chunk_scores.keys()))
        visited: set[str] = set(entity_scores.keys())
        frontier = dict(entity_scores)

        for hop in range(self.max_hops):
            decay = self.score_decay ** (hop + 1)
            next_frontier: dict[str, float] = {}

            for entity, parent_score in frontier.items():
                neighbors = self.kg.get_neighbors(entity, hops=1)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        base_score = parent_score * decay
                        demo_weight = self._compute_demographic_weight(
                            neighbor, demographic_counts
                        )
                        adjusted = base_score * demo_weight
                        if neighbor not in next_frontier or adjusted > next_frontier[neighbor]:
                            next_frontier[neighbor] = adjusted

            visited.update(next_frontier.keys())
            frontier = next_frontier

            if next_frontier:
                ent_chunk_map = self.kg.get_chunks_for_entities(
                    list(next_frontier.keys())
                )
                for entity, score in next_frontier.items():
                    for cid in ent_chunk_map.get(entity, []):
                        if cid not in chunk_scores or score > chunk_scores[cid]:
                            chunk_scores[cid] = score
                            self._update_demographic_counts(cid, demographic_counts)

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_chunks[: self.max_expanded_chunks]

    # ------------------------------------------------------------------
    # Strategy 3: Constraint-based
    # ------------------------------------------------------------------

    def _expand_constraint(
        self, seed_results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        expanded = self._standard_bfs(seed_results)

        selected: list[tuple[str, float]] = []
        demographic_counts: dict[str, Counter] = {
            attr: Counter() for attr in self.demographic_attrs
        }

        for chunk_id, score in expanded:
            if len(selected) >= self.max_expanded_chunks:
                break
            if self._would_violate_constraint(
                chunk_id, demographic_counts, len(selected) + 1
            ):
                continue
            selected.append((chunk_id, score))
            self._update_demographic_counts(chunk_id, demographic_counts)

        return selected

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _standard_bfs(
        self, seed_results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """Standard BFS expansion without fairness adjustments."""
        entity_scores: dict[str, float] = {}
        chunk_scores: dict[str, float] = {}

        seed_cids = [cid for cid, _ in seed_results]
        chunk_entity_map = self.kg.get_entities_for_chunks(seed_cids)

        for chunk_id, score in seed_results:
            chunk_scores[chunk_id] = score
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
                        s = parent_score * decay
                        if neighbor not in next_frontier or s > next_frontier[neighbor]:
                            next_frontier[neighbor] = s
            visited.update(next_frontier.keys())
            frontier = next_frontier

            if next_frontier:
                ent_chunk_map = self.kg.get_chunks_for_entities(
                    list(next_frontier.keys())
                )
                for entity, score in next_frontier.items():
                    for cid in ent_chunk_map.get(entity, []):
                        if cid not in chunk_scores or score > chunk_scores[cid]:
                            chunk_scores[cid] = score

        return sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

    def _compute_demographic_weight(
        self,
        entity: str,
        demographic_counts: dict[str, Counter],
    ) -> float:
        chunk_ids = self.kg.get_entity_chunks(entity)
        if not chunk_ids:
            return 1.0

        weights: list[float] = []
        for cid in chunk_ids:
            demo = self.chunk_demographics.get(cid, {})
            for attr in self.demographic_attrs:
                group = demo.get(attr)
                if group is None:
                    continue
                counts = demographic_counts.get(attr, Counter())
                total = sum(counts.values())
                if total == 0:
                    weights.append(1.0)
                    continue
                group_ratio = counts.get(group, 0) / total
                ideal = 1.0 / max(len(counts), 2)
                if group_ratio > ideal:
                    weights.append(self.balance_factor)
                else:
                    weights.append(1.0 / self.balance_factor)

        return sum(weights) / len(weights) if weights else 1.0

    def _count_demographics(self, chunk_ids: list[str]) -> dict[str, Counter]:
        counts: dict[str, Counter] = {attr: Counter() for attr in self.demographic_attrs}
        for cid in chunk_ids:
            demo = self.chunk_demographics.get(cid, {})
            for attr in self.demographic_attrs:
                group = demo.get(attr)
                if group is not None:
                    counts[attr][group] += 1
        return counts

    def _update_demographic_counts(
        self, chunk_id: str, counts: dict[str, Counter]
    ) -> None:
        demo = self.chunk_demographics.get(chunk_id, {})
        for attr in self.demographic_attrs:
            group = demo.get(attr)
            if group is not None:
                counts[attr][group] += 1

    def _group_by_demographics(
        self, chunks: list[tuple[str, float]]
    ) -> dict[str, list[tuple[str, float]]]:
        groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
        primary = self.demographic_attrs[0] if self.demographic_attrs else None
        for cid, score in chunks:
            demo = self.chunk_demographics.get(cid, {})
            group = demo.get(primary, "unknown") if primary else "unknown"
            groups[group].append((cid, score))
        return groups

    def _resample_balanced(
        self,
        groups: dict[str, list[tuple[str, float]]],
        target_size: int,
    ) -> list[tuple[str, float]]:
        if not groups:
            return []
        for g in groups:
            groups[g].sort(key=lambda x: x[1], reverse=True)
        n_groups = len(groups)
        per_group = target_size // n_groups
        remainder = target_size % n_groups
        result: list[tuple[str, float]] = []
        for i, (_, items) in enumerate(groups.items()):
            n = per_group + (1 if i < remainder else 0)
            result.extend(items[:n])
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def _would_violate_constraint(
        self,
        chunk_id: str,
        demographic_counts: dict[str, Counter],
        new_total: int,
    ) -> bool:
        demo = self.chunk_demographics.get(chunk_id, {})
        for attr in self.demographic_attrs:
            group = demo.get(attr)
            if group is None:
                continue
            new_count = demographic_counts.get(attr, Counter()).get(group, 0) + 1
            if new_total > 0 and new_count / new_total > self.max_group_ratio:
                return True
        return False
