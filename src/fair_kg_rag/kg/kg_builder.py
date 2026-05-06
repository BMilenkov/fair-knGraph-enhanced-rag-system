"""Build and query a Neo4j-backed knowledge graph from extracted triplets."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from contextlib import suppress

import networkx as nx
from neo4j import Driver, GraphDatabase

from fair_kg_rag.kg.triplet_extractor import Triplet

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Knowledge graph stored in Neo4j with a NetworkX-compatible API.

    Stores graph data once in Neo4j, then executes Cypher queries for traversal
    and lookup operations used by retrieval modules.

    Attributes:
        uri: Neo4j URI.
        user: Neo4j username.
        database: Neo4j database name.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        clear_on_init: bool = False,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USERNAME", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        if not self._password:
            raise ValueError(
                "Missing Neo4j password. Set NEO4J_PASSWORD or pass password explicitly."
            )

        self._driver: Driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self._password),
        )
        self._ensure_schema()

        if clear_on_init:
            self.clear()

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        with suppress(Exception):
            self._driver.close()

    def clear(self) -> None:
        """Delete all KG nodes and relationships from Neo4j."""
        with self._driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def _ensure_schema(self) -> None:
        """Create constraints required for efficient KG operations."""
        with self._driver.session(database=self.database) as session:
            session.run(
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
            )

    def add_triplets(self, triplets: list[Triplet]) -> None:
        """Add a batch of triplets to the knowledge graph.

        Args:
            triplets: List of Triplet objects to add.
        """
        if not triplets:
            return

        payload = [
            {
                "subject": t.subject,
                "relation": t.relation,
                "obj": t.obj,
                "source_chunk_id": t.source_chunk_id,
                "confidence": float(t.confidence),
            }
            for t in triplets
            if t.subject and t.relation and t.obj
        ]
        if not payload:
            return

        query = """
        UNWIND $triplets AS row
        MERGE (s:Entity {name: row.subject})
        MERGE (o:Entity {name: row.obj})
        CREATE (s)-[:RELATED {
            relation: row.relation,
            source_chunk_id: row.source_chunk_id,
            confidence: row.confidence
        }]->(o)
        WITH s, o, row
        WHERE row.source_chunk_id IS NOT NULL AND row.source_chunk_id <> ""
        MERGE (c:Chunk {chunk_id: row.source_chunk_id})
        MERGE (c)-[:MENTIONS]->(s)
        MERGE (c)-[:MENTIONS]->(o)
        """

        with self._driver.session(database=self.database) as session:
            session.run(query, triplets=payload)

    def get_neighbors(self, entity: str, hops: int = 1) -> set[str]:
        """Get all entities reachable within n hops.

        Args:
            entity: Starting entity.
            hops: Maximum number of hops.

        Returns:
            Set of reachable entity names.
        """
        if hops < 1:
            return set()

        hops = max(1, int(hops))
        query = f"""
        MATCH (start:Entity {{name: $entity}})
        MATCH path = (start)-[:RELATED*1..{hops}]-(nbr:Entity)
        WHERE nbr.name <> $entity
        RETURN DISTINCT nbr.name AS name
        """
        with self._driver.session(database=self.database) as session:
            records = session.run(query, entity=entity)
            return {record["name"] for record in records}

    def get_subgraph(self, entities: set[str]) -> nx.MultiDiGraph:
        """Extract the subgraph induced by a set of entities.

        Args:
            entities: Set of entity names.

        Returns:
            Subgraph containing only the specified entities and their edges.
        """
        graph = nx.MultiDiGraph()
        names = sorted({name for name in entities if name})
        if not names:
            return graph

        with self._driver.session(database=self.database) as session:
            node_records = session.run(
                "MATCH (e:Entity) WHERE e.name IN $entities RETURN e.name AS name",
                entities=names,
            )
            present_names = [record["name"] for record in node_records]
            graph.add_nodes_from(present_names)

            edge_records = session.run(
                """
                MATCH (s:Entity)-[r:RELATED]->(o:Entity)
                WHERE s.name IN $entities AND o.name IN $entities
                RETURN s.name AS source, o.name AS target,
                       r.relation AS relation,
                       r.source_chunk_id AS source_chunk_id,
                       r.confidence AS confidence
                """,
                entities=present_names,
            )
            for record in edge_records:
                graph.add_edge(
                    record["source"],
                    record["target"],
                    relation=record["relation"],
                    source_chunk_id=record["source_chunk_id"],
                    confidence=record["confidence"],
                )

        return graph

    def get_entity_chunks(self, entity: str) -> list[str]:
        """Get all chunk IDs associated with an entity.

        Args:
            entity: Entity name.

        Returns:
            List of chunk IDs.
        """
        with self._driver.session(database=self.database) as session:
            records = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name: $entity})
                RETURN DISTINCT c.chunk_id AS chunk_id
                """,
                entity=entity,
            )
            return [record["chunk_id"] for record in records]

    def get_chunk_entities(self, chunk_id: str) -> list[str]:
        """Get all entities mentioned in a chunk.

        Args:
            chunk_id: Chunk identifier.

        Returns:
            List of entity names.
        """
        with self._driver.session(database=self.database) as session:
            records = session.run(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(e:Entity)
                RETURN DISTINCT e.name AS entity
                """,
                chunk_id=chunk_id,
            )
            return [record["entity"] for record in records]

    def get_edge_data(self, source: str, target: str) -> list[dict]:
        """Get all edge data between two entities.

        Args:
            source: Source entity.
            target: Target entity.

        Returns:
            List of edge attribute dictionaries.
        """
        with self._driver.session(database=self.database) as session:
            records = session.run(
                """
                MATCH (:Entity {name: $source})-[r:RELATED]->(:Entity {name: $target})
                RETURN properties(r) AS edge
                """,
                source=source,
                target=target,
            )
            return [dict(record["edge"]) for record in records]

    def get_all_edges(self) -> list[tuple[str, str, dict]]:
        """Return all KG edges as (source, target, attributes)."""
        with self._driver.session(database=self.database) as session:
            records = session.run(
                """
                MATCH (s:Entity)-[r:RELATED]->(o:Entity)
                RETURN s.name AS source, o.name AS target, properties(r) AS edge
                """
            )
            return [
                (record["source"], record["target"], dict(record["edge"]))
                for record in records
            ]

    def get_entities_for_chunks(self, chunk_ids: Iterable[str]) -> dict[str, list[str]]:
        """Bulk query entities mentioned by each chunk ID."""
        unique_ids = sorted({cid for cid in chunk_ids if cid})
        if not unique_ids:
            return {}

        result: dict[str, list[str]] = {cid: [] for cid in unique_ids}
        with self._driver.session(database=self.database) as session:
            records = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE c.chunk_id IN $chunk_ids
                RETURN c.chunk_id AS chunk_id, collect(DISTINCT e.name) AS entities
                """,
                chunk_ids=unique_ids,
            )
            for record in records:
                result[record["chunk_id"]] = list(record["entities"])
        return result

    def get_chunks_for_entities(self, entities: Iterable[str]) -> dict[str, list[str]]:
        """Bulk query chunk IDs mentioning each entity."""
        unique_entities = sorted({entity for entity in entities if entity})
        if not unique_entities:
            return {}

        result: dict[str, list[str]] = {entity: [] for entity in unique_entities}
        with self._driver.session(database=self.database) as session:
            records = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE e.name IN $entities
                RETURN e.name AS entity, collect(DISTINCT c.chunk_id) AS chunk_ids
                """,
                entities=unique_entities,
            )
            for record in records:
                result[record["entity"]] = list(record["chunk_ids"])
        return result

    def to_networkx(self) -> nx.MultiDiGraph:
        """Export the full Neo4j KG to a NetworkX MultiDiGraph."""
        graph = nx.MultiDiGraph()
        with self._driver.session(database=self.database) as session:
            node_records = session.run("MATCH (e:Entity) RETURN e.name AS name")
            graph.add_nodes_from(record["name"] for record in node_records)

            edge_records = session.run(
                """
                MATCH (s:Entity)-[r:RELATED]->(o:Entity)
                RETURN s.name AS source, o.name AS target, properties(r) AS edge
                """
            )
            for record in edge_records:
                graph.add_edge(
                    record["source"],
                    record["target"],
                    **dict(record["edge"]),
                )
        return graph

    @property
    def num_entities(self) -> int:
        """Number of entities (nodes) in the graph."""
        with self._driver.session(database=self.database) as session:
            record = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()
            return int(record["n"]) if record else 0

    @property
    def num_relations(self) -> int:
        """Number of relations (edges) in the graph."""
        with self._driver.session(database=self.database) as session:
            record = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) AS n").single()
            return int(record["n"]) if record else 0

    @property
    def relation_types(self) -> set[str]:
        """Set of unique relation types in the graph."""
        with self._driver.session(database=self.database) as session:
            record = session.run(
                "MATCH ()-[r:RELATED]->() RETURN collect(DISTINCT r.relation) AS rels"
            ).single()
            rels = record["rels"] if record else []
            return {rel for rel in rels if rel}

    def summary(self) -> dict:
        """Get KG statistics using Cypher queries (no full graph export)."""
        num_entities = self.num_entities
        num_relations = self.num_relations

        with self._driver.session(database=self.database) as session:
            chunk_rec = session.run("MATCH (c:Chunk) RETURN count(c) AS n").single()
            num_chunks = int(chunk_rec["n"]) if chunk_rec else 0

        density = 0.0
        if num_entities > 1:
            density = num_relations / (num_entities * (num_entities - 1))

        return {
            "num_entities": num_entities,
            "num_relations": num_relations,
            "num_relation_types": len(self.relation_types),
            "num_chunks_mapped": num_chunks,
            "density": round(density, 6),
        }
