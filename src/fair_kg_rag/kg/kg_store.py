"""Persist and connect to Neo4j knowledge graph storage."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fair_kg_rag.kg.kg_builder import KnowledgeGraph
from fair_kg_rag.kg.triplet_extractor import Triplet

logger = logging.getLogger(__name__)


def connect_kg(
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str = "neo4j",
    clear_on_init: bool = False,
) -> KnowledgeGraph:
    """Create a Neo4j-backed knowledge graph connection.

    Args:
        uri: Neo4j bolt URI (falls back to NEO4J_URI if None).
        user: Neo4j username (falls back to NEO4J_USERNAME if None).
        password: Neo4j password (falls back to NEO4J_PASSWORD if None).
        database: Neo4j database name.
        clear_on_init: If True, clear all existing KG data first.

    Returns:
        Connected KnowledgeGraph instance.
    """
    uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.getenv("NEO4J_USERNAME", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD")

    if not password:
        raise ValueError(
            "Missing Neo4j password. Set NEO4J_PASSWORD or pass password explicitly."
        )

    kg = KnowledgeGraph(
        uri=uri,
        user=user,
        password=password,
        database=database,
        clear_on_init=clear_on_init,
    )
    logger.info("Connected to Neo4j KG at %s (db=%s)", uri, database)
    return kg


def save_kg(kg: KnowledgeGraph, output_dir: str | Path | None = None, name: str = "kg") -> None:
    """Persist KG metadata to disk while graph remains in Neo4j.

    This keeps previous API usage intact. Neo4j is the source of truth, and this
    function only writes a lightweight metadata snapshot for experiment tracking.

    Args:
        kg: Connected Neo4j-backed KnowledgeGraph.
        output_dir: Optional directory for metadata snapshot JSON.
        name: Base filename for metadata snapshot.
    """
    if output_dir is None:
        logger.info(
            "KG persistence is handled by Neo4j; skipping local snapshot "
            "(entities=%s, relations=%s)",
            kg.num_entities,
            kg.num_relations,
        )
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / f"{name}_metadata.json"

    payload = {
        "backend": "neo4j",
        "uri": kg.uri,
        "user": kg.user,
        "database": kg.database,
        "summary": kg.summary(),
    }
    with open(metadata_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)

    logger.info(
        "Saved KG metadata snapshot to %s (entities=%s, relations=%s)",
        metadata_path,
        kg.num_entities,
        kg.num_relations,
    )


def load_kg(
    input_dir: str | Path | None = None,
    name: str = "kg",
    *,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str = "neo4j",
) -> KnowledgeGraph:
    """Load a Neo4j-backed KG connection.

    Args:
        input_dir: Optional path containing metadata snapshot. Not required.
        name: Snapshot base name. Kept for API compatibility.
        uri: Neo4j bolt URI (optional, can come from env).
        user: Neo4j username (optional, can come from env).
        password: Neo4j password (optional, can come from env).
        database: Neo4j database name.

    Returns:
        Connected KnowledgeGraph instance.
    """
    if input_dir is not None:
        metadata_path = Path(input_dir) / f"{name}_metadata.json"
        if metadata_path.exists():
            logger.info("Found KG metadata snapshot: %s", metadata_path)

    kg = connect_kg(uri=uri, user=user, password=password, database=database)
    logger.info("Loaded Neo4j KG: %s entities, %s relations", kg.num_entities, kg.num_relations)
    return kg


def import_triplets_json(
    kg: KnowledgeGraph,
    triplets_path: str | Path,
    batch_size: int = 2000,
) -> None:
    """Import triplets from JSON into Neo4j KG.

    Expected JSON schema:
    - List of dict items with keys: subject, relation, object,
      source_chunk_id (optional), confidence (optional).

    Args:
        kg: Connected Neo4j-backed KnowledgeGraph.
        triplets_path: Path to triplets JSON file.
        batch_size: Number of triplets per insert batch.
    """
    path = Path(triplets_path)
    if not path.exists():
        raise FileNotFoundError(f"Triplets file not found: {path}")

    with open(path, "r", encoding="utf-8") as file_handle:
        raw_items = json.load(file_handle)

    if not isinstance(raw_items, list):
        raise ValueError("Triplets JSON must be a list of objects")

    triplets = [
        Triplet(
            subject=str(item.get("subject", "")).strip(),
            relation=str(item.get("relation", "")).strip(),
            obj=str(item.get("object", item.get("obj", ""))).strip(),
            source_chunk_id=str(item.get("source_chunk_id", "")).strip(),
            confidence=float(item.get("confidence", 1.0)),
        )
        for item in raw_items
    ]

    total = len(triplets)
    logger.info("Importing %s triplets from %s", total, path)
    for i in range(0, total, batch_size):
        batch = triplets[i : i + batch_size]
        kg.add_triplets(batch)

    logger.info(
        "Triplet import complete: %s entities, %s relations",
        kg.num_entities,
        kg.num_relations,
    )
