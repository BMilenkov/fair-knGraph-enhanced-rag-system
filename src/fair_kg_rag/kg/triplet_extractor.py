"""Extract knowledge triples from 2WikiMultiHopQA evidence fields.

Uses the dataset's ground-truth Wikidata triples directly (no LLM needed).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Triplet:
    """A knowledge graph triple: (subject, relation, object)."""
    subject: str
    relation: str
    obj: str
    source_chunk_id: str = ""
    confidence: float = 1.0


def extract_triplets_from_evidence(
    evidences: list[tuple[str, str, str]],
    chunk_id: str = "",
) -> list[Triplet]:
    """Convert ground-truth evidence triples to Triplet objects.

    Args:
        evidences: List of (subject, relation, object) from dataset's 'evidences' field.
        chunk_id: Chunk ID to link these triples to.

    Returns:
        Validated Triplet objects (no self-loops, no empty fields).
    """
    triplets = []
    for subject, relation, obj in evidences:
        s, r, o = str(subject).strip(), str(relation).strip(), str(obj).strip()
        if s and r and o and s.lower() != o.lower():
            triplets.append(Triplet(
                subject=s, relation=r, obj=o,
                source_chunk_id=chunk_id, confidence=1.0,
            ))
    return triplets


def triplets_to_dicts(triplets: list[Triplet]) -> list[dict]:
    """Serialize triplets for JSON output."""
    return [
        {
            "subject": t.subject,
            "relation": t.relation,
            "object": t.obj,
            "source_chunk_id": t.source_chunk_id,
            "confidence": t.confidence,
        }
        for t in triplets
    ]
