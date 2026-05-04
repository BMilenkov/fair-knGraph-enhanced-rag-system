"""LLM-based triplet extraction from text chunks.

Extracts (subject, relation, object) triples using few-shot prompting,
following the KG²RAG approach. Also supports using the ground-truth
evidence triples from 2WikiMultiHopQA.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """Extract all knowledge triplets (subject, relation, object) from the text.
Each triplet should capture a factual relationship stated in the text.

Rules:
- Subject and object must be named entities or specific concepts from the text
- Relation should be a concise verb phrase describing the relationship
- Do NOT create triplets where subject equals object
- Only extract facts explicitly stated in the text

Format each triplet as: (subject | relation | object)

Examples:
Text: "Albert Einstein was born in Ulm, Germany. He developed the theory of relativity."
Triplets:
(Albert Einstein | born in | Ulm)
(Ulm | located in | Germany)
(Albert Einstein | developed | theory of relativity)

Text: "{text}"
Triplets:
"""


@dataclass
class Triplet:
    """A knowledge graph triplet.

    Attributes:
        subject: Head entity.
        relation: Relationship type.
        obj: Tail entity.
        source_chunk_id: ID of the chunk this was extracted from.
        confidence: Extraction confidence score (0-1).
    """

    subject: str
    relation: str
    obj: str
    source_chunk_id: str = ""
    confidence: float = 1.0


class LLMInterface(Protocol):
    """Protocol for LLM generate calls."""

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str: ...



def extract_triplets_llm(
    text: str,
    chunk_id: str,
    llm: LLMInterface,
    max_new_tokens: int = 256,
) -> list[Triplet]:
    """Extract triplets from text using an LLM.

    Args:
        text: Source text to extract triplets from.
        chunk_id: ID of the source chunk.
        llm: LLM backend for generation.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        List of extracted Triplet objects.
    """
    prompt = EXTRACTION_PROMPT.format(text=text)
    response = llm.generate(prompt, max_new_tokens=max_new_tokens)
    return parse_triplets(response, chunk_id, source_text=text)


def parse_triplets(
    response: str,
    chunk_id: str = "",
    source_text: str = "",
) -> list[Triplet]:
    """Parse LLM response into Triplet objects.

    Args:
        response: Raw LLM output containing triplets.
        chunk_id: Source chunk ID.
        source_text: Original text for grounding validation.

    Returns:
        List of validated Triplet objects.
    """
    triplets = []
    # Match patterns like (subject | relation | object)
    pattern = r"\(([^|]+)\|([^|]+)\|([^)]+)\)"
    matches = re.findall(pattern, response)

    for match in matches:
        subject = match[0].strip()
        relation = match[1].strip()
        obj = match[2].strip()

        triplet = Triplet(
            subject=subject,
            relation=relation,
            obj=obj,
            source_chunk_id=chunk_id,
        )

        if _validate_triplet(triplet, source_text):
            triplets.append(triplet)

    return triplets


def extract_triplets_from_evidence(
    evidences: list[tuple[str, str, str]],
    chunk_id: str = "",
) -> list[Triplet]:
    """Convert 2WikiMultiHopQA ground-truth evidence triples to Triplet objects.

    This provides a high-quality alternative to LLM extraction by using the
    dataset's built-in Wikidata evidence triples.

    Args:
        evidences: List of (subject, relation, object) tuples from the dataset.
        chunk_id: Associated chunk ID.

    Returns:
        List of Triplet objects.
    """
    triplets = []
    for subject, relation, obj in evidences:
        triplet = Triplet(
            subject=str(subject).strip(),
            relation=str(relation).strip(),
            obj=str(obj).strip(),
            source_chunk_id=chunk_id,
            confidence=1.0,  # Ground-truth evidence
        )
        if triplet.subject and triplet.relation and triplet.obj:
            if triplet.subject.lower() != triplet.obj.lower():
                triplets.append(triplet)
    return triplets


def _validate_triplet(
    triplet: Triplet,
    source_text: str = "",
    min_relation_length: int = 2,
    require_grounding: bool = False,
) -> bool:
    """Validate an extracted triplet.

    Args:
        triplet: The triplet to validate.
        source_text: Original text for grounding check.
        min_relation_length: Minimum relation string length.
        require_grounding: Whether to require text grounding.

    Returns:
        True if triplet passes validation.
    """
    # Reject empty fields
    if not triplet.subject or not triplet.relation or not triplet.obj:
        return False

    # Reject self-loops
    if triplet.subject.lower() == triplet.obj.lower():
        return False

    # Reject short relations
    if len(triplet.relation) < min_relation_length:
        return False

    # Optional text grounding check
    if require_grounding and source_text:
        source_lower = source_text.lower()
        if triplet.obj.lower() not in source_lower:
            return False

    return True


def triplets_to_dicts(triplets: list[Triplet]) -> list[dict]:
    """Convert triplets to serializable dictionaries.

    Args:
        triplets: List of Triplet objects.

    Returns:
        List of dictionaries.
    """
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
