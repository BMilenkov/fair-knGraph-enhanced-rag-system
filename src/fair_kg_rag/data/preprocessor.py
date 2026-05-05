"""Chunk 2WikiMultiHopQA context paragraphs for retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field

from fair_kg_rag.data.dataset_loader import Paragraph, QARecord


@dataclass
class Chunk:
    """A text chunk from a context paragraph.

    Attributes:
        chunk_id: Unique ID, e.g. "doc_Albert_Einstein_0".
        source_title: Title of the source paragraph.
        text: The chunk text.
        question_ids: Questions this chunk is associated with.
        is_supporting: Whether it contains supporting evidence.
    """
    chunk_id: str
    source_title: str
    text: str
    question_ids: list[str] = field(default_factory=list)
    is_supporting: bool = False


def _safe_id(text: str) -> str:
    """Sanitize text for use in chunk IDs."""
    return text.replace(" ", "_").replace("/", "_")[:80]


def chunk_paragraph(paragraph: Paragraph, max_tokens: int = 512) -> list[Chunk]:
    """Split a paragraph into chunks at sentence boundaries.

    Most 2WikiMultiHopQA paragraphs are short and become a single chunk.
    """
    if not paragraph.sentences:
        return []

    full_text = " ".join(paragraph.sentences)

    # Most paragraphs fit in one chunk
    if len(full_text.split()) <= max_tokens:
        return [Chunk(
            chunk_id=f"doc_{_safe_id(paragraph.title)}_0",
            source_title=paragraph.title,
            text=full_text,
        )]

    # Split by sentences for long paragraphs
    chunks = []
    chunk_idx = 0
    start = 0
    while start < len(paragraph.sentences):
        end, word_count = start, 0
        while end < len(paragraph.sentences):
            sent_words = len(paragraph.sentences[end].split())
            if word_count + sent_words > max_tokens and end > start:
                break
            word_count += sent_words
            end += 1

        chunks.append(Chunk(
            chunk_id=f"doc_{_safe_id(paragraph.title)}_{chunk_idx}",
            source_title=paragraph.title,
            text=" ".join(paragraph.sentences[start:end]),
        ))
        chunk_idx += 1
        start = end

    return chunks


def preprocess_dataset(
    records: list[QARecord],
    max_tokens: int = 512,
) -> tuple[list[Chunk], dict[str, list[str]]]:
    """Chunk all records, deduplicate by chunk_id, track question→chunk mapping.

    Returns:
        (all_chunks, question_to_chunk_ids)
    """
    seen_chunks: dict[str, Chunk] = {}  # chunk_id → Chunk
    question_to_chunks: dict[str, list[str]] = {}

    for record in records:
        supporting_titles = record.supporting_titles

        for paragraph in record.context:
            chunks = chunk_paragraph(paragraph, max_tokens)

            for chunk in chunks:
                # Deduplicate: first time seeing this chunk_id, store it
                if chunk.chunk_id not in seen_chunks:
                    seen_chunks[chunk.chunk_id] = chunk

                stored = seen_chunks[chunk.chunk_id]

                # Track question association
                if record.id not in stored.question_ids:
                    stored.question_ids.append(record.id)

                # Mark supporting
                if paragraph.title in supporting_titles:
                    stored.is_supporting = True

                # Map question → chunk
                question_to_chunks.setdefault(record.id, [])
                if chunk.chunk_id not in question_to_chunks[record.id]:
                    question_to_chunks[record.id].append(chunk.chunk_id)

    all_chunks = list(seen_chunks.values())
    return all_chunks, question_to_chunks


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    """Serialize chunks to JSON-compatible dicts."""
    return [
        {
            "chunk_id": c.chunk_id,
            "source_title": c.source_title,
            "text": c.text,
            "question_ids": c.question_ids,
            "is_supporting": c.is_supporting,
        }
        for c in chunks
    ]
