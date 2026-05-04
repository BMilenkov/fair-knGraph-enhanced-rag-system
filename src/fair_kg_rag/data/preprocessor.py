"""Preprocessing and chunking for 2WikiMultiHopQA paragraphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from fair_kg_rag.data.dataset_loader import Paragraph, QARecord


@dataclass
class Chunk:
    """A text chunk derived from a paragraph.

    Attributes:
        chunk_id: Unique chunk identifier (e.g., "doc_Title_0").
        source_title: Title of the source paragraph/document.
        text: The chunk text content.
        question_ids: IDs of questions this chunk is associated with.
        is_supporting: Whether this chunk contains supporting evidence.
        sentence_indices: Original sentence indices within the paragraph.
    """

    chunk_id: str
    source_title: str
    text: str
    question_ids: list[str] = field(default_factory=list)
    is_supporting: bool = False
    sentence_indices: list[int] = field(default_factory=list)


def chunk_paragraph(
    paragraph: Paragraph,
    max_tokens: int = 512,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Split a paragraph into chunks respecting sentence boundaries.

    2WikiMultiHopQA paragraphs are typically short (Wikipedia paragraphs),
    so most will remain as single chunks. Only split if exceeding max_tokens.

    Args:
        paragraph: Source paragraph to chunk.
        max_tokens: Maximum approximate word count per chunk.
        overlap_sentences: Number of overlapping sentences between chunks.

    Returns:
        List of Chunk objects.
    """
    sentences = paragraph.sentences
    if not sentences:
        return []

    full_text = " ".join(sentences)
    word_count = len(full_text.split())

    # If short enough, keep as a single chunk
    if word_count <= max_tokens:
        return [
            Chunk(
                chunk_id=f"doc_{_safe_id(paragraph.title)}_0",
                source_title=paragraph.title,
                text=full_text,
                sentence_indices=list(range(len(sentences))),
            )
        ]

    # Otherwise, split by sentences into windows
    chunks = []
    chunk_idx = 0
    start = 0

    while start < len(sentences):
        end = start
        current_words = 0

        # Accumulate sentences until hitting the token limit
        while end < len(sentences):
            sent_words = len(sentences[end].split())
            if current_words + sent_words > max_tokens and end > start:
                break
            current_words += sent_words
            end += 1

        chunk_text = " ".join(sentences[start:end])
        chunks.append(
            Chunk(
                chunk_id=f"doc_{_safe_id(paragraph.title)}_{chunk_idx}",
                source_title=paragraph.title,
                text=chunk_text,
                sentence_indices=list(range(start, end)),
            )
        )
        chunk_idx += 1
        start = max(start + 1, end - overlap_sentences)

    return chunks


def preprocess_record(
    record: QARecord,
    max_tokens: int = 512,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Preprocess a single QA record: chunk all context paragraphs.

    Args:
        record: A QARecord instance.
        max_tokens: Maximum word count per chunk.
        overlap_sentences: Sentence overlap between consecutive chunks.

    Returns:
        List of Chunk objects with supporting labels set.
    """
    supporting_titles = record.supporting_titles
    supporting_sents = {
        (sf.title, sf.sent_id) for sf in record.supporting_facts
    }

    all_chunks = []
    for paragraph in record.context:
        chunks = chunk_paragraph(paragraph, max_tokens, overlap_sentences)

        for chunk in chunks:
            chunk.question_ids.append(record.id)

            # Mark chunk as supporting if it contains any supporting sentence
            if paragraph.title in supporting_titles:
                for sent_idx in chunk.sentence_indices:
                    if (paragraph.title, sent_idx) in supporting_sents:
                        chunk.is_supporting = True
                        break

        all_chunks.extend(chunks)

    return all_chunks


def preprocess_dataset(
    records: list[QARecord],
    max_tokens: int = 512,
) -> tuple[list[Chunk], dict[str, list[str]]]:
    """Preprocess an entire dataset split.

    Args:
        records: List of QARecord instances.
        max_tokens: Maximum word count per chunk.

    Returns:
        Tuple of (all_chunks, question_to_chunk_ids mapping).
    """
    all_chunks = []
    question_to_chunks: dict[str, list[str]] = {}

    # Track seen titles to avoid duplicate chunks across questions
    title_to_chunk_ids: dict[str, list[str]] = {}

    for record in records:
        chunks = preprocess_record(record, max_tokens)

        for chunk in chunks:
            # Deduplicate by source title
            if chunk.source_title not in title_to_chunk_ids:
                title_to_chunk_ids[chunk.source_title] = []
                all_chunks.append(chunk)

            chunk_ids = title_to_chunk_ids[chunk.source_title]
            if chunk.chunk_id not in chunk_ids:
                chunk_ids.append(chunk.chunk_id)
                if chunk not in all_chunks:
                    all_chunks.append(chunk)

            # Map question to its relevant chunks
            if record.id not in question_to_chunks:
                question_to_chunks[record.id] = []
            question_to_chunks[record.id].append(chunk.chunk_id)

    return all_chunks, question_to_chunks


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    """Convert chunks to serializable dictionaries.

    Args:
        chunks: List of Chunk objects.

    Returns:
        List of dictionaries.
    """
    return [
        {
            "chunk_id": c.chunk_id,
            "source_title": c.source_title,
            "text": c.text,
            "question_ids": c.question_ids,
            "is_supporting": c.is_supporting,
            "sentence_indices": c.sentence_indices,
        }
        for c in chunks
    ]


def _safe_id(text: str) -> str:
    """Create a filesystem-safe identifier from text.

    Args:
        text: Input text.

    Returns:
        Sanitized string safe for use in IDs.
    """
    return text.replace(" ", "_").replace("/", "_")[:80]
