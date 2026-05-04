"""Load and parse 2WikiMultiHopQA dataset into structured dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from fair_kg_rag.utils.io_utils import read_json


@dataclass
class EvidenceTriple:
    """A single evidence triple from Wikidata."""

    subject: str
    relation: str
    obj: str


@dataclass
class SupportingFact:
    """A supporting fact reference (title + sentence index)."""

    title: str
    sent_id: int


@dataclass
class Paragraph:
    """A context paragraph with title and sentences."""

    title: str
    sentences: list[str]

    @property
    def full_text(self) -> str:
        """Return the full paragraph text."""
        return " ".join(self.sentences)


@dataclass
class QARecord:
    """A single 2WikiMultiHopQA question-answer record.

    Attributes:
        id: Unique question identifier.
        question: The multi-hop question text.
        answer: Ground-truth answer string.
        question_type: One of: comparison, inference, compositional, bridge_comparison.
        entity_ids: Wikidata entity IDs (e.g., "Q7320430_Q51759").
        supporting_facts: List of (title, sent_id) evidence references.
        evidences: List of (subject, relation, object) knowledge triples.
        context: List of context paragraphs (including distractors).
    """

    id: str
    question: str
    answer: str
    question_type: str = ""
    entity_ids: str = ""
    supporting_facts: list[SupportingFact] = field(default_factory=list)
    evidences: list[EvidenceTriple] = field(default_factory=list)
    context: list[Paragraph] = field(default_factory=list)

    @property
    def wikidata_ids(self) -> list[str]:
        """Extract individual Wikidata QIDs from entity_ids string."""
        if not self.entity_ids:
            return []
        return [eid.strip() for eid in self.entity_ids.split("_") if eid.startswith("Q")]

    @property
    def supporting_titles(self) -> set[str]:
        """Get the set of titles that contain supporting facts."""
        return {sf.title for sf in self.supporting_facts}

    @property
    def supporting_paragraphs(self) -> list[Paragraph]:
        """Get only the paragraphs that contain supporting facts."""
        titles = self.supporting_titles
        return [p for p in self.context if p.title in titles]


def _parse_record(raw: dict) -> QARecord:
    """Parse a raw JSON dict into a QARecord.

    Args:
        raw: Raw dictionary from the dataset JSON.

    Returns:
        Parsed QARecord instance.
    """
    supporting_facts = [
        SupportingFact(title=sf[0], sent_id=sf[1])
        for sf in raw.get("supporting_facts", [])
    ]

    evidences = [
        EvidenceTriple(subject=ev[0], relation=ev[1], obj=ev[2])
        for ev in raw.get("evidences", [])
        if len(ev) >= 3
    ]

    context = [
        Paragraph(title=ctx[0], sentences=ctx[1])
        for ctx in raw.get("context", [])
    ]

    return QARecord(
        id=raw.get("_id", ""),
        question=raw.get("question", ""),
        answer=raw.get("answer", ""),
        question_type=raw.get("type", ""),
        entity_ids=raw.get("entity_ids", ""),
        supporting_facts=supporting_facts,
        evidences=evidences,
        context=context,
    )


def load_dataset(path: str | Path) -> list[QARecord]:
    """Load a 2WikiMultiHopQA split from a JSON file.

    Args:
        path: Path to the JSON file (e.g., train.json, dev.json).

    Returns:
        List of QARecord instances.
    """
    path = Path(path)
    raw_data = read_json(path)
    return [_parse_record(record) for record in raw_data]


def iter_dataset(path: str | Path, batch_size: int = 100) -> Iterator[list[QARecord]]:
    """Iterate over a dataset in batches to manage memory.

    Args:
        path: Path to the JSON file.
        batch_size: Number of records per batch.

    Yields:
        Batches of QARecord instances.
    """
    records = load_dataset(path)
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]


def get_split_path(data_dir: str | Path, split: str = "dev") -> Path:
    """Get the file path for a dataset split.

    Args:
        data_dir: Directory containing dataset files.
        split: Dataset split name (train, dev, test).

    Returns:
        Path to the split JSON file.
    """
    data_dir = Path(data_dir)
    return data_dir / f"{split}.json"
