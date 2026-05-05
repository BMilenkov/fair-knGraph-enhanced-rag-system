"""Load and parse 2WikiMultiHopQA dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fair_kg_rag.utils.io_utils import read_json


@dataclass
class EvidenceTriple:
    """A ground-truth evidence triple from Wikidata."""
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
        return " ".join(self.sentences)


@dataclass
class QARecord:
    """A single 2WikiMultiHopQA record.

    Fields map directly to the dataset JSON:
      _id, question, answer, type, entity_ids,
      supporting_facts, evidences, context
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
        """Extract Wikidata QIDs from entity_ids string (e.g., 'Q42_Q937')."""
        if not self.entity_ids:
            return []
        return [eid.strip() for eid in self.entity_ids.split("_") if eid.startswith("Q")]

    @property
    def supporting_titles(self) -> set[str]:
        """Titles that contain supporting facts."""
        return {sf.title for sf in self.supporting_facts}


def _parse_record(raw: dict) -> QARecord:
    """Parse a raw JSON dict into a QARecord."""
    return QARecord(
        id=raw.get("_id", ""),
        question=raw.get("question", ""),
        answer=raw.get("answer", ""),
        question_type=raw.get("type", ""),
        entity_ids=raw.get("entity_ids", ""),
        supporting_facts=[
            SupportingFact(title=sf[0], sent_id=sf[1])
            for sf in raw.get("supporting_facts", [])
        ],
        evidences=[
            EvidenceTriple(subject=ev[0], relation=ev[1], obj=ev[2])
            for ev in raw.get("evidences", [])
            if len(ev) >= 3
        ],
        context=[
            Paragraph(title=ctx[0], sentences=ctx[1])
            for ctx in raw.get("context", [])
        ],
    )


def load_dataset(path: str | Path) -> list[QARecord]:
    """Load a 2WikiMultiHopQA split from JSON."""
    return [_parse_record(r) for r in read_json(Path(path))]
