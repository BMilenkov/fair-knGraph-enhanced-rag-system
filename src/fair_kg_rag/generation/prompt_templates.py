"""Jinja2-based prompt templates for QA, triplet extraction, and annotation."""

from __future__ import annotations

from jinja2 import Template

# Multi-hop QA with retrieved context
QA_WITH_CONTEXT = Template("""Answer the following question based on the provided context.
Give a concise answer in a few words.

Context:
{{ context }}

Question: {{ question }}

Answer:""")

# QA without context (zero-shot baseline)
QA_ZERO_SHOT = Template("""Answer the following question concisely in a few words.

Question: {{ question }}

Answer:""")

# Triplet extraction from text
TRIPLET_EXTRACTION = Template("""Extract all knowledge triplets (subject, relation, object) from the text below.
Each triplet should capture a factual relationship explicitly stated in the text.

Rules:
- Subject and object must be named entities or specific concepts from the text
- Relation should be a concise verb phrase
- Do NOT create triplets where subject equals object
- Only extract facts explicitly stated in the text

Format: (subject | relation | object)

Examples:
Text: "Albert Einstein was born in Ulm, Germany. He developed the theory of relativity."
(Albert Einstein | born in | Ulm)
(Ulm | located in | Germany)
(Albert Einstein | developed | theory of relativity)

Text: "{{ text }}"
Triplets:""")

# QA with KG evidence triples included
QA_WITH_KG_EVIDENCE = Template("""Answer the question using the context and knowledge graph facts below.

Context:
{{ context }}

Knowledge Graph Facts:
{% for triple in triples %}
- {{ triple.subject }} → {{ triple.relation }} → {{ triple.obj }}
{% endfor %}

Question: {{ question }}

Answer:""")


def render_qa_prompt(
    question: str,
    context: str = "",
    triples: list[dict] | None = None,
    include_evidence: bool = False,
) -> str:
    """Render a QA prompt with the appropriate template.

    Args:
        question: The question to answer.
        context: Retrieved context text.
        triples: Optional KG evidence triples.
        include_evidence: Whether to include KG triples in prompt.

    Returns:
        Rendered prompt string.
    """
    if not context:
        return QA_ZERO_SHOT.render(question=question)

    if include_evidence and triples:
        return QA_WITH_KG_EVIDENCE.render(
            question=question, context=context, triples=triples
        )

    return QA_WITH_CONTEXT.render(question=question, context=context)


def render_extraction_prompt(text: str) -> str:
    """Render a triplet extraction prompt.

    Args:
        text: Source text to extract triplets from.

    Returns:
        Rendered prompt string.
    """
    return TRIPLET_EXTRACTION.render(text=text)
