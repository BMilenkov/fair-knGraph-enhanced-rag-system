"""Jinja2-based prompt templates for QA generation."""

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
