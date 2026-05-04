"""Answer generation: context -> prompt -> LLM -> parsed answer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from fair_kg_rag.generation.llm_backend import LLMBackend
from fair_kg_rag.generation.prompt_templates import render_qa_prompt
from fair_kg_rag.generation.response_parser import ParsedAnswer, parse_answer

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from the generation pipeline.

    Attributes:
        answer: Parsed answer text.
        raw_response: Full LLM response.
        prompt: The prompt that was sent to the LLM.
        metadata: Additional generation metadata.
    """

    answer: str = ""
    raw_response: str = ""
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Generator:
    """Answer generator combining prompt construction, LLM, and parsing.

    Args:
        llm: LLM backend instance.
        include_evidence: Whether to include KG triples in prompts.
        max_new_tokens: Maximum tokens to generate.
    """

    def __init__(
        self,
        llm: LLMBackend,
        include_evidence: bool = False,
        max_new_tokens: int = 128,
    ) -> None:
        self.llm = llm
        self.include_evidence = include_evidence
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        question: str,
        context: str = "",
        triples: list[dict] | None = None,
    ) -> GenerationResult:
        """Generate an answer for a question with optional context.

        Args:
            question: The question to answer.
            context: Retrieved context text.
            triples: Optional KG evidence triples.

        Returns:
            GenerationResult with answer and metadata.
        """
        prompt = render_qa_prompt(
            question=question,
            context=context,
            triples=triples,
            include_evidence=self.include_evidence,
        )

        raw_response = self.llm.generate(
            prompt, max_new_tokens=self.max_new_tokens
        )

        parsed = parse_answer(raw_response)

        return GenerationResult(
            answer=parsed.answer,
            raw_response=raw_response,
            prompt=prompt,
            metadata={
                "context_length": len(context.split()) if context else 0,
                "num_triples": len(triples) if triples else 0,
            },
        )

    def generate_batch(
        self,
        questions: list[str],
        contexts: list[str],
        batch_triples: list[list[dict]] | None = None,
    ) -> list[GenerationResult]:
        """Generate answers for a batch of questions.

        Args:
            questions: List of questions.
            contexts: Corresponding contexts.
            batch_triples: Optional KG triples per question.

        Returns:
            List of GenerationResult objects.
        """
        results = []
        for i, (question, context) in enumerate(zip(questions, contexts)):
            triples = batch_triples[i] if batch_triples else None
            result = self.generate(question, context, triples)
            results.append(result)
        return results
