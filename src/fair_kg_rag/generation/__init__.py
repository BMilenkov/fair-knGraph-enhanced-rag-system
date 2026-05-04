"""LLM generation: backend, prompts, parsing, and answer generation."""

from fair_kg_rag.generation.generator import Generator, GenerationResult
from fair_kg_rag.generation.llm_backend import LLMBackend

__all__ = ["Generator", "GenerationResult", "LLMBackend"]
