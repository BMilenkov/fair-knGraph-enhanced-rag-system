"""Parse LLM responses into structured answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ParsedAnswer:
    """Structured answer parsed from LLM output.

    Attributes:
        answer: The extracted answer text.
        raw_response: Full LLM response.
        confidence: Estimated confidence (if parseable).
    """

    answer: str
    raw_response: str = ""
    confidence: float = 1.0


def parse_answer(response: str) -> ParsedAnswer:
    """Parse an answer from LLM response text.

    Handles various output formats:
    - Direct answer text
    - "Answer: X" format
    - "The answer is X" format

    Args:
        response: Raw LLM output.

    Returns:
        ParsedAnswer with extracted answer.
    """
    response = response.strip()

    if not response:
        return ParsedAnswer(answer="", raw_response=response)

    # Try "Answer: X" pattern
    match = re.search(r"(?:answer|ans)[\s:]+(.+?)(?:\n|$)", response, re.IGNORECASE)
    if match:
        answer = match.group(1).strip().rstrip(".")
        return ParsedAnswer(answer=answer, raw_response=response)

    # Try "The answer is X" pattern
    match = re.search(r"the answer is\s+(.+?)(?:\.|$)", response, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        return ParsedAnswer(answer=answer, raw_response=response)

    # Take the first line/sentence as the answer
    first_line = response.split("\n")[0].strip()
    # Remove trailing punctuation
    answer = first_line.rstrip(".")
    return ParsedAnswer(answer=answer, raw_response=response)
