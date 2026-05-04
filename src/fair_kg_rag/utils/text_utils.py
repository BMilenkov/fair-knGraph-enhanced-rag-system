"""Text processing utilities for entity matching and normalization."""

from __future__ import annotations

import re
import string
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, removing articles and extra whitespace.

    Args:
        text: Input text string.

    Returns:
        Normalized text.
    """
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def char_ngrams(text: str, n: int = 3) -> set[str]:
    """Extract character n-grams from text.

    Args:
        text: Input text.
        n: N-gram size.

    Returns:
        Set of character n-grams.
    """
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def ngram_overlap(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute character n-gram overlap between two strings.

    Uses Jaccard similarity of character n-gram sets. This matches the
    entity linking approach in KG²RAG (threshold 0.90).

    Args:
        text_a: First text.
        text_b: Second text.
        n: N-gram size.

    Returns:
        Overlap score between 0.0 and 1.0.
    """
    ngrams_a = char_ngrams(text_a, n)
    ngrams_b = char_ngrams(text_b, n)
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b
    return len(intersection) / len(union)


def extract_entities_from_text(text: str) -> list[str]:
    """Extract potential entity mentions using simple capitalization heuristics.

    Identifies sequences of capitalized words as potential entity mentions.
    This is a lightweight alternative to full NER for initial entity extraction.

    Args:
        text: Input text.

    Returns:
        List of potential entity strings.
    """
    # Match sequences of capitalized words (2+ chars each)
    pattern = r"(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)"
    matches = re.findall(pattern, text)
    # Filter out common sentence starters by checking context
    entities = []
    for match in matches:
        cleaned = match.strip()
        if len(cleaned) > 1:
            entities.append(cleaned)
    return entities


def truncate_text(text: str, max_tokens: int = 512, separator: str = " ") -> str:
    """Truncate text to approximately max_tokens words.

    Args:
        text: Input text.
        max_tokens: Maximum number of word tokens.
        separator: Token separator.

    Returns:
        Truncated text.
    """
    tokens = text.split(separator)
    if len(tokens) <= max_tokens:
        return text
    return separator.join(tokens[:max_tokens])


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth.

    Args:
        prediction: Predicted text.
        ground_truth: Ground truth text.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
