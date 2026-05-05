"""Text processing utilities for entity matching and evaluation."""

from __future__ import annotations

import re
import string
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize text: lowercase, remove articles/punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def ngram_overlap(text_a: str, text_b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity (used for entity linking, KG²RAG §3.2)."""
    a = text_a.lower().strip()
    b = text_b.lower().strip()
    ngrams_a = {a[i:i + n] for i in range(max(1, len(a) - n + 1))}
    ngrams_b = {b[i:i + n] for i in range(max(1, len(b) - n + 1))}
    if not ngrams_a or not ngrams_b:
        return 0.0
    return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between normalized prediction and ground truth."""
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


def truncate_text(text: str, max_tokens: int = 512) -> str:
    """Truncate text to max_tokens words."""
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])
