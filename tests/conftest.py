"""Shared pytest fixtures for Fair KG-RAG tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_predictions() -> list[dict]:
    return [
        {
            "id": "q1",
            "answer": "Paris",
            "retrieved_chunk_ids": ["c1", "c2"],
            "gender": "female",
            "geo_group": "european",
        },
        {
            "id": "q2",
            "answer": "Munich",
            "retrieved_chunk_ids": ["c3", "c4"],
            "gender": "male",
            "geo_group": "european",
        },
    ]


@pytest.fixture
def sample_ground_truths() -> list[dict]:
    return [
        {
            "id": "q1",
            "answer": "Paris",
            "supporting_chunk_ids": ["c2"],
            "supporting_titles": ["Paris"],
        },
        {
            "id": "q2",
            "answer": "Berlin",
            "supporting_chunk_ids": ["c3"],
            "supporting_titles": ["Berlin"],
        },
    ]
