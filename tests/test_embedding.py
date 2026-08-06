import pytest

from career_match_agent.services.embedding import InvalidEmbeddingResponseError
from career_match_agent.services.semantic_ranker import (
    cosine_similarity,
    split_text_into_chunks
)


def test_cosine_similarity_for_identical_vectors() -> None:
    similarity = cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert similarity == pytest.approx(1.0)

def test_cosine_similarity_for_orthogonal_vectors() -> None:
    similarity = cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert similarity == pytest.approx(0.0)

def test_cosine_similarity_rejects_different_dimensions() -> None:
    with pytest.raises(InvalidEmbeddingResponseError, match="different dimensions"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_cosine_similarity_rejects_zero_vector() -> None:
    with pytest.raises(InvalidEmbeddingResponseError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])

def test_split_text_into_chunks_respects_limit() -> None:
    text = " ".join(["machine learning"] * 200)
    chunks = split_text_into_chunks(text, maximum_characters=300)
    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)
