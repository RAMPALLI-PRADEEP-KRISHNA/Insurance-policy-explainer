"""Unit tests for multilingual Embedder module."""

import numpy as np
import pytest

from src.ingestion.embedder import Embedder
from src.utils.exceptions import EmbeddingError


@pytest.fixture(scope="module")
def shared_embedder():
    """Shared Embedder instance across module to avoid reloading weights."""
    emb = Embedder()
    emb.load()
    return emb


def test_embedder_load_and_dimension(shared_embedder):
    """Verify model loads and returns correct 768 dimension."""
    assert shared_embedder.is_loaded is True
    assert shared_embedder.dimension == 768


def test_embed_texts_normalization(shared_embedder):
    """Verify embedded vectors are L2-normalized float32 arrays with unit norm."""
    texts = [
        "Waiting period of 30 days applies to all illnesses.",
        "Room rent is capped at 1% of Sum Insured per day.",
    ]
    vectors = shared_embedder.embed_texts(texts)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, 768)
    assert vectors.dtype == np.float32

    # Check L2 norm is 1.0 for each vector
    norm0 = np.linalg.norm(vectors[0])
    norm1 = np.linalg.norm(vectors[1])
    assert np.isclose(norm0, 1.0, atol=1e-4)
    assert np.isclose(norm1, 1.0, atol=1e-4)


def test_embed_query_format(shared_embedder):
    """Verify embed_query returns 1D vector of shape (768,) with unit norm."""
    query = "What is the co-payment percentage?"
    vec = shared_embedder.embed_query(query)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (768,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


def test_batch_encoding(shared_embedder):
    """Verify batch encoding works consistently across batch sizes."""
    texts = [f"Clause {i}: The company will cover accidental hospitalizations." for i in range(5)]

    vecs_bs2 = shared_embedder.embed_texts(texts, batch_size=2)
    vecs_bs5 = shared_embedder.embed_texts(texts, batch_size=5)

    assert vecs_bs2.shape == (5, 768)
    assert vecs_bs5.shape == (5, 768)
    assert np.allclose(vecs_bs2, vecs_bs5, atol=1e-4)


def test_empty_input_behavior(shared_embedder):
    """Verify handling of empty texts list and empty/invalid query strings."""
    # Empty list of texts
    empty_res = shared_embedder.embed_texts([])
    assert isinstance(empty_res, np.ndarray)
    assert empty_res.shape == (0, 768)

    # Empty query string
    with pytest.raises(EmbeddingError) as exc1:
        shared_embedder.embed_query("")
    assert "non-empty string" in str(exc1.value)

    # Whitespace-only query
    with pytest.raises(EmbeddingError) as exc2:
        shared_embedder.embed_query("    \n\t   ")
    assert "non-empty string" in str(exc2.value)


def test_deterministic_encoding(shared_embedder):
    """Verify identical texts produce bitwise or numerical identical vectors."""
    text = "Co-payment is 10% for senior citizens."
    v1 = shared_embedder.embed_query(text)
    v2 = shared_embedder.embed_query(text)

    assert np.allclose(v1, v2, atol=1e-6)


def test_multilingual_compatibility_smoke_test(shared_embedder):
    """
    Multilingual compatibility test:
    Verify that an English policy clause has higher cosine similarity
    with an equivalent Tamil query than with an unrelated English statement.
    """
    # English clause
    en_clause = "Diabetes has a waiting period of 36 months of continuous coverage."
    # Tamil equivalent: நீரிழிவு நோய்க்கான காத்திருப்பு காலம் 36 மாதங்கள் ஆகும்.
    ta_query = "நீரிழிவு நோய்க்கான காத்திருப்பு காலம் 36 மாதங்கள்"
    # Unrelated English text
    unrelated = "The automobile was repaired at an authorized garage yesterday afternoon."

    v_en = shared_embedder.embed_query(en_clause)
    v_ta = shared_embedder.embed_query(ta_query)
    v_unrelated = shared_embedder.embed_query(unrelated)

    # Inner product on normalized vectors is cosine similarity
    sim_ta = float(np.dot(v_en, v_ta))
    sim_unrelated = float(np.dot(v_en, v_unrelated))

    assert sim_ta > sim_unrelated, f"Tamil similarity ({sim_ta:.3f}) should exceed unrelated ({sim_unrelated:.3f})"
    assert sim_ta > 0.40, f"Cross-lingual similarity ({sim_ta:.3f}) should be positive and substantial"
