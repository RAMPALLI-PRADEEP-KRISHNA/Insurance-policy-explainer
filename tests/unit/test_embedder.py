"""Unit tests for Embedder module."""

import numpy as np
import pytest

from backend.ingestion.embedder import Embedder
from backend.utils.exceptions import EmbeddingError


@pytest.fixture(scope="module")
def shared_embedder():
    """Shared Embedder instance across module to avoid reloading weights."""
    emb = Embedder()
    emb.load()
    return emb


def test_embedder_load_and_dimension(shared_embedder):
    """Verify model loads and returns correct dimension."""
    assert shared_embedder.is_loaded is True
    assert shared_embedder.dimension in (384, 768)


def test_embed_texts_normalization(shared_embedder):
    """Verify embedded vectors are L2-normalized float32 arrays with unit norm."""
    dim = shared_embedder.dimension
    texts = [
        "Waiting period of 30 days applies to all illnesses.",
        "Room rent is capped at 1% of Sum Insured per day.",
    ]
    vectors = shared_embedder.embed_texts(texts)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, dim)
    assert vectors.dtype == np.float32

    # Check L2 norm is 1.0 for each vector
    norm0 = np.linalg.norm(vectors[0])
    norm1 = np.linalg.norm(vectors[1])
    assert np.isclose(norm0, 1.0, atol=1e-4)
    assert np.isclose(norm1, 1.0, atol=1e-4)


def test_embed_query_format(shared_embedder):
    """Verify embed_query returns 1D vector with unit norm."""
    dim = shared_embedder.dimension
    query = "What is the co-payment percentage?"
    vec = shared_embedder.embed_query(query)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (dim,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


def test_batch_encoding(shared_embedder):
    """Verify batch encoding works consistently across batch sizes."""
    dim = shared_embedder.dimension
    texts = [f"Clause {i}: The company will cover accidental hospitalizations." for i in range(5)]

    vecs_bs2 = shared_embedder.embed_texts(texts, batch_size=2)
    vecs_bs5 = shared_embedder.embed_texts(texts, batch_size=5)

    assert vecs_bs2.shape == (5, dim)
    assert vecs_bs5.shape == (5, dim)
    assert np.allclose(vecs_bs2, vecs_bs5, atol=1e-3)


def test_empty_input_behavior(shared_embedder):
    """Verify handling of empty texts list and empty/invalid query strings."""
    dim = shared_embedder.dimension
    # Empty list of texts
    empty_res = shared_embedder.embed_texts([])
    assert isinstance(empty_res, np.ndarray)
    assert empty_res.shape == (0, dim)

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


def test_semantic_discrimination_smoke_test(shared_embedder):
    """
    Semantic discrimination test:
    Verify that an English policy clause has significantly higher cosine similarity
    with an equivalent English query than with an unrelated statement.
    """
    en_clause = "Diabetes has a waiting period of 36 months of continuous coverage."
    en_query = "What is the waiting period for diabetes?"
    unrelated = "The automobile was repaired at an authorized garage yesterday afternoon."

    v_clause = shared_embedder.embed_texts([en_clause])[0]
    v_query = shared_embedder.embed_query(en_query)
    v_unrelated = shared_embedder.embed_texts([unrelated])[0]

    sim_query = float(np.dot(v_clause, v_query))
    sim_unrelated = float(np.dot(v_clause, v_unrelated))

    assert sim_query > sim_unrelated, f"Relevant similarity ({sim_query:.3f}) should exceed unrelated ({sim_unrelated:.3f})"
    assert sim_query > 0.50, f"Relevant semantic similarity ({sim_query:.3f}) should be high"
