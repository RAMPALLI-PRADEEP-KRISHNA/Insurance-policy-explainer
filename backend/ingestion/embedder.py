"""Embedding module for insurance policy chunks and queries.

Supports:
1. FastEmbed with snowflake/snowflake-arctic-embed-xs (384-dim, high-throughput CPU inference)
2. SentenceTransformers with paraphrase-multilingual-mpnet-base-v2 (768-dim, multilingual fallback)

Implements lazy-loading, singleton reuse, batch encoding, L2-normalization,
and strict dimension/finite validation.
"""

import time
from typing import List, Optional, Union
import numpy as np

from backend.config import config
from backend.models.data_models import Chunk
from backend.utils.exceptions import EmbeddingError
from backend.utils.logging_utils import logger


class Embedder:
    """
    Manages loading and inference for document and query embedding models.
    Supports FastEmbed (primary) and SentenceTransformers (fallback).
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        backend: str = config.EMBEDDING_BACKEND,
        device: str = config.EMBEDDING_DEVICE,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.backend = backend.lower()
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._dimension: int = config.EMBEDDING_DIM

    @property
    def is_loaded(self) -> bool:
        """Check if the underlying model is loaded in memory."""
        return self._model is not None

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        return self._dimension

    def load(self) -> None:
        """
        Lazily load the embedding model runtime.
        Does nothing if already loaded.
        """
        if self._model is not None:
            print("[TIMING] embedding model acquisition: model reused")
            return

        logger.info(
            "Loading embedding model '%s' via backend '%s'...",
            self.model_name,
            self.backend,
        )
        t0 = time.time()
        try:
            if self.backend == "fastembed":
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
                # Verify dimension with a lightweight probe if needed
                test_vec = list(self._model.embed(["probe"]))[0]
                self._dimension = len(test_vec)
            else:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = self._model.get_embedding_dimension()
                elif hasattr(self._model, "get_sentence_embedding_dimension"):
                    self._dimension = self._model.get_sentence_embedding_dimension()
                else:
                    self._dimension = config.EMBEDDING_DIM

            elapsed = time.time() - t0
            logger.info(
                "Embedding model loaded successfully in %.2fs (backend=%s, dim=%d)",
                elapsed,
                self.backend,
                self._dimension,
            )
            print(f"[TIMING] embedding model acquisition: model loaded in {elapsed:.4f} s")
        except Exception as e:
            logger.error("Failed to load embedding model '%s': %s", self.model_name, e)
            raise EmbeddingError(f"Could not load embedding model '{self.model_name}': {str(e)}")

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Batch embed a list of text strings into L2-normalized float32 vectors.

        Args:
            texts: List of strings to encode
            batch_size: Optional batch size override

        Returns:
            np.ndarray of shape (len(texts), dimension) with dtype float32.

        Raises:
            EmbeddingError on invalid input or failure.
        """
        if not isinstance(texts, list):
            raise EmbeddingError(f"Expected list of strings, got {type(texts)}")

        if len(texts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Ensure model is loaded
        self.load()

        bs = batch_size or self.batch_size
        t0 = time.time()

        try:
            if self.backend == "fastembed":
                # Handle possible empty strings safely by sanitizing
                sanitized_texts = [t if t.strip() else " " for t in texts]
                raw_embeddings = list(self._model.embed(sanitized_texts, batch_size=bs))
                embeddings = np.array(raw_embeddings, dtype=np.float32)
                # FastEmbed returns unit norm vectors for arctic, but ensure strict L2 norm
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeddings = embeddings / norms
            else:
                # sentence-transformers normalize_embeddings=True applies L2-normalization
                embeddings = self._model.encode(
                    texts,
                    batch_size=bs,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

            # Ensure float32 contiguous array
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

            # Validation checks
            if not np.all(np.isfinite(embeddings)):
                raise EmbeddingError("Generated embeddings contain NaN or infinite values.")
            if embeddings.shape[1] != self.dimension:
                raise EmbeddingError(
                    f"Generated dimension {embeddings.shape[1]} does not match expected {self.dimension}."
                )

            elapsed = time.time() - t0
            logger.debug("Encoded %d texts in %.2fs (shape=%s)", len(texts), elapsed, embeddings.shape)
            return embeddings

        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            raise EmbeddingError(f"Failed to encode texts: {str(e)}")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single user search query string into a normalized 1D vector.

        Args:
            query: User query string

        Returns:
            np.ndarray of shape (dimension,) with dtype float32.

        Raises:
            EmbeddingError if query is empty or encoding fails.
        """
        if not isinstance(query, str) or not query.strip():
            raise EmbeddingError("Search query must be a non-empty string.")

        clean_query = query.strip()
        self.load()

        # Snowflake Arctic Embed models require query instruction prefix for queries
        q_text = (
            f"Represent this sentence for searching relevant passages: {clean_query}"
            if "arctic-embed" in self.model_name
            else clean_query
        )

        try:
            if self.backend == "fastembed":
                raw = list(self._model.embed([q_text]))[0]
                q_vec = np.array(raw, dtype=np.float32)
                norm = np.linalg.norm(q_vec)
                if norm > 0:
                    q_vec = q_vec / norm
            else:
                # Embed as a single-element batch
                vectors = self.embed_texts([q_text])
                q_vec = vectors[0]

            if not np.all(np.isfinite(q_vec)):
                raise EmbeddingError("Query vector contains NaN or infinite values.")
            if len(q_vec) != self.dimension:
                raise EmbeddingError(f"Query vector dimension {len(q_vec)} != expected {self.dimension}.")

            return np.ascontiguousarray(q_vec, dtype=np.float32)

        except Exception as e:
            logger.error("Query embedding failed: %s", e)
            raise EmbeddingError(f"Query embedding failed: {str(e)}")

    def embed_chunks(
        self,
        chunks: List[Chunk],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract canonical texts from chunks, encode them, and attach to Chunk.embedding.

        Returns:
            np.ndarray of shape (len(chunks), dimension).
        """
        if not chunks:
            return np.empty((0, self.dimension), dtype=np.float32)

        texts = [chunk.text for chunk in chunks]
        vectors = self.embed_texts(texts, batch_size=batch_size)

        # Attach embeddings to chunks
        for idx, chunk in enumerate(chunks):
            chunk.embedding = vectors[idx].tolist()

        return vectors


# Singleton instance
embedder = Embedder()
