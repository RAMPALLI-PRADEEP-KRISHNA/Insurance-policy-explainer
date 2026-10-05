"""Multilingual embedding module for insurance policy chunks and queries."""

import time
from typing import List, Optional, Union
import numpy as np

from src.config import config
from src.models.data_models import Chunk
from src.utils.exceptions import EmbeddingError
from src.utils.logging_utils import logger


class Embedder:
    """
    Manages loading and inference for the multilingual embedding model.
    Implements lazy-loading, batch encoding, L2-normalization, and dimension validation.
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        device: str = config.EMBEDDING_DEVICE,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._dimension: int = config.EMBEDDING_DIM

    @property
    def is_loaded(self) -> bool:
        """Check if the underlying sentence-transformers model is loaded in memory."""
        return self._model is not None

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        if self._model is not None:
            if hasattr(self._model, "get_embedding_dimension"):
                return self._model.get_embedding_dimension()
            elif hasattr(self._model, "get_sentence_embedding_dimension"):
                return self._model.get_sentence_embedding_dimension()
        return self._dimension

    def load(self) -> None:
        """
        Lazily load the sentence-transformers model.
        Does nothing if already loaded.
        """
        if self._model is not None:
            return

        logger.info("Loading embedding model '%s' on device '%s'...", self.model_name, self.device)
        t0 = time.time()
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()
            elapsed = time.time() - t0
            logger.info("Embedding model loaded successfully in %.2fs (dim=%d)", elapsed, self._dimension)
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
            EmbeddingError on failure.
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

        # Embed as a single-element batch
        vectors = self.embed_texts([query.strip()])
        return vectors[0]

    def embed_chunks(
        self,
        chunks: List[Chunk],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract canonical texts from chunks, encode them, and optionally attach to Chunk.embedding.

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
