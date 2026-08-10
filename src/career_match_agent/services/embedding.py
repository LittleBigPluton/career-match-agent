import asyncio
from threading import Lock
from typing import Literal, Protocol, cast

import numpy as np
from sentence_transformers import SentenceTransformer


EmbeddingMode = Literal["query", "document"]

class EmbeddingServiceError(RuntimeError):
    """Base error raised by embedding implementations."""

class EmbeddingModelUnavailableError(EmbeddingServiceError):
    """Raised when the embedding model cannot be loaded."""

class InvalidEmbeddingResponseError(EmbeddingServiceError):
    """Raised when the model returns invalid vectors."""

class EmbeddingProvider(Protocol):
    """Interface implemented by embedding providers."""
    provider_name: str
    model_name: str
    dimension: int | None

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed candidate or search-query texts."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed job-document texts."""

class SentenceTransformerEmbeddingProvider:
    """Local Sentence Transformers embedding implementation."""
    provider_name = "sentence_transformers"

    def __init__(self, *, model_name: str, device: str, batch_size: int) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.dimension: int | None = None
        self._model: SentenceTransformer | None = None
        self._model_lock = Lock()

    def _get_model(self) -> SentenceTransformer:
        """Load the embedding model once per application process."""
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is None:
                self._model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=False)
                self.dimension = (self._model.get_embedding_dimension())

        return self._model

    def _encode_sync(self, texts: list[str], *, mode: EmbeddingMode) -> list[list[float]]:
        """Perform blocking model inference in a worker thread."""
        if not texts:
            return []

        model = self._get_model()
        with self._model_lock:
            if mode == "query":
                raw_embeddings = model.encode_query(texts, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            else:
                raw_embeddings = model.encode_document(texts, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

        embedding_matrix = np.asarray(raw_embeddings, dtype=np.float32)
        if embedding_matrix.ndim == 1:
            embedding_matrix = embedding_matrix.reshape(1, -1)

        if embedding_matrix.ndim != 2:
            raise InvalidEmbeddingResponseError("The embedding model returned a non-matrix result.")

        if embedding_matrix.shape[0] != len(texts):
            raise InvalidEmbeddingResponseError(
                "The number of returned vectors does not match the number of submitted texts.")

        if embedding_matrix.shape[1] < 1:
            raise InvalidEmbeddingResponseError("The embedding model returned empty vectors.")

        self.dimension = int(embedding_matrix.shape[1])
        return cast(list[list[float]], embedding_matrix.tolist())

    async def _encode(self, texts: list[str], *, mode: EmbeddingMode) -> list[list[float]]:
        try:
            return await asyncio.to_thread(self._encode_sync, texts, mode=mode)

        except InvalidEmbeddingResponseError:
            raise

        except (OSError, RuntimeError, ValueError) as error:
            raise EmbeddingModelUnavailableError("The configured embedding model could not be loaded or executed.") from error

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return await self._encode(texts, mode="query")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._encode(texts, mode="document")
