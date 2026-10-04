"""OpenAI embedding client."""

from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI


class OpenAIEmbedder:
    """Thin wrapper around the OpenAI embeddings API with batching."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 100,
        client: OpenAI | None = None,
    ) -> None:
        """Configure the embedding model, vector dimensions and batch size."""
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._client = client or OpenAI(api_key=api_key, max_retries=3)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a list of texts in batches, preserving the input order."""
        if any(not text or not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text.")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self._client.embeddings.create(model=self.model, input=batch, dimensions=self.dimensions)
            # The API returns items with an explicit index; sort to be safe.
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        return self.embed_documents([text])[0]
