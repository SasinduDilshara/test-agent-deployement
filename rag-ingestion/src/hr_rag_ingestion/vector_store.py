"""Pinecone vector store wrapper (Pinecone Python SDK v10)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from hr_rag_ingestion.chunking import ID_SEPARATOR, Chunk, chunk_id_prefix

logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 100
DELETE_BATCH_SIZE = 1000


@dataclass(frozen=True)
class SearchResult:
    """One search hit: chunk id, similarity score, chunk text and metadata."""

    id: str
    score: float
    text: str
    metadata: dict[str, Any]

    @property
    def doc_id(self) -> str:
        """Id of the document the chunk belongs to."""
        return str(self.metadata.get("doc_id", ""))

    @property
    def title(self) -> str:
        """Title of the document the chunk belongs to."""
        return str(self.metadata.get("title", ""))

    @property
    def section(self) -> str:
        """Section path of the chunk within its document."""
        return str(self.metadata.get("section", ""))


class PineconeVectorStore:
    """Read/write access to one namespace of a Pinecone dense index."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str,
        *,
        host: str | None = None,
        client: Pinecone | None = None,
    ) -> None:
        """Configure the Pinecone client for one index and namespace (the index is opened lazily)."""
        self.index_name = index_name
        self.namespace = namespace
        self._host = host
        self._client = client or Pinecone(api_key=api_key)
        self._index = None

    # ---------------------------------------------------------------- index management
    @property
    def index(self):
        """Lazily open the data-plane client (resolves the host on first use)."""
        if self._index is None:
            if self._host:
                self._index = self._client.index(host=self._host)
            else:
                self._index = self._client.index(name=self.index_name)
        return self._index

    def ensure_index(self, dimension: int, metric: str = "cosine", cloud: str = "aws", region: str = "us-east-1") -> bool:
        """Create the serverless index if it does not exist. Returns True when it was created.

        If it exists, verifies that its dimension matches the embedding dimension.
        """
        if self._client.has_index(self.index_name):
            existing = self.describe_stats().dimension
            if existing is not None and existing != dimension:
                raise ValueError(
                    f"Pinecone index {self.index_name!r} has dimension {existing}, but the embedding "
                    f"model produces {dimension}-dimensional vectors."
                )
            return False
        logger.info("Creating Pinecone index %s (dim=%s, metric=%s, %s/%s)", self.index_name, dimension, metric, cloud, region)
        self._client.create_index(
            name=self.index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )  # blocks until the index is ready
        return True

    def describe_stats(self):
        """Return the Pinecone index statistics (dimension, vector counts per namespace)."""
        return self.index.describe_index_stats()

    def namespace_vector_count(self) -> int:
        """Return the number of vectors in the configured namespace."""
        summary = self.describe_stats().namespaces.get(self.namespace)
        return summary.vector_count if summary else 0

    # ---------------------------------------------------------------- writes
    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        """Upsert chunks with their embeddings and metadata in batches; returns the count."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return 0
        records = [
            {"id": chunk.id, "values": list(vector), "metadata": _clean_metadata(chunk.metadata)}
            for chunk, vector in zip(chunks, vectors)
        ]
        self.index.upsert(
            vectors=records,
            namespace=self.namespace,
            batch_size=UPSERT_BATCH_SIZE,
            show_progress=False,
        )
        return len(records)

    def delete_ids(self, ids: Sequence[str]) -> int:
        """Delete vectors by id in batches; returns the number of ids requested."""
        ids = list(ids)
        for start in range(0, len(ids), DELETE_BATCH_SIZE):
            self.index.delete(ids=ids[start : start + DELETE_BATCH_SIZE], namespace=self.namespace)
        return len(ids)

    def list_ids(self, prefix: str | None = None) -> list[str]:
        """List all vector ids in the namespace, optionally filtered by id prefix."""
        ids: list[str] = []
        for page in self.index.list(prefix=prefix, namespace=self.namespace):
            ids.extend(item.id for item in page.vectors)
        return ids

    def list_doc_ids(self) -> set[str]:
        """Return the set of doc_ids that have chunks in the namespace."""
        return {vector_id.split(ID_SEPARATOR, 1)[0] for vector_id in self.list_ids()}

    def replace_document(self, doc_id: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> tuple[int, int]:
        """Upsert a document's chunks, then delete chunk ids left over from a previous version.

        Upserting first means the document is never absent from the index during an update.
        Returns ``(upserted, deleted_stale)``.
        """
        existing = set(self.list_ids(prefix=chunk_id_prefix(doc_id)))
        upserted = self.upsert_chunks(chunks, vectors)
        stale = sorted(existing - {chunk.id for chunk in chunks})
        deleted = self.delete_ids(stale) if stale else 0
        return upserted, deleted

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks of one document; returns the number deleted."""
        return self.delete_ids(self.list_ids(prefix=chunk_id_prefix(doc_id)))

    # ---------------------------------------------------------------- reads
    def query(self, vector: Sequence[float], top_k: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[SearchResult]:
        """Run a similarity search and return the matches with their metadata."""
        response = self.index.query(
            vector=list(vector),
            top_k=top_k,
            namespace=self.namespace,
            filter=metadata_filter or None,
            include_metadata=True,
        )
        results = []
        for match in response.matches:
            metadata = dict(match.metadata or {})
            results.append(
                SearchResult(id=match.id, score=float(match.score), text=str(metadata.get("text", "")), metadata=metadata)
            )
        return results


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Pinecone metadata values must be strings, numbers, booleans or lists of strings (no nulls)."""
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
        elif isinstance(value, (list, tuple, set)):
            cleaned[key] = [str(item) for item in value]
        else:
            cleaned[key] = str(value)
    return cleaned
