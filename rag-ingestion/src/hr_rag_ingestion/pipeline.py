"""End-to-end ingestion and query pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from hr_rag_ingestion.chunking import Chunk, chunk_document
from hr_rag_ingestion.documents import HRDocument, load_documents
from hr_rag_ingestion.vector_store import PineconeVectorStore, SearchResult

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Interface of an embedding client (implemented by OpenAIEmbedder)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""


@dataclass(frozen=True)
class IndexResult:
    """Outcome of indexing one document (chunk, upsert and stale-deletion counts)."""

    doc_id: str
    chunks: int
    upserted: int
    deleted_stale: int


@dataclass(frozen=True)
class PreparedDocument:
    """A document that has been chunked and embedded but not yet written to Pinecone."""

    document: HRDocument
    chunks: list[Chunk]
    vectors: list[list[float]]


class IngestionPipeline:
    """Chunks, embeds and indexes HR documents, and runs semantic search over them."""

    def __init__(
        self,
        embedder: Embedder,
        store: PineconeVectorStore,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
    ) -> None:
        """Wire the embedder and vector store together with the chunking parameters."""
        self.embedder = embedder
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------ ingestion
    def prepare(self, document: HRDocument) -> PreparedDocument:
        """Chunk and embed a document without writing anything to Pinecone."""
        chunks = chunk_document(document, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise ValueError(f"Document {document.doc_id!r} produced no chunks (is the body empty?).")
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        return PreparedDocument(document=document, chunks=chunks, vectors=vectors)

    def write(self, prepared: PreparedDocument) -> IndexResult:
        """Write a prepared document to Pinecone, replacing its previous chunks."""
        upserted, deleted = self.store.replace_document(prepared.document.doc_id, prepared.chunks, prepared.vectors)
        return IndexResult(prepared.document.doc_id, len(prepared.chunks), upserted, deleted)

    def index_document(self, document: HRDocument) -> IndexResult:
        """Chunk, embed and write one document to Pinecone."""
        return self.write(self.prepare(document))

    def ingest_directory(self, docs_dir: Path, *, prune: bool = False, dry_run: bool = False) -> tuple[list[IndexResult], list[str]]:
        """Index every document in ``docs_dir``.

        With ``prune=True``, documents present in the index but no longer on disk are deleted.
        With ``dry_run=True``, documents are only parsed and chunked (no OpenAI or Pinecone calls).
        Returns ``(results, pruned_doc_ids)``.
        """
        documents = load_documents(docs_dir)
        results: list[IndexResult] = []
        for document in documents:
            if dry_run:
                chunks = chunk_document(document, self.chunk_size, self.chunk_overlap)
                results.append(IndexResult(document.doc_id, len(chunks), 0, 0))
                continue
            result = self.index_document(document)
            logger.info("Indexed %s: %d chunks (%d stale removed)", result.doc_id, result.chunks, result.deleted_stale)
            results.append(result)

        pruned: list[str] = []
        if prune and not dry_run:
            local_ids = {document.doc_id for document in documents}
            for doc_id in sorted(self.store.list_doc_ids() - local_ids):
                self.store.delete_document(doc_id)
                pruned.append(doc_id)
                logger.info("Pruned %s (no longer present on disk)", doc_id)
        return results, pruned

    # ------------------------------------------------------------------ query
    def search(self, query: str, top_k: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[SearchResult]:
        """Embed the query and return the top_k most similar chunks (optionally metadata-filtered)."""
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        vector = self.embedder.embed_query(query.strip())
        return self.store.query(vector, top_k=top_k, metadata_filter=metadata_filter)
