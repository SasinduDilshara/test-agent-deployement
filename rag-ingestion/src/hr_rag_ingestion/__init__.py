"""HR document ingestion pipeline: Markdown -> chunks -> OpenAI embeddings -> Pinecone."""

from hr_rag_ingestion.chunking import Chunk, chunk_document, chunk_id_prefix
from hr_rag_ingestion.documents import (
    DocumentError,
    HRDocument,
    load_document,
    load_documents,
    parse_document,
    split_sections,
    upsert_section,
)
from hr_rag_ingestion.embeddings import OpenAIEmbedder
from hr_rag_ingestion.pipeline import IndexResult, IngestionPipeline
from hr_rag_ingestion.vector_store import PineconeVectorStore, SearchResult

__all__ = [
    "Chunk",
    "DocumentError",
    "HRDocument",
    "IndexResult",
    "IngestionPipeline",
    "OpenAIEmbedder",
    "PineconeVectorStore",
    "SearchResult",
    "chunk_document",
    "chunk_id_prefix",
    "load_document",
    "load_documents",
    "parse_document",
    "split_sections",
    "upsert_section",
]
