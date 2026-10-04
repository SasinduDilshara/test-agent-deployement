"""Wires settings into the concrete services used by the tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.access import AccessProfile, WriteProbe, resolve_access
from app.company import CompanyDirectory
from app.config import Settings
from app.doc_repository import DocumentRepository
from app.updater import DocumentUpdateService
from hr_rag_ingestion import IngestionPipeline, OpenAIEmbedder, PineconeVectorStore

logger = logging.getLogger(__name__)


@dataclass
class Services:
    """Container for the services the tools use (retriever, documents, company data, updater)."""

    settings: Settings
    access: AccessProfile
    retriever: IngestionPipeline  # read path (PINECONE_API_KEY)
    repository: DocumentRepository
    company: CompanyDirectory
    updater: DocumentUpdateService


def build_services(settings: Settings, probe: WriteProbe | None = None) -> Services:
    """Create all services from the settings, including resolving write access."""
    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_embedding_model,
        dimensions=settings.embedding_dimensions,
    )

    def make_store(api_key: str) -> PineconeVectorStore:
        """Create a Pinecone store for the configured index and namespace using the given key."""
        return PineconeVectorStore(
            api_key=api_key,
            index_name=settings.pinecone_index_name,
            namespace=settings.pinecone_namespace,
            host=settings.pinecone_index_host,
        )

    retriever = IngestionPipeline(
        embedder, make_store(settings.pinecone_api_key.get_secret_value()), settings.chunk_size, settings.chunk_overlap
    )

    access = resolve_access(settings, probe)
    logger.info("Document write access: %s (%s)", "ENABLED" if access.write_enabled else "DISABLED", access.reason)

    write_pipeline = None
    if access.write_enabled and access.write_api_key:
        write_pipeline = IngestionPipeline(embedder, make_store(access.write_api_key), settings.chunk_size, settings.chunk_overlap)

    repository = DocumentRepository(settings.hr_docs_dir)
    return Services(
        settings=settings,
        access=access,
        retriever=retriever,
        repository=repository,
        company=CompanyDirectory(settings.company_profile_path),
        updater=DocumentUpdateService(access, repository, write_pipeline, settings.audit_log_path),
    )
