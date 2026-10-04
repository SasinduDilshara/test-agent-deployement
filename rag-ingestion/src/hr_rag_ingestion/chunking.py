"""Structure-aware chunking: split by Markdown headings, then by size."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from hr_rag_ingestion.documents import HRDocument, split_sections

# Chunk ids look like "leave-policy#0003". The "#" separator guarantees that listing by the
# prefix "leave-policy#" never matches another document such as "leave-policy-uk".
ID_SEPARATOR = "#"

# Front-matter fields copied onto every chunk's Pinecone metadata (when present).
_DOC_METADATA_FIELDS = ("version", "effective_date", "last_updated", "owner", "applies_to")


@dataclass(frozen=True)
class Chunk:
    """One retrieval chunk: deterministic id, the text to embed and its Pinecone metadata."""

    id: str
    text: str
    metadata: dict[str, Any]


def chunk_id_prefix(doc_id: str) -> str:
    """Return the id prefix shared by all chunks of a document (e.g. 'leave-policy#')."""
    return f"{doc_id}{ID_SEPARATOR}"


def chunk_document(doc: HRDocument, chunk_size: int = 1200, chunk_overlap: int = 150) -> list[Chunk]:
    """Split a document into retrieval chunks.

    Each chunk carries a short context header (document title + section path) so that the
    embedding captures *where* the text comes from, which noticeably improves retrieval for
    short passages such as bullet lists.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    base_metadata: dict[str, Any] = {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "category": doc.category,
        "source": doc.source_path.name if doc.source_path else f"{doc.doc_id}.md",
    }
    for name in _DOC_METADATA_FIELDS:
        value = doc.metadata.get(name)
        if value is not None:
            base_metadata[name] = str(value)

    chunks: list[Chunk] = []
    for section in split_sections(doc.body):
        section_path = section.path or doc.title
        for piece in splitter.split_text(section.content):
            piece = piece.strip()
            if not piece:
                continue
            index = len(chunks)
            text = f"Document: {doc.title}\nSection: {section_path}\n\n{piece}"
            chunks.append(
                Chunk(
                    id=f"{chunk_id_prefix(doc.doc_id)}{index:04d}",
                    text=text,
                    metadata={**base_metadata, "section": section_path, "chunk_index": index, "text": text},
                )
            )
    return chunks
