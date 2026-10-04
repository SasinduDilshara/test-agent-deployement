from pathlib import Path

import pytest

from hr_rag_ingestion import (
    DocumentError,
    IngestionPipeline,
    chunk_document,
    load_documents,
    parse_document,
    split_sections,
    upsert_section,
)
from hr_rag_ingestion.documents import get_section, list_section_titles
from hr_rag_ingestion.vector_store import _clean_metadata

DOCS_DIR = Path(__file__).resolve().parents[2] / "agent" / "docs" / "hr"

SAMPLE = """---
doc_id: sample-policy
title: Sample Policy
category: sample
version: "1.0"
last_updated: 2026-01-01
---

# Sample Policy

## Alpha

Alpha text.

### Alpha Detail

Detail text.

## Beta

- one
- two
"""


def test_parse_and_render_round_trip():
    """Parsing and then rendering a document gives back the same document."""
    doc = parse_document(SAMPLE)
    assert doc.doc_id == "sample-policy"
    assert doc.metadata["last_updated"] == "2026-01-01"  # YAML date normalised to str
    again = parse_document(doc.render())
    assert again == doc


def test_parse_rejects_missing_front_matter_and_bad_ids():
    """Invalid front matter, doc ids and file names are rejected."""
    with pytest.raises(DocumentError):
        parse_document("# No front matter")
    with pytest.raises(DocumentError):
        parse_document(SAMPLE.replace("doc_id: sample-policy", "doc_id: ../etc/passwd"))
    with pytest.raises(DocumentError):
        parse_document(SAMPLE, source_path=Path("other-name.md"))


def test_split_sections_paths():
    """Sections get the correct heading paths."""
    doc = parse_document(SAMPLE)
    paths = [section.path for section in split_sections(doc.body)]
    assert paths == ["Alpha", "Alpha > Alpha Detail", "Beta"]


def test_chunk_ids_and_metadata():
    """Chunks get deterministic ids, a context header and the expected metadata."""
    doc = parse_document(SAMPLE)
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
    assert [c.id for c in chunks] == ["sample-policy#0000", "sample-policy#0001", "sample-policy#0002"]
    assert chunks[0].text.startswith("Document: Sample Policy\nSection: Alpha\n\nAlpha text.")
    meta = chunks[2].metadata
    assert meta["doc_id"] == "sample-policy" and meta["section"] == "Beta" and meta["chunk_index"] == 2
    assert meta["text"] == chunks[2].text and meta["version"] == "1.0"


def test_upsert_section_replace_and_append():
    """A section can be replaced (including its sub-sections) or appended."""
    body = parse_document(SAMPLE).body
    new_body, created = upsert_section(body, "alpha", "## Alpha\n\nNew alpha text.")
    assert created is False
    assert get_section(new_body, "Alpha") == "New alpha text."
    assert "Alpha Detail" not in new_body  # sub-sections belong to the replaced section
    assert get_section(new_body, "Beta") == "- one\n- two"

    new_body, created = upsert_section(new_body, "Gamma", "Gamma text.\n\n### Gamma Sub\n\nMore.")
    assert created is True
    assert list_section_titles(new_body) == ["Alpha", "Beta", "Gamma"]


def test_upsert_section_validation():
    """Empty content and '#'/'##' headings inside new content are rejected."""
    body = parse_document(SAMPLE).body
    with pytest.raises(DocumentError):
        upsert_section(body, "Alpha", "")
    with pytest.raises(DocumentError):
        upsert_section(body, "Alpha", "text\n## Another Section\nmore")


def test_all_shipped_docs_are_valid():
    """Every shipped HR document parses and chunks within the size limit."""
    documents = load_documents(DOCS_DIR)
    assert len(documents) >= 10
    for doc in documents:
        chunks = chunk_document(doc)
        assert chunks, doc.doc_id
        assert all(len(c.text) < 1600 for c in chunks)


def test_clean_metadata():
    """Pinecone metadata drops nulls and stringifies unsupported types."""
    assert _clean_metadata({"a": None, "b": 1, "c": ["x", 2], "d": Path("p")}) == {"b": 1, "c": ["x", "2"], "d": "p"}


class FakeEmbedder:
    """Deterministic offline embedder for pipeline tests."""

    def embed_documents(self, texts):
        """Return a fake vector per text."""
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        """Return a fixed query vector."""
        return [1.0, 1.0]


class FakeStore:
    """Minimal in-memory vector store for pipeline tests."""

    def __init__(self):
        """Start with one stale chunk and one document that no longer exists on disk."""
        self.ids = {"sample-policy#0000", "sample-policy#0009", "removed-doc#0000"}

    def replace_document(self, doc_id, chunks, vectors):
        """Replace a document's ids and report how many stale ones were removed."""
        existing = {i for i in self.ids if i.startswith(doc_id + "#")}
        new_ids = {c.id for c in chunks}
        self.ids = (self.ids - existing) | new_ids
        return len(chunks), len(existing - new_ids)

    def list_doc_ids(self):
        """Return the doc ids present in the store."""
        return {i.split("#")[0] for i in self.ids}

    def delete_document(self, doc_id):
        """Remove all ids of a document."""
        self.ids = {i for i in self.ids if not i.startswith(doc_id + "#")}


def test_pipeline_ingest_with_prune(tmp_path):
    """Ingest with prune removes stale chunks and documents deleted from disk."""
    (tmp_path / "sample-policy.md").write_text(SAMPLE, encoding="utf-8")
    store = FakeStore()
    pipeline = IngestionPipeline(FakeEmbedder(), store, chunk_size=500, chunk_overlap=50)
    results, pruned = pipeline.ingest_directory(tmp_path, prune=True)
    assert results[0].chunks == 3 and results[0].deleted_stale == 1  # #0009 was stale
    assert pruned == ["removed-doc"]
    assert store.list_doc_ids() == {"sample-policy"}
