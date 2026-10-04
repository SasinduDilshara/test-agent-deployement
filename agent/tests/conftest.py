"""Offline test fixtures: fake LLM, fake embedder and in-memory Pinecone stand-in."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.access import AccessProfile
from app.company import CompanyDirectory
from app.config import Settings
from app.doc_repository import DocumentRepository
from app.services import Services
from app.updater import DocumentUpdateService
from hr_rag_ingestion import IngestionPipeline, SearchResult

DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs"


class FakeEmbedder:
    """Deterministic offline embedder used instead of OpenAI."""

    def embed_documents(self, texts):
        """Return a tiny fake vector per text."""
        return [[1.0, float(len(t))] for t in texts]

    def embed_query(self, text):
        """Return a fixed fake query vector."""
        return [1.0, 0.0]


class FakeStore:
    """In-memory replacement for PineconeVectorStore."""

    def __init__(self, results=None, fail_upsert=False):
        """Create the store with canned search results; optionally make upserts fail."""
        self.vectors: dict[str, dict] = {}
        self.results = results or []
        self.fail_upsert = fail_upsert
        self.queries = []

    def list_ids(self, prefix=None):
        """List stored ids, optionally filtered by prefix."""
        return [i for i in self.vectors if prefix is None or i.startswith(prefix)]

    def upsert_chunks(self, chunks, vectors):
        """Store chunk metadata by id, or raise if configured to fail."""
        if self.fail_upsert:
            raise RuntimeError("pinecone unavailable")
        for chunk in chunks:
            self.vectors[chunk.id] = chunk.metadata
        return len(chunks)

    def delete_ids(self, ids):
        """Remove the given ids from the store."""
        for i in ids:
            self.vectors.pop(i, None)
        return len(ids)

    def query(self, vector, top_k=5, metadata_filter=None):
        """Record the query and return the canned results."""
        self.queries.append((vector, top_k, metadata_filter))
        return self.results[:top_k]


def make_result(doc_id="leave-policy", section="Annual Leave", text="Permanent employees are entitled to 18 working days.", score=0.82):
    """Build a SearchResult that looks like a Leave Policy chunk."""
    return SearchResult(
        id=f"{doc_id}#0002",
        score=score,
        text=f"Document: Leave Policy\nSection: {section}\n\n{text}",
        metadata={"doc_id": doc_id, "title": "Leave Policy", "section": section, "version": "2.1", "source": f"{doc_id}.md"},
    )


class _Bound:
    """Fake model with tools bound; returns the next scripted agent response."""

    def __init__(self, parent):
        """Keep a reference to the parent fake model."""
        self.parent = parent

    def invoke(self, messages):
        """Record the agent input and return the next scripted response."""
        self.parent.agent_inputs.append(list(messages))
        return self.parent.agent_responses.pop(0)


class _Grader:
    """Fake structured-output grader; returns the next scripted relevance grade."""

    def __init__(self, parent, schema):
        """Keep the parent fake model and the output schema."""
        self.parent, self.schema = parent, schema

    def invoke(self, prompt):
        """Record the grading prompt and return the next scripted grade."""
        self.parent.grader_inputs.append(prompt)
        return self.schema(relevant=self.parent.grades.pop(0))


class FakeChatModel:
    """Scripted stand-in for ChatOpenAI supporting bind_tools / with_structured_output / invoke."""

    def __init__(self, agent_responses, grades=(), rewrites=()):
        """Store the scripted agent responses, grades and rewritten queries."""
        self.agent_responses = list(agent_responses)
        self.grades = list(grades)
        self.rewrites = list(rewrites)
        self.agent_inputs, self.grader_inputs, self.bound_tools = [], [], []

    def bind_tools(self, tools):
        """Record the bound tool names and return the fake tool-calling model."""
        self.bound_tools = [t.name for t in tools]
        return _Bound(self)

    def with_structured_output(self, schema):
        """Return the fake relevance grader."""
        return _Grader(self, schema)

    def invoke(self, prompt):
        """Return the next scripted rewritten query."""
        return AIMessage(content=self.rewrites.pop(0))


def tool_call(name, args, call_id):
    """Build a LangChain tool-call dict for a scripted AIMessage."""
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


@pytest.fixture
def docs_dir(tmp_path):
    """Copy the shipped docs into a temp directory so tests can modify them."""
    target = tmp_path / "docs"
    shutil.copytree(DOCS_ROOT, target)
    return target


@pytest.fixture
def make_settings(docs_dir, tmp_path):
    """Factory fixture for Settings pointing at the temp docs, ignoring any .env file."""
    def factory(**overrides):
        """Create Settings with test defaults plus the given overrides."""
        values = dict(
            openai_api_key="sk-test",
            pinecone_api_key="pc-test",
            hr_docs_dir=docs_dir / "hr",
            company_profile_path=docs_dir / "company" / "company_profile.json",
            audit_log_path=tmp_path / "audit.jsonl",
        )
        values.update(overrides)
        return Settings(_env_file=None, **values)

    return factory


@pytest.fixture
def make_services(make_settings):
    """Factory fixture for Services wired with fake embedder/stores and a chosen access level."""
    def factory(write_enabled=False, results=None, write_store=None, **settings_overrides):
        """Build Services that are either read-only or write-enabled."""
        settings = make_settings(**settings_overrides)
        access = AccessProfile(True, "test write access", "pc-write") if write_enabled else AccessProfile(False, "test read-only")
        retriever = IngestionPipeline(FakeEmbedder(), FakeStore(results=results), settings.chunk_size, settings.chunk_overlap)
        write_pipeline = (
            IngestionPipeline(FakeEmbedder(), write_store or FakeStore(), settings.chunk_size, settings.chunk_overlap)
            if write_enabled
            else None
        )
        repository = DocumentRepository(settings.hr_docs_dir)
        return Services(
            settings=settings,
            access=access,
            retriever=retriever,
            repository=repository,
            company=CompanyDirectory(settings.company_profile_path),
            updater=DocumentUpdateService(access, repository, write_pipeline, settings.audit_log_path),
        )

    return factory
