"""REST API request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body of POST /chat."""

    message: str = Field(min_length=1, max_length=8000, description="The user's message.")
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:-]+$",
        description="Conversation id. Omit to start a new conversation; reuse the returned id for follow-ups.",
    )


class Source(BaseModel):
    """A Pinecone passage retrieved during the turn."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    version: str | None = None
    source: str | None = None
    score: float


class ToolCallInfo(BaseModel):
    """A tool call made by the agent during the turn."""

    name: str
    args: dict[str, Any]


class ChatResponse(BaseModel):
    """Response body of POST /chat."""

    session_id: str
    answer: str
    sources: list[Source] = Field(description="Passages retrieved from Pinecone during this turn.")
    tool_calls: list[ToolCallInfo]
    documents_updated: list[dict[str, Any]] = Field(description="Details of documents changed during this turn.")
    query_rewrites: int = Field(description="How many times the agent rewrote its search query after relevance grading.")


class CapabilitiesResponse(BaseModel):
    """Response body of GET /capabilities."""

    model: str
    embedding_model: str
    pinecone_index: str
    pinecone_namespace: str
    write_enabled: bool
    write_access_reason: str
    tools: list[str]


class DocumentSummary(BaseModel):
    """One entry of GET /documents."""

    doc_id: str
    title: str
    category: str
    version: str
    last_updated: str | None = None
