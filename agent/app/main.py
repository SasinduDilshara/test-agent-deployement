"""FastAPI application exposing the HR agent as a REST API.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from langgraph.errors import GraphRecursionError

from app.agent import HRAgent
from app.config import get_settings
from app.schemas import CapabilitiesResponse, ChatRequest, ChatResponse, DocumentSummary
from app.services import build_services

logger = logging.getLogger("hr_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load settings, resolve write access and build the HR agent."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Building services resolves write access (including the optional Pinecone write probe).
    services = await run_in_threadpool(build_services, settings)
    app.state.agent = HRAgent(services)
    logger.info("HR agent ready. Tools: %s", ", ".join(app.state.agent.tool_names))
    yield


app = FastAPI(
    title="Lumora Systems HR Agent",
    version="1.0.0",
    description="Agentic RAG HR assistant (LangGraph + OpenAI + Pinecone).",
    lifespan=lifespan,
)


def _agent(request: Request) -> HRAgent:
    """Return the HR agent created at startup."""
    return request.app.state.agent


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/capabilities", response_model=CapabilitiesResponse, tags=["system"])
def capabilities(request: Request) -> CapabilitiesResponse:
    """Report the model, Pinecone index, write-access status and active tools."""
    agent = _agent(request)
    settings = agent.services.settings
    return CapabilitiesResponse(
        model=settings.openai_chat_model,
        embedding_model=settings.openai_embedding_model,
        pinecone_index=settings.pinecone_index_name,
        pinecone_namespace=settings.pinecone_namespace,
        write_enabled=agent.services.access.write_enabled,
        write_access_reason=agent.services.access.reason,
        tools=agent.tool_names,
    )


@app.post("/chat", response_model=ChatResponse, tags=["agent"])
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Send a message to the HR agent and return its answer, sources and actions."""
    # Sync endpoint: FastAPI runs it in a worker thread, so the blocking graph call is fine.
    session_id = body.session_id or str(uuid.uuid4())
    try:
        result = _agent(request).chat(body.message.strip(), session_id)
    except GraphRecursionError as exc:
        logger.warning("Recursion limit reached for session %s", session_id)
        raise HTTPException(status_code=500, detail="The agent could not finish within its step limit. Please rephrase the request.") from exc
    except openai.APIError as exc:
        logger.exception("OpenAI API error")
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {exc.__class__.__name__}") from exc
    return ChatResponse(
        session_id=result.session_id,
        answer=result.answer,
        sources=result.sources,
        tool_calls=result.tool_calls,
        documents_updated=result.documents_updated,
        query_rewrites=result.query_rewrites,
    )


@app.delete("/sessions/{session_id}", status_code=204, tags=["agent"])
def delete_session(session_id: str, request: Request) -> None:
    """Forget the conversation history of a session."""
    _agent(request).reset(session_id)


@app.get("/documents", response_model=list[DocumentSummary], tags=["documents"])
def list_documents(request: Request) -> list[DocumentSummary]:
    """List the HR documents with their category, version and last update date."""
    return [
        DocumentSummary(
            doc_id=doc.doc_id,
            title=doc.title,
            category=doc.category,
            version=doc.version,
            last_updated=doc.metadata.get("last_updated"),
        )
        for doc in _agent(request).services.repository.list_documents()
    ]
