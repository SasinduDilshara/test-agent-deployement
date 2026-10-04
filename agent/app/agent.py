"""High-level HR agent: builds the model, tools and graph, and runs conversation turns."""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_graph, current_turn
from app.services import Services
from app.tools import SEARCH_TOOL, UPDATE_TOOL, build_tools


@dataclass
class ChatResult:
    """Result of one conversation turn: the answer plus sources, tool calls and document updates."""

    session_id: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    documents_updated: list[dict[str, Any]] = field(default_factory=list)
    query_rewrites: int = 0


def build_chat_model(services: Services) -> ChatOpenAI:
    """Create the ChatOpenAI model (gpt-5-mini by default) from the settings."""
    settings = services.settings
    kwargs: dict[str, Any] = {
        "model": settings.openai_chat_model,
        "api_key": settings.openai_api_key.get_secret_value(),
        "max_retries": 3,
        "timeout": 120,
    }
    if settings.openai_reasoning_effort:
        kwargs["reasoning_effort"] = settings.openai_reasoning_effort
    # gpt-5 family models only support the default temperature, so none is set.
    return ChatOpenAI(**kwargs)


class HRAgent:
    """High-level HR agent that owns the tools, the compiled LangGraph and the conversation memory."""

    def __init__(self, services: Services, model: Any | None = None, checkpointer: Any | None = None) -> None:
        """Build the tools (gated by write access) and compile the agentic-RAG graph."""
        self.services = services
        self.tools = build_tools(services)
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = build_graph(
            model or build_chat_model(services),
            self.tools,
            write_enabled=services.access.write_enabled,
            write_reason=services.access.reason,
            catalog_provider=self._catalog,
            max_rewrites=services.settings.max_query_rewrites,
            checkpointer=self.checkpointer,
        )
        self._session_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    @property
    def tool_names(self) -> list[str]:
        """Names of the tools currently available to the agent."""
        return [tool.name for tool in self.tools]

    def _catalog(self) -> str:
        """Render a short list of the HR documents for the system prompt."""
        return "\n".join(
            f"- {doc.doc_id}: {doc.title} (category: {doc.category}, version {doc.version})"
            for doc in self.services.repository.list_documents()
        )

    def _lock_for(self, session_id: str) -> threading.Lock:
        """Return the lock that serialises the turns of one session."""
        with self._locks_guard:
            return self._session_locks[session_id]

    def chat(self, message: str, session_id: str) -> ChatResult:
        """Run one conversation turn for a session and return the agent's answer and metadata."""
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": self.services.settings.graph_recursion_limit,
        }
        # Turns of the same session are serialised so the conversation state stays consistent.
        with self._lock_for(session_id):
            state = self.graph.invoke(
                {"messages": [HumanMessage(content=message)], "question": message, "rewrite_count": 0},
                config=config,
            )
        return self._to_result(session_id, state)

    def reset(self, session_id: str) -> None:
        """Delete the stored conversation history of a session."""
        with self._lock_for(session_id):
            self.checkpointer.delete_thread(session_id)

    @staticmethod
    def _to_result(session_id: str, state: dict[str, Any]) -> ChatResult:
        """Extract the answer, sources, tool calls and updates of the latest turn from the graph state."""
        turn = current_turn(state["messages"])
        result = ChatResult(session_id=session_id, answer="", query_rewrites=state.get("rewrite_count", 0))
        seen_chunks: set[str] = set()
        for message in turn:
            if isinstance(message, AIMessage):
                result.tool_calls.extend({"name": call["name"], "args": call["args"]} for call in message.tool_calls)
            elif isinstance(message, ToolMessage) and message.artifact:
                if message.name == SEARCH_TOOL:
                    for source in message.artifact:
                        if source["chunk_id"] not in seen_chunks:
                            seen_chunks.add(source["chunk_id"])
                            result.sources.append(source)
                elif message.name == UPDATE_TOOL:
                    result.documents_updated.append(message.artifact)
        final = turn[-1] if turn else None
        if isinstance(final, AIMessage):
            result.answer = final.text.strip()
        return result
