"""LangGraph agentic-RAG workflow.

                 ┌──────────────────────────────────────────────┐
                 ▼                                              │
    START ──► agent ──(tool calls?)──► tools ──► grade_retrieval ┤
                 │ no                               │ irrelevant │ relevant / non-search tool /
                 ▼                                  ▼            │ rewrite budget exhausted
                END                          rewrite_query ──────┘

* ``agent``          – the LLM decides whether to answer directly or call tools
                       (search, list/get documents, company details, update).
* ``tools``          – executes the requested tool calls.
* ``grade_retrieval``– when the agent searched the HR documents, an LLM grader checks whether the
                       retrieved passages are relevant to the user's question.
* ``rewrite_query``  – if not, produces a better query and sends the agent back to search again
                       (bounded by ``MAX_QUERY_REWRITES``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from app.prompts import (
    AGENT_SYSTEM_PROMPT,
    GRADER_PROMPT,
    REWRITE_FEEDBACK,
    REWRITE_PROMPT,
    WRITE_DISABLED_INSTRUCTIONS,
    WRITE_ENABLED_INSTRUCTIONS,
)
from app.tools import SEARCH_TOOL

logger = logging.getLogger(__name__)

RETRIEVAL_CHECK_NAME = "retrieval_check"


class AgentState(MessagesState):
    """Graph state: the conversation messages plus the per-turn question and rewrite counter."""

    question: str  # the user's latest message (set per request)
    rewrite_count: int  # query rewrites used in the current turn (reset per request)


class RelevanceGrade(BaseModel):
    """Binary relevance grade for retrieved passages."""

    relevant: Literal["yes", "no"] = Field(description="'yes' if the passages are relevant to the request, otherwise 'no'.")


def is_user_message(message: AnyMessage) -> bool:
    """True for real user messages, False for the agent's internal retrieval-check feedback."""
    return isinstance(message, HumanMessage) and message.name != RETRIEVAL_CHECK_NAME


def current_turn(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Messages after (and including) the latest real user message."""
    for index in range(len(messages) - 1, -1, -1):
        if is_user_message(messages[index]):
            return list(messages[index:])
    return list(messages)


def _latest_tool_round(messages: Sequence[AnyMessage]) -> tuple[AIMessage | None, list[ToolMessage]]:
    """Return the most recent AI message and the tool results that followed it."""
    tool_messages: list[ToolMessage] = []
    index = len(messages) - 1
    while index >= 0 and isinstance(messages[index], ToolMessage):
        tool_messages.insert(0, messages[index])
        index -= 1
    ai_message = messages[index] if index >= 0 and isinstance(messages[index], AIMessage) else None
    return ai_message, tool_messages


def _search_queries(messages: Sequence[AnyMessage]) -> list[str]:
    """Collect the search queries the agent issued in the given messages."""
    return [
        str(call["args"].get("query", ""))
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] == SEARCH_TOOL
    ]


def build_graph(
    model: Any,
    tools: list[BaseTool],
    *,
    write_enabled: bool,
    write_reason: str,
    catalog_provider: Callable[[], str],
    max_rewrites: int = 2,
    checkpointer: Any = None,
):
    """Compile the agentic-RAG graph.

    ``model`` is a LangChain chat model (``ChatOpenAI``); it must support ``bind_tools`` and
    ``with_structured_output``.
    """
    model_with_tools = model.bind_tools(tools)
    grader = model.with_structured_output(RelevanceGrade)

    def system_prompt() -> str:
        """Build the system prompt with today's date, the write-access instructions and the document catalog."""
        write_instructions = (
            WRITE_ENABLED_INSTRUCTIONS if write_enabled else WRITE_DISABLED_INSTRUCTIONS.format(reason=write_reason)
        )
        try:
            catalog = catalog_provider()
        except Exception as exc:  # the catalog is a convenience; never fail the request because of it
            logger.warning("Could not build document catalog: %s", exc)
            catalog = "(unavailable - use list_hr_documents)"
        return AGENT_SYSTEM_PROMPT.format(today=date.today().isoformat(), write_instructions=write_instructions, catalog=catalog)

    # ------------------------------------------------------------------ nodes
    def agent(state: AgentState) -> dict:
        """Agent node: call the LLM with the tools bound and append its response."""
        response = model_with_tools.invoke([SystemMessage(content=system_prompt()), *state["messages"]])
        return {"messages": [response]}

    def grade_retrieval(state: AgentState) -> Literal["agent", "rewrite_query"]:
        """Router after the tools node: grade search results and continue to the agent or to rewrite_query."""
        ai_message, tool_messages = _latest_tool_round(state["messages"])
        search_messages = [m for m in tool_messages if m.name == SEARCH_TOOL]
        if not search_messages:
            return "agent"  # company details, document listing, updates, ... need no grading
        if state.get("rewrite_count", 0) >= max_rewrites:
            return "agent"  # budget exhausted: the agent must answer with what it has
        if all(str(m.content).startswith("ERROR") for m in search_messages):
            return "agent"  # infrastructure error: rewriting the query will not help
        with_results = [m for m in search_messages if m.artifact]
        if not with_results:
            return "rewrite_query"

        queries = [call["args"].get("query", "") for call in (ai_message.tool_calls if ai_message else []) if call["name"] == SEARCH_TOOL]
        prompt = GRADER_PROMPT.format(
            question=state.get("question", ""),
            query=" | ".join(queries),
            context="\n\n".join(str(m.content) for m in with_results),
        )
        try:
            grade = grader.invoke(prompt)
        except Exception:
            logger.exception("Relevance grading failed; continuing without grading")
            return "agent"
        logger.info("Retrieval graded relevant=%s for queries %s", grade.relevant, queries)
        return "agent" if grade.relevant == "yes" else "rewrite_query"

    def rewrite_query(state: AgentState) -> dict:
        """Rewrite node: generate a better search query and tell the agent to search again."""
        tried = _search_queries(current_turn(state["messages"]))
        question = state.get("question", "")
        try:
            response = model.invoke(REWRITE_PROMPT.format(question=question, tried="\n".join(f"- {q}" for q in tried) or "- (none)"))
            new_query = response.text.strip().strip('"').strip()
        except Exception:
            logger.exception("Query rewrite failed")
            new_query = ""
        new_query = new_query or question
        logger.info("Rewrote query (attempt %d): %s", state.get("rewrite_count", 0) + 1, new_query)
        return {
            "messages": [HumanMessage(content=REWRITE_FEEDBACK.format(query=new_query), name=RETRIEVAL_CHECK_NAME)],
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    # ------------------------------------------------------------------ graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("rewrite_query", rewrite_query)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", grade_retrieval, {"agent": "agent", "rewrite_query": "rewrite_query"})
    graph.add_edge("rewrite_query", "agent")
    return graph.compile(checkpointer=checkpointer)
