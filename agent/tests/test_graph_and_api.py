from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.agent import HRAgent
from app.main import app
from app.tools import COMPANY_TOOL, SEARCH_TOOL, UPDATE_TOOL
from conftest import FakeChatModel, make_result, tool_call


def test_agentic_rag_rewrites_irrelevant_retrieval(make_services):
    """Irrelevant results trigger a query rewrite and a second search before answering."""
    services = make_services(results=[make_result()])
    model = FakeChatModel(
        agent_responses=[
            AIMessage(content="", tool_calls=[tool_call(SEARCH_TOOL, {"query": "holidays"}, "c1")]),
            AIMessage(content="", tool_calls=[tool_call(SEARCH_TOOL, {"query": "annual leave entitlement days"}, "c2")]),
            AIMessage(content="You get 18 working days.\n\n*Source: Leave Policy > Annual Leave*"),
        ],
        grades=["no", "yes"],
        rewrites=["annual leave entitlement days"],
    )
    agent = HRAgent(services, model=model)
    result = agent.chat("How many vacation days do I get?", "s1")

    assert result.answer.startswith("You get 18 working days")
    assert result.query_rewrites == 1
    assert [c["args"]["query"] for c in result.tool_calls] == ["holidays", "annual leave entitlement days"]
    assert result.sources[0]["doc_id"] == "leave-policy"
    # The second agent call saw the retrieval feedback with the rewritten query.
    assert "annual leave entitlement days" in model.agent_inputs[1][-1].content
    # Read-only: the update tool is not bound and the prompt says so.
    assert UPDATE_TOOL not in model.bound_tools
    assert "You do NOT have write access" in model.agent_inputs[0][0].content


def test_rewrite_budget_is_bounded(make_services):
    """The rewrite loop stops after MAX_QUERY_REWRITES and the agent answers anyway."""
    services = make_services(results=[make_result()], max_query_rewrites=1)
    model = FakeChatModel(
        agent_responses=[
            AIMessage(content="", tool_calls=[tool_call(SEARCH_TOOL, {"query": "q1"}, "c1")]),
            AIMessage(content="", tool_calls=[tool_call(SEARCH_TOOL, {"query": "q2"}, "c2")]),
            AIMessage(content="The HR documents do not cover this."),
        ],
        grades=["no"],  # only one grading happens; afterwards the budget is exhausted
        rewrites=["q2"],
    )
    result = HRAgent(services, model=model).chat("What is the dress code?", "s2")
    assert result.query_rewrites == 1 and len(model.grader_inputs) == 1
    assert result.answer == "The HR documents do not cover this."


def test_company_tool_skips_grading_and_memory_persists(make_services):
    """Non-search tools skip grading, and follow-up turns see the earlier conversation."""
    model = FakeChatModel(
        agent_responses=[
            AIMessage(content="", tool_calls=[tool_call(COMPANY_TOOL, {"section": "leadership"}, "c1")]),
            AIMessage(content="The CEO is Dilani Perera."),
            AIMessage(content="She has been CEO since 2014."),
        ]
    )
    agent = HRAgent(make_services(), model=model)
    first = agent.chat("Who is the CEO?", "s3")
    assert first.answer == "The CEO is Dilani Perera." and model.grader_inputs == []
    second = agent.chat("Since when?", "s3")
    assert second.tool_calls == [] and second.answer.startswith("She has been")
    # Conversation memory: the second call includes the first turn.
    assert any(getattr(m, "content", "") == "Who is the CEO?" for m in model.agent_inputs[2])


def test_write_enabled_prompt_and_tool(make_services):
    """With write access the update tool is bound and the prompt allows updates."""
    model = FakeChatModel(agent_responses=[AIMessage(content="ok")])
    HRAgent(make_services(write_enabled=True), model=model).chat("hi", "s4")
    assert UPDATE_TOOL in model.bound_tools
    assert "You HAVE write access" in model.agent_inputs[0][0].content


def test_rest_api(make_services):
    """The REST endpoints return the expected status codes and payloads."""
    model = FakeChatModel(
        agent_responses=[
            AIMessage(content="", tool_calls=[tool_call(SEARCH_TOOL, {"query": "annual leave"}, "c1")]),
            AIMessage(content="18 working days."),
        ],
        grades=["yes"],
    )
    app.state.agent = HRAgent(make_services(results=[make_result()]), model=model)
    client = TestClient(app)  # not used as a context manager, so the real lifespan is skipped

    caps = client.get("/capabilities").json()
    assert caps["write_enabled"] is False and UPDATE_TOOL not in caps["tools"]

    response = client.post("/chat", json={"message": "How much annual leave?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "18 working days." and body["session_id"]
    assert body["sources"][0]["section"] == "Annual Leave"

    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.delete(f"/sessions/{body['session_id']}").status_code == 204
    assert len(client.get("/documents").json()) == 10
