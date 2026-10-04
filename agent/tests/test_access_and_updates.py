import json

import pytest
from pinecone import ForbiddenError

from app.access import WriteAccessDenied, resolve_access
from app.tools import UPDATE_TOOL, build_tools
from app.updater import bump_version
from conftest import FakeStore
from hr_rag_ingestion import load_document
from hr_rag_ingestion.documents import get_section


# ----------------------------------------------------------------------------- access resolution
def test_read_mode_disables_write(make_settings):
    """Read mode disables writes and never runs the probe."""
    called = []
    access = resolve_access(make_settings(pinecone_access_mode="read"), probe=called.append)
    assert not access.write_enabled and called == []  # never probes a read-only configuration
    with pytest.raises(WriteAccessDenied):
        access.require_write()


def test_readwrite_mode_verified_by_probe(make_settings):
    """readwrite mode is enabled after a successful probe with the main key."""
    keys = []
    access = resolve_access(make_settings(pinecone_access_mode="readwrite"), probe=keys.append)
    assert access.write_enabled and keys == ["pc-test"]
    assert access.require_write() == "pc-test"


def test_probe_forbidden_disables_write(make_settings):
    """A 403 from Pinecone during the probe disables writes."""
    def probe(_key):
        """Simulate Pinecone rejecting a read-only key."""
        raise ForbiddenError("Forbidden")

    access = resolve_access(make_settings(pinecone_access_mode="readwrite"), probe=probe)
    assert not access.write_enabled
    assert "does not have write permission" in access.reason


def test_probe_unexpected_error_fails_closed(make_settings):
    """Any unexpected probe error disables writes (fail closed)."""
    def probe(_key):
        """Simulate a network failure during the probe."""
        raise ConnectionError("network down")

    assert not resolve_access(make_settings(pinecone_access_mode="readwrite"), probe=probe).write_enabled


def test_separate_write_key_takes_precedence(make_settings):
    """A separate write key enables writes and is the key that gets probed."""
    keys = []
    access = resolve_access(make_settings(pinecone_access_mode="read", pinecone_write_api_key="pc-write"), probe=keys.append)
    assert access.write_enabled and keys == ["pc-write"] and access.write_api_key == "pc-write"


def test_unverified_write_when_probe_disabled(make_settings):
    """With verification off, readwrite mode is enabled but marked as not verified."""
    access = resolve_access(make_settings(pinecone_access_mode="readwrite", pinecone_verify_write_access=False), probe=None)
    assert access.write_enabled and "not verified" in access.reason


def test_missing_docs_dir_disables_write(make_settings, tmp_path):
    """Writes are disabled when the docs folder does not exist."""
    settings = make_settings(pinecone_access_mode="readwrite", hr_docs_dir=tmp_path / "missing")
    assert not resolve_access(settings, probe=lambda _k: None).write_enabled


# ----------------------------------------------------------------------------- tool gating
def test_update_tool_only_registered_with_write_access(make_services):
    """The update tool exists only when write access is enabled."""
    assert UPDATE_TOOL not in [t.name for t in build_tools(make_services(write_enabled=False))]
    assert UPDATE_TOOL in [t.name for t in build_tools(make_services(write_enabled=True))]


def test_updater_refuses_without_write_access(make_services):
    """The update service itself refuses writes without access."""
    services = make_services(write_enabled=False)
    with pytest.raises(WriteAccessDenied):
        services.updater.update_section("leave-policy", "Paternity Leave", "- 10 days.", "test")


# ----------------------------------------------------------------------------- updates
def test_bump_version():
    """Version numbers are bumped correctly."""
    assert bump_version("2.1") == "2.2"
    assert bump_version("3") == "3.1"
    assert bump_version("1.9") == "1.10"
    assert bump_version("draft") == "draft.1"


def test_update_section_indexes_and_writes_back(make_services):
    """An update re-indexes Pinecone, removes stale chunks, writes back the file and is audited."""
    store = FakeStore()
    store.vectors["leave-policy#0099"] = {"stale": True}  # left over from an older version
    services = make_services(write_enabled=True, write_store=store)
    new_text = "- Male employees are entitled to **10 working days** of fully paid paternity leave, to be taken within **3 months** of the birth or adoption of a child."

    tools = {t.name: t for t in build_tools(services)}
    message = tools[UPDATE_TOOL].invoke(
        {
            "type": "tool_call",
            "id": "call-1",
            "name": UPDATE_TOOL,
            "args": {
                "document_id": "leave-policy",
                "section_title": "Paternity Leave",
                "new_content": new_text,
                "change_summary": "Increase paternity leave from 5 to 10 days.",
            },
        },
        config={"configurable": {"thread_id": "session-42"}},
    )
    assert message.content.startswith("SUCCESS"), message.content
    assert message.artifact["previous_version"] == "2.1" and message.artifact["new_version"] == "2.2"

    # Write-back to the Markdown file.
    doc = load_document(services.settings.hr_docs_dir / "leave-policy.md")
    assert doc.version == "2.2"
    assert get_section(doc.body, "Paternity Leave") == new_text
    assert get_section(doc.body, "Bereavement Leave") is not None  # other sections untouched
    assert not list(services.settings.hr_docs_dir.glob(".*.tmp"))

    # Pinecone: new chunks upserted with the new version, stale chunk removed.
    assert "leave-policy#0099" not in store.vectors
    assert all(meta["version"] == "2.2" for meta in store.vectors.values())
    assert any("10 working days" in meta["text"] for meta in store.vectors.values())

    # Audit log.
    entry = json.loads(services.settings.audit_log_path.read_text().strip())
    assert entry["requested_by"] == "session-42" and entry["doc_id"] == "leave-policy"


def test_failed_pinecone_upsert_leaves_file_unchanged(make_services):
    """If the Pinecone upsert fails, the Markdown file is left unchanged."""
    services = make_services(write_enabled=True, write_store=FakeStore(fail_upsert=True))
    path = services.settings.hr_docs_dir / "leave-policy.md"
    before = path.read_text()
    with pytest.raises(RuntimeError):
        services.updater.update_section("leave-policy", "Paternity Leave", "- 10 days.", "test")
    assert path.read_text() == before
    assert not list(services.settings.hr_docs_dir.glob(".*.tmp"))


def test_update_tool_reports_errors(make_services):
    """The update tool returns a readable error for an unknown document."""
    tools = {t.name: t for t in build_tools(make_services(write_enabled=True))}
    result = tools[UPDATE_TOOL].invoke(
        {"document_id": "no-such-doc", "section_title": "X", "new_content": "y", "change_summary": "z"}
    )
    assert result.startswith("ERROR") and "leave-policy" in result


def test_company_tool(make_services):
    """The company tool returns one section and rejects unknown sections."""
    tools = {t.name: t for t in build_tools(make_services())}
    data = json.loads(tools["get_company_details"].invoke({"section": "headquarters"}))
    assert data["headquarters"]["address"]["country"] == "Sri Lanka"
    assert tools["get_company_details"].invoke({"section": "bogus"}).startswith("ERROR")
