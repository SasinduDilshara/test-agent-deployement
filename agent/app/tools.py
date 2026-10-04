"""LangChain tools exposed to the HR agent.

Read tools are always registered. The document-update tool is registered ONLY when the
configured credentials have write access (see ``app.access``); the update service also
re-checks access on every call.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.access import WriteAccessDenied
from app.doc_repository import DocumentNotFound
from app.services import Services
from hr_rag_ingestion import DocumentError
from hr_rag_ingestion.documents import list_section_titles

logger = logging.getLogger(__name__)

SEARCH_TOOL = "search_hr_documents"
LIST_DOCS_TOOL = "list_hr_documents"
GET_DOC_TOOL = "get_hr_document"
COMPANY_TOOL = "get_company_details"
UPDATE_TOOL = "update_hr_document_section"
NO_RESULTS = "NO_RESULTS"


class SearchArgs(BaseModel):
    """Arguments of the search_hr_documents tool."""

    query: str = Field(description="A self-contained natural-language search query, e.g. 'annual leave entitlement after 5 years of service'.")
    category: str | None = Field(
        default=None,
        description="Optional document category to restrict the search to (see list_hr_documents for categories). Omit to search all HR documents.",
    )


class GetDocumentArgs(BaseModel):
    """Arguments of the get_hr_document tool."""

    document_id: str = Field(description="The document id, e.g. 'leave-policy'.")


class CompanyArgs(BaseModel):
    """Arguments of the get_company_details tool."""

    section: str | None = Field(
        default=None,
        description=(
            "Optional section of the company profile: company, headquarters, offices, headcount, leadership, "
            "departments, products, certifications, key_contacts, working_calendar, social. Omit for everything."
        ),
    )


class UpdateSectionArgs(BaseModel):
    """Arguments of the update_hr_document_section tool."""

    document_id: str = Field(description="Id of the HR document to change, e.g. 'leave-policy'.")
    section_title: str = Field(
        description="Exact text of the '##' section heading to replace (e.g. 'Annual Leave'). If no section has this title, a new section is appended."
    )
    new_content: str = Field(
        description=(
            "The COMPLETE new Markdown body of the section (without the '## heading' line). It replaces the whole "
            "existing section, so include all unchanged sentences too. Use '###' for sub-headings; never '#' or '##'."
        )
    )
    change_summary: str = Field(description="One-sentence summary of what changed and why, for the audit log.")


def _format_results(results) -> str:
    """Format search results as numbered passages with their source for the LLM."""
    blocks = []
    for rank, result in enumerate(results, start=1):
        meta = result.metadata
        blocks.append(
            f"[{rank}] {result.title} > {result.section} "
            f"(doc_id: {result.doc_id}, version: {meta.get('version', '?')}, "
            f"last_updated: {meta.get('last_updated', '?')}, score: {result.score:.3f})\n{result.text}"
        )
    return "\n\n---\n\n".join(blocks)


def build_tools(services: Services) -> list[BaseTool]:
    """Create the agent's tools; the update tool is included only when write access is enabled."""
    settings = services.settings

    @tool(SEARCH_TOOL, args_schema=SearchArgs, response_format="content_and_artifact")
    def search_hr_documents(query: str, category: str | None = None) -> tuple[str, list[dict]]:
        """Semantic search over the Lumora Systems HR policy documents indexed in Pinecone.
        Use it for ANY question about HR policies (leave, pay, benefits, conduct, remote work, travel,
        performance, onboarding, resignation, ...). Returns the most relevant passages with their source."""
        metadata_filter = {"category": {"$eq": category.strip()}} if category and category.strip() else None
        try:
            results = services.retriever.search(query, top_k=settings.retrieval_top_k, metadata_filter=metadata_filter)
        except Exception as exc:
            logger.exception("HR document search failed")
            return f"ERROR: the HR document search failed ({type(exc).__name__}: {exc}).", []
        if not results:
            hint = f" in category '{category}'" if metadata_filter else ""
            return f"{NO_RESULTS}: no passages found{hint} for query '{query}'.", []
        artifact = [
            {
                "chunk_id": r.id,
                "doc_id": r.doc_id,
                "title": r.title,
                "section": r.section,
                "version": r.metadata.get("version"),
                "source": r.metadata.get("source"),
                "score": round(r.score, 4),
            }
            for r in results
        ]
        return _format_results(results), artifact

    @tool(LIST_DOCS_TOOL)
    def list_hr_documents() -> str:
        """List all HR documents with their id, title, category, version, last update date and section titles."""
        try:
            documents = services.repository.list_documents()
        except Exception as exc:
            return f"ERROR: could not list HR documents ({exc})."
        lines = []
        for doc in documents:
            sections = "; ".join(list_section_titles(doc.body))
            lines.append(
                f"- {doc.doc_id} | {doc.title} | category: {doc.category} | version: {doc.version} | "
                f"last_updated: {doc.metadata.get('last_updated', '?')}\n  sections: {sections}"
            )
        return "\n".join(lines) or "No HR documents found."

    @tool(GET_DOC_TOOL, args_schema=GetDocumentArgs)
    def get_hr_document(document_id: str) -> str:
        """Return the full current Markdown text of one HR document (including front matter).
        Use it before editing a document, or when the user needs the complete policy text."""
        try:
            return services.repository.get(document_id).render()
        except (DocumentNotFound, DocumentError) as exc:
            return f"ERROR: {exc}"

    @tool(COMPANY_TOOL, args_schema=CompanyArgs)
    def get_company_details(section: str | None = None) -> str:
        """Get Lumora Systems company metadata (NOT HR policy): legal name, registration, founding date,
        headquarters and office locations, headcount, leadership team, departments and their heads,
        products, certifications, key contact emails, fiscal year and core values."""
        try:
            return json.dumps(services.company.get(section), indent=2, ensure_ascii=False)
        except KeyError as exc:
            return f"ERROR: {exc.args[0]}"
        except Exception as exc:
            logger.exception("Company profile lookup failed")
            return f"ERROR: could not read the company profile ({exc})."

    tools: list[BaseTool] = [search_hr_documents, list_hr_documents, get_hr_document, get_company_details]

    if services.access.write_enabled:

        @tool(UPDATE_TOOL, args_schema=UpdateSectionArgs, response_format="content_and_artifact")
        def update_hr_document_section(
            document_id: str,
            section_title: str,
            new_content: str,
            change_summary: str,
            config: RunnableConfig,
        ) -> tuple[str, dict | None]:
            """Create or replace one '##' section of an HR document. The change is re-embedded and upserted
            into Pinecone and written back to the Markdown source file; the document version is bumped.
            Only use when the user explicitly asks to change a policy."""
            requested_by = str((config or {}).get("configurable", {}).get("thread_id", "unknown"))
            try:
                result = services.updater.update_section(document_id, section_title, new_content, change_summary, requested_by)
            except WriteAccessDenied as exc:
                return f"DENIED: {exc}", None
            except (DocumentNotFound, DocumentError, ValueError) as exc:
                return f"ERROR: {exc}", None
            except Exception as exc:
                logger.exception("Document update failed")
                return f"ERROR: the update failed and no changes were committed ({type(exc).__name__}: {exc}).", None
            action = "Created new section" if result.section_created else "Updated section"
            message = (
                f"SUCCESS: {action} '{result.section_title}' in '{result.title}' ({result.doc_id}). "
                f"Version {result.previous_version} -> {result.new_version}, last_updated {result.last_updated}. "
                f"Re-indexed {result.chunks_indexed} chunks ({result.stale_chunks_deleted} stale removed) and wrote back to "
                f"{result.file_path}."
            )
            if result.warnings:
                message += " WARNINGS: " + " ".join(result.warnings)
            return message, asdict(result)

        tools.append(update_hr_document_section)

    return tools
