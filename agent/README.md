# HR Agent (LangGraph agentic RAG + REST API)

## Agentic RAG graph

```
                 ┌──────────────────────────────────────────────┐
                 ▼                                              │
    START ──► agent ──(tool calls?)──► tools ──► grade_retrieval ┤
                 │ no                               │ irrelevant │ relevant / other tool /
                 ▼                                  ▼            │ rewrite budget used up
                END                          rewrite_query ──────┘
```

- **agent** (`gpt-5-mini` with tools bound): decides whether to search, which query and category filter to use, whether to search again, read a full document, look up company details, or update a document.
- **tools**: runs the tool calls the agent requested.
- **grade_retrieval**: after every `search_hr_documents` call, an LLM grader with structured output checks whether the retrieved passages are relevant.
- **rewrite_query**: if they are not, writes a better query and sends the agent back to search again. This happens at most `MAX_QUERY_REWRITES` times per turn. After that, the agent must say the documents don't cover the question instead of guessing.
- **Conversation memory**: each `session_id` is a LangGraph thread kept in an `InMemorySaver` checkpointer.

## Tools

| Tool | Purpose | Registered |
|---|---|---|
| `search_hr_documents(query, category?)` | Semantic search in Pinecone | always |
| `list_hr_documents()` | Document ids, titles, categories, versions, section titles | always |
| `get_hr_document(document_id)` | Full current Markdown of a document | always |
| `get_company_details(section?)` | Company metadata from `docs/company/company_profile.json` | always |
| `update_hr_document_section(document_id, section_title, new_content, change_summary)` | Create or replace a `##` section, re-index it in Pinecone, and write the change back to the `.md` file | **only with write access** |

## Write access (configurable credentials)

| `.env` | Result |
|---|---|
| `PINECONE_ACCESS_MODE=read`, no `PINECONE_WRITE_API_KEY` | Read-only. The update tool is **not registered**, and the agent is told it cannot change documents. |
| `PINECONE_ACCESS_MODE=readwrite` | `PINECONE_API_KEY` is used for writes. |
| `PINECONE_WRITE_API_KEY=...` | Separate key used for writes; reads still use `PINECONE_API_KEY`. |
| `PINECONE_VERIFY_WRITE_ACCESS=true` (default) | At startup the write key is **proven** by upserting and then deleting a probe vector in the namespace `<namespace>__write-probe`. Pinecone rejects a *ReadOnly* key with a 403 and writes are disabled. Any probe error also disables writes (fail closed). |

Writes also require the local `HR_DOCS_DIR` to be writable, because of the write-back.

There are **two** guards. The update tool is only given to the LLM when access is granted, and `DocumentUpdateService` checks access again on every call.

`GET /capabilities` shows the outcome:

```json
{ "write_enabled": false,
  "write_access_reason": "Pinecone rejected a write with PINECONE_API_KEY (PINECONE_ACCESS_MODE=readwrite) (the key does not have write permission)",
  "tools": ["search_hr_documents", "list_hr_documents", "get_hr_document", "get_company_details"] }
```

### What an update does

1. Load the current document from `docs/hr/<doc_id>.md` and replace (or append) the `##` section.
2. Bump `version` (`2.1 → 2.2`) and set `last_updated` to today.
3. Chunk and embed the new version with the same code as `rag-ingestion`.
4. Write the new file to a temp file next to the original.
5. Upsert the chunks to Pinecone. **If this fails, the temp file is discarded and nothing changes.**
6. Atomically replace the Markdown file (write-back).
7. Delete stale chunk ids from Pinecone.
8. Append an audit record to `AUDIT_LOG_PATH` (JSONL, including the session id and change summary).

## Run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # also installs ../rag-ingestion in editable mode
cp .env.example .env                   # fill in keys and choose the access mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Run the ingestion (`hr-ingest ingest` in `../rag-ingestion`) before the first chat. Interactive API docs are served at http://localhost:8000/docs.

## REST API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/capabilities` | Model, index, write access status and active tools |
| `POST` | `/chat` | `{"message": "...", "session_id": "optional"}` → answer, sources, tool calls, updates |
| `DELETE` | `/sessions/{session_id}` | Forget a conversation |
| `GET` | `/documents` | List HR documents (id, title, category, version, last_updated) |

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"How many annual leave days do I get after 6 years?"}'
```

```json
{
  "session_id": "6f0c…",
  "answer": "Employees with 5 or more years of continuous service receive 21 working days of annual leave per leave year…\n\n*Source: Leave Policy > Annual Leave*",
  "sources": [{"chunk_id": "leave-policy#0002", "doc_id": "leave-policy", "title": "Leave Policy", "section": "Annual Leave", "version": "2.1", "source": "leave-policy.md", "score": 0.71}],
  "tool_calls": [{"name": "search_hr_documents", "args": {"query": "annual leave entitlement after 5 years of service"}}],
  "documents_updated": [],
  "query_rewrites": 0
}
```

Update example (needs write access). Pass the same `session_id` for follow-ups:

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"Update the leave policy: paternity leave is now 10 working days instead of 5.", "session_id":"hr-admin-1"}'
```

## Configuration

See `.env.example`. Relative paths are resolved against this `agent/` directory. `CHUNK_SIZE` and `CHUNK_OVERLAP` must match the ingestion project.

## Production notes

- `InMemorySaver` loses conversations on restart and isn't shared between workers. For multi-worker deployments, use a persistent LangGraph checkpointer (for example `langgraph-checkpoint-postgres`).
- The write-back changes the local `docs/` folder. If the docs live in git, commit the changes (the audit log records every update).
- The API has no authentication. Put it behind your gateway or identity provider before exposing it.

## Tests

The tests run offline with a scripted fake LLM and an in-memory vector store:

```bash
pytest -q
```
