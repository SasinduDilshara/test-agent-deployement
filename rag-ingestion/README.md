# HR RAG Ingestion

This project loads the HR Markdown documents from `../docs/hr`, chunks them by structure, embeds them with OpenAI `text-embedding-3-small` (1536 dimensions), and upserts them into a Pinecone serverless index. It also includes the **query** code used for retrieval, which the agent reuses as a library.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # installs this package in editable mode + pytest
cp .env.example .env                 # fill in OPENAI_API_KEY and PINECONE_API_KEY
```

The Pinecone key used here must be able to **write** (role *ReadWrite* or *All*).

## Usage

```bash
hr-ingest ingest --dry-run           # parse + chunk only, no API calls, no keys needed
hr-ingest ingest                     # create the index if missing, then index every document
hr-ingest ingest --prune             # also delete documents from the index that no longer exist on disk
hr-ingest query "notice period for L5 employees" --top-k 3
hr-ingest query "gym allowance" --category benefits
hr-ingest stats                      # vector counts per namespace and indexed doc ids
hr-ingest delete-doc travel-and-expense
```

`python -m hr_rag_ingestion <command>` works too.

Ingestion is **idempotent**. Chunk ids are deterministic (`<doc_id>#0000`, `<doc_id>#0001`, ...), so re-running replaces vectors in place. When a document shrinks, the chunks it no longer has are deleted. New chunks are upserted *before* the stale ones are removed, so a document never disappears from search while it is being re-indexed.

## Chunking strategy

1. Split the Markdown body on `#`, `##` and `###` headings, ignoring headings inside code fences.
2. Split long sections with a recursive character splitter (`CHUNK_SIZE=1200`, `CHUNK_OVERLAP=150`).
3. Add a context header to each chunk before embedding:
   ```
   Document: Leave Policy
   Section: Annual Leave

   - Permanent employees are entitled to **18 working days** ...
   ```

## Pinecone record

| field | value |
|---|---|
| `id` | `leave-policy#0002` |
| `values` | 1536-dim embedding |
| `metadata` | `doc_id`, `title`, `category`, `section`, `chunk_index`, `version`, `effective_date`, `last_updated`, `owner`, `applies_to`, `source`, `text` |

## Library API (used by the agent)

```python
from hr_rag_ingestion import IngestionPipeline, OpenAIEmbedder, PineconeVectorStore

pipeline = IngestionPipeline(
    OpenAIEmbedder(api_key="sk-...", model="text-embedding-3-small", dimensions=1536),
    PineconeVectorStore(api_key="pcsk_...", index_name="lumora-hr-docs", namespace="hr-policies"),
)
results = pipeline.search("maternity leave days", top_k=5, metadata_filter={"category": {"$eq": "leave"}})
```

## Tests

```bash
pytest -q
```
