# Lumora Systems HR Agent

An HR assistant built with **LangGraph** that uses **agentic RAG** over HR policies stored in **Pinecone**. It uses OpenAI `gpt-5-mini` for chat and `text-embedding-3-small` for embeddings, and is served as a **REST API** with FastAPI.

```
hr-agent/
├── agent/               # LangGraph agentic-RAG HR agent exposed as a FastAPI REST API
│   └── docs/            # Source documents bundled with the agent (used by default, no paths to configure)
│       ├── hr/          #   10 HR policy documents (Markdown + YAML front matter)
│       └── company/     #   company_profile.json – company metadata (not HR policy)
└── rag-ingestion/       # RAG ingestion + query project (chunk → embed → Pinecone), CLI: hr-ingest
```

## How the pieces fit

```
agent/docs/hr/*.md ──► rag-ingestion (OpenAI embeddings)──► Pinecone index / namespace
                                                              ▲   │
                                       search (read key)      │   │ upsert (write key, only if allowed)
                                                              │   ▼
REST client ──► agent (FastAPI) ──► LangGraph agentic RAG ──► tools ──► agent/docs/hr/*.md (write-back)
                                                              └──► agent/docs/company/company_profile.json
```

The agent installs `rag-ingestion` as a library. When the agent updates a document, it is chunked, embedded and indexed with **exactly the same code** as the batch ingestion.

## Quick start

Requires **Python 3.10 or newer** (3.11 or 3.12 recommended). The macOS system Python 3.9 is too old.

```bash
# 1) Ingest the documents into Pinecone
cd rag-ingestion
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # set OPENAI_API_KEY, PINECONE_API_KEY (read-write key), index name
hr-ingest ingest                  # creates the serverless index if needed, then indexes all docs
hr-ingest query "How many annual leave days do I get?"
deactivate

# 2) Run the agent
cd ../agent
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # set keys; choose PINECONE_ACCESS_MODE=read or readwrite
uvicorn app.main:app --port 8000
```

```bash
curl -s localhost:8000/capabilities | jq
curl -s localhost:8000/chat -H 'content-type: application/json' \
     -d '{"message":"How much paternity leave do we get?"}' | jq
```

See [`rag-ingestion/README.md`](rag-ingestion/README.md) and [`agent/README.md`](agent/README.md) for details.
