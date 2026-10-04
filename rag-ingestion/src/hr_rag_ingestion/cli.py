"""Command-line interface.

    python -m hr_rag_ingestion ingest [--docs-dir PATH] [--prune] [--dry-run]
    python -m hr_rag_ingestion query "How many annual leave days do I get?" [--top-k 5] [--category leave]
    python -m hr_rag_ingestion stats
    python -m hr_rag_ingestion delete-doc <doc_id>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from hr_rag_ingestion.config import IngestionSettings
from hr_rag_ingestion.documents import DocumentError, validate_doc_id
from hr_rag_ingestion.embeddings import OpenAIEmbedder
from hr_rag_ingestion.pipeline import IngestionPipeline
from hr_rag_ingestion.vector_store import PineconeVectorStore


class ConfigurationError(RuntimeError):
    """Raised when a required setting such as an API key is missing."""

    pass


def _build(settings: IngestionSettings) -> IngestionPipeline:
    """Create the ingestion pipeline (OpenAI embedder + Pinecone store) from the settings."""
    missing = [name for name in ("openai_api_key", "pinecone_api_key") if getattr(settings, name) is None]
    if missing:
        raise ConfigurationError(f"Missing required setting(s): {', '.join(n.upper() for n in missing)} (set them in .env).")
    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    store = PineconeVectorStore(
        api_key=settings.pinecone_api_key.get_secret_value(),
        index_name=settings.pinecone_index_name,
        namespace=settings.pinecone_namespace,
        host=settings.pinecone_index_host,
    )
    return IngestionPipeline(embedder, store, settings.chunk_size, settings.chunk_overlap)


def cmd_ingest(settings: IngestionSettings, args: argparse.Namespace) -> int:
    """Handle `ingest`: create the index if needed and index all documents (or only chunk them in dry-run)."""
    docs_dir = Path(args.docs_dir).resolve() if args.docs_dir else settings.docs_dir
    if args.dry_run:
        pipeline = IngestionPipeline(embedder=None, store=None, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    else:
        pipeline = _build(settings)
        created = pipeline.store.ensure_index(
            dimension=settings.embedding_dimensions,
            metric=settings.pinecone_metric,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
        print(f"Index '{settings.pinecone_index_name}': {'created' if created else 'exists'}")

    results, pruned = pipeline.ingest_directory(docs_dir, prune=args.prune, dry_run=args.dry_run)
    print(f"\n{'DRY RUN - ' if args.dry_run else ''}Documents from {docs_dir}:")
    for result in results:
        stale = f", {result.deleted_stale} stale removed" if result.deleted_stale else ""
        print(f"  - {result.doc_id:<34} {result.chunks:>3} chunks{stale}")
    print(f"Total: {len(results)} documents, {sum(r.chunks for r in results)} chunks "
          f"-> namespace '{settings.pinecone_namespace}'")
    if pruned:
        print(f"Pruned documents no longer on disk: {', '.join(pruned)}")
    return 0


def cmd_query(settings: IngestionSettings, args: argparse.Namespace) -> int:
    """Handle `query`: run a semantic search and print the matching chunks."""
    pipeline = _build(settings)
    metadata_filter = {"category": {"$eq": args.category}} if args.category else None
    results = pipeline.search(args.query, top_k=args.top_k, metadata_filter=metadata_filter)
    if not results:
        print("No results.")
        return 0
    for rank, result in enumerate(results, start=1):
        print(f"\n[{rank}] score={result.score:.4f}  {result.title} > {result.section}  ({result.id})")
        print("-" * 80)
        print(result.text)
    return 0


def cmd_stats(settings: IngestionSettings, args: argparse.Namespace) -> int:
    """Handle `stats`: print index/namespace vector counts and the indexed doc ids."""
    pipeline = _build(settings)
    stats = pipeline.store.describe_stats()
    print(f"Index: {settings.pinecone_index_name}  dimension={stats.dimension}  total_vectors={stats.total_vector_count}")
    for name, summary in sorted(stats.namespaces.items()):
        marker = "  <- configured" if name == settings.pinecone_namespace else ""
        print(f"  namespace '{name}': {summary.vector_count} vectors{marker}")
    doc_ids = sorted(pipeline.store.list_doc_ids())
    print(f"Documents in '{settings.pinecone_namespace}': {', '.join(doc_ids) if doc_ids else '(none)'}")
    return 0


def cmd_delete_doc(settings: IngestionSettings, args: argparse.Namespace) -> int:
    """Handle `delete-doc`: remove all chunks of one document from the index."""
    doc_id = validate_doc_id(args.doc_id)
    deleted = _build(settings).store.delete_document(doc_id)
    print(f"Deleted {deleted} chunks of '{doc_id}' from namespace '{settings.pinecone_namespace}'.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI commands and their arguments."""
    parser = argparse.ArgumentParser(prog="hr-ingest", description="HR documents RAG ingestion for Pinecone.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Chunk, embed and upsert all HR documents.")
    ingest.add_argument("--docs-dir", help="Override DOCS_DIR.")
    ingest.add_argument("--prune", action="store_true", help="Delete indexed documents that no longer exist on disk.")
    ingest.add_argument("--dry-run", action="store_true", help="Parse and chunk only; no API calls.")
    ingest.set_defaults(func=cmd_ingest)

    query = sub.add_parser("query", help="Run a semantic search against the index.")
    query.add_argument("query")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--category", help="Restrict to a document category (e.g. leave, benefits).")
    query.set_defaults(func=cmd_query)

    stats = sub.add_parser("stats", help="Show index / namespace statistics.")
    stats.set_defaults(func=cmd_stats)

    delete = sub.add_parser("delete-doc", help="Remove one document's chunks from the index.")
    delete.add_argument("doc_id")
    delete.set_defaults(func=cmd_delete_doc)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse arguments, load settings and run the selected command."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        settings = IngestionSettings()
        return args.func(settings, args)
    except DocumentError as exc:
        print(f"Document error: {exc}", file=sys.stderr)
        return 2
    except (ConfigurationError, ValidationError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
