"""Updates an HR document section in Pinecone and writes the change back to the Markdown file."""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from app.access import AccessProfile
from app.doc_repository import DocumentRepository
from hr_rag_ingestion import IngestionPipeline, chunk_id_prefix, upsert_section

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UpdateResult:
    """Details of a completed document update (versions, chunk counts, file path, warnings)."""

    doc_id: str
    title: str
    section_title: str
    section_created: bool
    previous_version: str
    new_version: str
    last_updated: str
    chunks_indexed: int
    stale_chunks_deleted: int
    file_path: str
    warnings: list[str] = field(default_factory=list)


def bump_version(version: str) -> str:
    """``"2.1" -> "2.2"``, ``"3" -> "3.1"``; non-numeric versions get ``.1`` appended."""
    parts = str(version).strip().split(".")
    if all(part.isdigit() for part in parts):
        if len(parts) == 1:
            return f"{parts[0]}.1"
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    return f"{version}.1"


class DocumentUpdateService:
    """Applies section updates to HR documents in Pinecone and in the Markdown source files."""

    def __init__(
        self,
        access: AccessProfile,
        repository: DocumentRepository,
        write_pipeline: IngestionPipeline | None,
        audit_log_path: Path,
    ) -> None:
        """Store the access profile, document repository, write pipeline and audit log path."""
        self.access = access
        self.repository = repository
        self.write_pipeline = write_pipeline
        self.audit_log_path = audit_log_path
        self._lock = threading.Lock()

    def update_section(
        self,
        doc_id: str,
        section_title: str,
        new_content: str,
        change_summary: str,
        requested_by: str = "unknown",
    ) -> UpdateResult:
        """Replace or add one '##' section, re-index the document in Pinecone and write it back to disk.

        Raises WriteAccessDenied without write access; if the Pinecone upsert fails nothing is changed.
        """
        # Guard 1: credentials. Raises WriteAccessDenied when writes are not allowed.
        self.access.require_write()
        if self.write_pipeline is None:
            raise RuntimeError("Write pipeline is not configured.")
        if not change_summary or not change_summary.strip():
            raise ValueError("change_summary must describe the change.")

        with self._lock:
            current = self.repository.get(doc_id)
            section_title = re.sub(r"\s+", " ", section_title).strip()
            new_body, created = upsert_section(current.body, section_title, new_content)
            if new_body.strip() == current.body.strip():
                raise ValueError("The new content is identical to the current section; nothing to update.")

            new_version = bump_version(current.version)
            today = date.today().isoformat()
            updated = current.with_changes(body=new_body, version=new_version, last_updated=today)

            # 1. Chunk + embed (no side effects yet).
            prepared = self.write_pipeline.prepare(updated)
            store = self.write_pipeline.store
            # 2. Stage the file next to the target (no visible change yet).
            staged = self.repository.stage(updated)
            try:
                existing_ids = set(store.list_ids(prefix=chunk_id_prefix(doc_id)))
                # 3. Upsert new chunks. If Pinecone rejects this, nothing has changed anywhere.
                store.upsert_chunks(prepared.chunks, prepared.vectors)
            except Exception:
                staged.discard()
                raise
            # 4. Atomically replace the Markdown file (write-back).
            staged.commit()

            # 5. Remove chunk ids that no longer exist in the new version.
            warnings: list[str] = []
            stale = sorted(existing_ids - {chunk.id for chunk in prepared.chunks})
            deleted = 0
            if stale:
                try:
                    deleted = store.delete_ids(stale)
                except Exception as exc:
                    logger.exception("Failed to delete stale chunks for %s", doc_id)
                    warnings.append(f"Could not delete {len(stale)} stale chunk(s): {exc}. Re-run ingestion to clean up.")

            result = UpdateResult(
                doc_id=doc_id,
                title=updated.title,
                section_title=section_title,
                section_created=created,
                previous_version=current.version,
                new_version=new_version,
                last_updated=today,
                chunks_indexed=len(prepared.chunks),
                stale_chunks_deleted=deleted,
                file_path=str(staged.target_path),
                warnings=warnings,
            )
            self._audit(result, change_summary.strip(), requested_by)
            return result

    def _audit(self, result: UpdateResult, change_summary: str, requested_by: str) -> None:
        """Append a JSON audit record of the update to the audit log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "requested_by": requested_by,
            "change_summary": change_summary,
            **asdict(result),
        }
        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("Failed to write audit log entry")
        logger.info("Document updated: %s", entry)
