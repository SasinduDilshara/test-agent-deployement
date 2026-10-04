"""Access to the HR Markdown documents on disk (source of truth for full documents and write-back)."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hr_rag_ingestion import DocumentError, HRDocument, load_document, load_documents
from hr_rag_ingestion.documents import validate_doc_id


class DocumentNotFound(LookupError):
    """Raised when no HR document exists for a given doc_id."""

    pass


@dataclass(frozen=True)
class StagedWrite:
    """A rendered document written to a temp file next to its target, awaiting commit."""

    temp_path: Path
    target_path: Path

    def commit(self) -> None:
        """Atomically replace the target file with the staged temp file."""
        os.replace(self.temp_path, self.target_path)  # atomic on the same filesystem

    def discard(self) -> None:
        """Delete the staged temp file without touching the target file."""
        self.temp_path.unlink(missing_ok=True)


class DocumentRepository:
    """Reads HR Markdown documents from disk and stages atomic write-backs."""

    def __init__(self, docs_dir: Path) -> None:
        """Store the directory that holds the HR Markdown documents."""
        self.docs_dir = docs_dir

    def path_for(self, doc_id: str) -> Path:
        """Return the file path for a doc_id after validating it (prevents path traversal)."""
        # validate_doc_id only allows [a-z0-9-], which rules out path traversal.
        return self.docs_dir / f"{validate_doc_id(doc_id)}.md"

    def list_documents(self) -> list[HRDocument]:
        """Load and return all HR documents in the directory."""
        return load_documents(self.docs_dir)

    def get(self, doc_id: str) -> HRDocument:
        """Load one HR document by id; raises DocumentNotFound listing the valid ids."""
        try:
            path = self.path_for(doc_id)
        except DocumentError as exc:
            raise DocumentNotFound(str(exc)) from exc
        if not path.is_file():
            available = ", ".join(doc.doc_id for doc in self.list_documents())
            raise DocumentNotFound(f"No HR document with id {doc_id!r}. Available ids: {available}")
        return load_document(path)

    def stage(self, document: HRDocument) -> StagedWrite:
        """Write the rendered document to a temp file next to its target, ready to commit()."""
        target = self.path_for(document.doc_id)
        fd, temp_name = tempfile.mkstemp(prefix=f".{document.doc_id}.", suffix=".tmp", dir=self.docs_dir)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(document.render())
        if target.exists():
            os.chmod(temp_name, target.stat().st_mode & 0o777)
        return StagedWrite(Path(temp_name), target)
