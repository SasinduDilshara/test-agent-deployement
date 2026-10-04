"""Resolves whether the configured credentials allow writing HR documents.

Write access is granted only when ALL of the following hold:

1. A write credential is configured: ``PINECONE_WRITE_API_KEY`` is set, or
   ``PINECONE_ACCESS_MODE=readwrite`` (``PINECONE_API_KEY`` is then used for writes).
2. If ``PINECONE_VERIFY_WRITE_ACCESS=true``: Pinecone actually accepts a write with that key
   (a probe vector is upserted to and deleted from a dedicated probe namespace). A ReadOnly
   key is rejected with HTTP 403. Any probe failure disables writes (fail closed).
3. The local HR documents directory is writable (needed for write-back to the Markdown files).

When write access is not granted, the update tool is not registered with the agent at all, and
the update service refuses to run if called directly.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from pinecone import ForbiddenError, UnauthorizedError

from app.config import Settings
from hr_rag_ingestion import Chunk, PineconeVectorStore

logger = logging.getLogger(__name__)

PROBE_VECTOR_ID = "__hr_agent_write_probe__"


class WriteAccessDenied(PermissionError):
    """Raised when a write is attempted without write access."""


@dataclass(frozen=True)
class AccessProfile:
    """Outcome of the write-access check: whether writes are allowed, why, and which key to use."""

    write_enabled: bool
    reason: str
    write_api_key: str | None = field(default=None, repr=False)

    def require_write(self) -> str:
        """Return the write API key, or raise WriteAccessDenied if writes are not allowed."""
        if not self.write_enabled or not self.write_api_key:
            raise WriteAccessDenied(f"Write access is disabled: {self.reason}")
        return self.write_api_key


WriteProbe = Callable[[str], None]


def pinecone_write_probe(settings: Settings) -> WriteProbe:
    """Build a probe that proves ``api_key`` can write to the configured index."""

    def probe(api_key: str) -> None:
        """Upsert and immediately delete a probe vector; Pinecone raises ForbiddenError for a read-only key."""
        store = PineconeVectorStore(
            api_key=api_key,
            index_name=settings.pinecone_index_name,
            namespace=f"{settings.pinecone_namespace}__write-probe",
            host=settings.pinecone_index_host,
        )
        # Dense cosine indexes reject all-zero vectors, so use a unit vector.
        vector = [0.0] * settings.embedding_dimensions
        vector[0] = 1.0
        store.upsert_chunks([Chunk(id=PROBE_VECTOR_ID, text="probe", metadata={"probe": True})], [vector])
        store.delete_ids([PROBE_VECTOR_ID])

    return probe


def resolve_access(settings: Settings, probe: WriteProbe | None = None) -> AccessProfile:
    """Decide whether the configured credentials may write HR documents.

    Checks the configured write key, optionally proves it with a probe write, and requires a writable docs folder.
    """
    if settings.pinecone_write_api_key is not None:
        candidate = settings.pinecone_write_api_key.get_secret_value()
        source = "PINECONE_WRITE_API_KEY"
    elif settings.pinecone_access_mode == "readwrite":
        candidate = settings.pinecone_api_key.get_secret_value()
        source = "PINECONE_API_KEY (PINECONE_ACCESS_MODE=readwrite)"
    else:
        return AccessProfile(False, "the configured Pinecone credentials are read-only (PINECONE_ACCESS_MODE=read and no PINECONE_WRITE_API_KEY)")

    docs_dir = settings.hr_docs_dir
    if not docs_dir.is_dir() or not os.access(docs_dir, os.W_OK):
        return AccessProfile(False, f"the HR documents directory {docs_dir} does not exist or is not writable")

    if settings.pinecone_verify_write_access:
        try:
            (probe or pinecone_write_probe(settings))(candidate)
        except (ForbiddenError, UnauthorizedError) as exc:
            logger.warning("Write probe rejected by Pinecone: %s", exc)
            return AccessProfile(False, f"Pinecone rejected a write with {source} (the key does not have write permission)")
        except Exception as exc:  # fail closed on anything unexpected
            logger.exception("Write probe failed")
            return AccessProfile(False, f"write access for {source} could not be verified ({type(exc).__name__}: {exc})")
        return AccessProfile(True, f"write access verified for {source}", candidate)

    return AccessProfile(True, f"write access configured via {source} (not verified)", candidate)
