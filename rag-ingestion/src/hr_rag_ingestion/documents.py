"""Loading, parsing, rendering and section-editing of HR Markdown documents.

Every HR document is a Markdown file with a YAML front-matter block::

    ---
    doc_id: leave-policy
    title: Leave Policy
    category: leave
    version: "2.1"
    ...
    ---

    # Leave Policy

    ## Annual Leave
    ...

The file name must be ``<doc_id>.md`` so a document can always be located by its id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DOC_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,99}$")
REQUIRED_FIELDS = ("doc_id", "title", "category")

_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n?(.*)\Z", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]*(```|~~~)")


class DocumentError(ValueError):
    """Raised when a document is malformed or an edit cannot be applied."""


@dataclass
class HRDocument:
    """An HR policy document: id, title, category, front-matter metadata and Markdown body."""

    doc_id: str
    title: str
    category: str
    metadata: dict[str, Any]
    body: str
    source_path: Path | None = field(default=None, compare=False)

    @property
    def version(self) -> str:
        """The document version from the front matter (defaults to '1.0')."""
        return str(self.metadata.get("version", "1.0"))

    def render(self) -> str:
        """Serialise the document back to Markdown with YAML front matter."""
        front = yaml.safe_dump(self.metadata, sort_keys=False, allow_unicode=True).strip()
        return f"---\n{front}\n---\n\n{self.body.strip()}\n"

    def with_changes(self, *, body: str | None = None, **metadata_updates: Any) -> HRDocument:
        """Return a copy with a new body and/or updated front-matter fields."""
        metadata = {**self.metadata, **metadata_updates}
        return HRDocument(
            doc_id=self.doc_id,
            title=str(metadata["title"]),
            category=str(metadata["category"]),
            metadata=metadata,
            body=self.body if body is None else body,
            source_path=self.source_path,
        )


def validate_doc_id(doc_id: str) -> str:
    """Check that a doc_id uses only lowercase letters, digits and hyphens; return it unchanged."""
    if not isinstance(doc_id, str) or not DOC_ID_PATTERN.match(doc_id):
        raise DocumentError(
            f"Invalid doc_id {doc_id!r}: use lowercase letters, digits and hyphens only "
            "(e.g. 'leave-policy')."
        )
    return doc_id


def parse_document(text: str, source_path: Path | None = None) -> HRDocument:
    """Parse Markdown text with YAML front matter into an HRDocument, validating required fields."""
    text = text.replace("\r\n", "\n").lstrip("﻿")
    match = _FRONT_MATTER.match(text)
    where = f" in {source_path}" if source_path else ""
    if not match:
        raise DocumentError(f"Missing YAML front matter{where}.")
    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise DocumentError(f"Invalid YAML front matter{where}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise DocumentError(f"Front matter must be a mapping{where}.")

    missing = [name for name in REQUIRED_FIELDS if not metadata.get(name)]
    if missing:
        raise DocumentError(f"Front matter is missing required field(s) {missing}{where}.")
    # Normalise scalar values (e.g. YAML dates/numbers) to strings so they round-trip cleanly.
    metadata = {key: _normalise(value) for key, value in metadata.items()}
    doc_id = validate_doc_id(metadata["doc_id"])

    if source_path is not None and source_path.stem != doc_id:
        raise DocumentError(f"File name {source_path.name!r} must be '{doc_id}.md'.")

    return HRDocument(
        doc_id=doc_id,
        title=metadata["title"],
        category=metadata["category"],
        metadata=metadata,
        body=match.group(2).strip(),
        source_path=source_path,
    )


def _normalise(value: Any) -> Any:
    """Convert a front-matter value to a string (lists to lists of strings) so it round-trips cleanly."""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    return str(value)


def load_document(path: Path) -> HRDocument:
    """Read and parse one HR document file."""
    return parse_document(path.read_text(encoding="utf-8"), source_path=path)


def load_documents(docs_dir: Path) -> list[HRDocument]:
    """Load every ``*.md`` document (non-recursive) from ``docs_dir``, sorted by doc_id."""
    if not docs_dir.is_dir():
        raise DocumentError(f"Documents directory not found: {docs_dir}")
    documents = [load_document(path) for path in sorted(docs_dir.glob("*.md"))]
    seen: set[str] = set()
    for doc in documents:
        if doc.doc_id in seen:
            raise DocumentError(f"Duplicate doc_id {doc.doc_id!r} in {docs_dir}")
        seen.add(doc.doc_id)
    return sorted(documents, key=lambda d: d.doc_id)


# --------------------------------------------------------------------------------------
# Section handling (shared by the chunker and by the agent's update tool)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Section:
    """A run of body text under a heading path such as ``("Leave Policy", "Annual Leave")``."""

    headings: tuple[str, ...]
    content: str

    @property
    def path(self) -> str:
        """Human-readable section path, e.g. 'Annual Leave > Carry Forward'."""
        # The H1 is the document title, so it is omitted from the section path when deeper headings exist.
        parts = self.headings[1:] if len(self.headings) > 1 else self.headings
        return " > ".join(parts)


def _iter_lines_with_headings(body: str):
    """Yield ``(line, level, title)`` where level/title are set only for real headings (not in code fences)."""
    in_fence = False
    for line in body.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            yield line, None, None
            continue
        match = None if in_fence else _HEADING.match(line)
        if match:
            yield line, len(match.group(1)), match.group(2).strip()
        else:
            yield line, None, None


def split_sections(body: str, max_level: int = 3) -> list[Section]:
    """Split a Markdown body on headings up to ``max_level`` (default ``#``, ``##``, ``###``)."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        """Save the buffered lines as a section under the current heading path."""
        content = "\n".join(buffer).strip()
        if content:
            sections.append(Section(headings=tuple(title for _, title in stack), content=content))
        buffer.clear()

    for line, level, title in _iter_lines_with_headings(body):
        if level is not None and level <= max_level:
            flush()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buffer.append(line)
    flush()
    return sections


def _normalise_title(title: str) -> str:
    """Normalise a heading for case- and whitespace-insensitive comparison."""
    return re.sub(r"\s+", " ", title).strip().casefold()


def list_section_titles(body: str) -> list[str]:
    """Return the titles of all level-2 (``##``) sections in document order."""
    return [title for _, level, title in _iter_lines_with_headings(body) if level == 2]


def get_section(body: str, section_title: str) -> str | None:
    """Return the content of the ``##`` section with the given title, or ``None`` if absent."""
    lines = body.split("\n")
    span = _find_section_span(lines, section_title)
    if span is None:
        return None
    start, end = span
    return "\n".join(lines[start + 1 : end]).strip()


def _find_section_span(lines: list[str], section_title: str) -> tuple[int, int] | None:
    """Return ``(heading_index, end_index_exclusive)`` of a ``##`` section."""
    wanted = _normalise_title(section_title)
    start: int | None = None
    for index, (_, level, title) in enumerate(_iter_lines_with_headings("\n".join(lines))):
        if level is None:
            continue
        if start is None:
            if level == 2 and _normalise_title(title) == wanted:
                start = index
        elif level <= 2:
            return start, index
    return (start, len(lines)) if start is not None else None


def upsert_section(body: str, section_title: str, new_content: str) -> tuple[str, bool]:
    """Replace the content of the ``##`` section titled ``section_title`` or append a new one.

    ``new_content`` is the section body only (it may contain ``###`` sub-headings, lists,
    tables, ...). A leading ``## <section_title>`` line is tolerated and removed.

    Returns ``(new_body, created)`` where ``created`` is True when a new section was appended.
    """
    section_title = re.sub(r"\s+", " ", section_title or "").strip()
    if not section_title:
        raise DocumentError("section_title must not be empty.")
    if section_title.startswith("#"):
        raise DocumentError("section_title must be the heading text without '#' characters.")

    content = (new_content or "").replace("\r\n", "\n").strip()
    content_lines = content.split("\n")
    first = _HEADING.match(content_lines[0]) if content_lines else None
    if first and len(first.group(1)) == 2 and _normalise_title(first.group(2)) == _normalise_title(section_title):
        content = "\n".join(content_lines[1:]).strip()
    if not content:
        raise DocumentError("new_content must not be empty.")
    for _, level, title in _iter_lines_with_headings(content):
        if level is not None and level <= 2:
            raise DocumentError(
                f"new_content must not contain '#' or '##' headings (found {title!r}); "
                "use '###' for sub-headings inside a section."
            )

    lines = body.strip().split("\n")
    span = _find_section_span(lines, section_title)
    if span is None:
        new_body = body.strip() + f"\n\n## {section_title}\n\n{content}\n"
        return new_body.strip() + "\n", True

    start, end = span
    replaced = lines[: start + 1] + ["", content, ""] + lines[end:]
    new_body = "\n".join(replaced)
    new_body = re.sub(r"\n{3,}", "\n\n", new_body).strip() + "\n"
    return new_body, False
