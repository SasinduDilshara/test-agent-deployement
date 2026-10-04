"""Settings for the ingestion CLI, loaded from environment variables / ``.env``."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class IngestionSettings(BaseSettings):
    """Ingestion configuration loaded from environment variables and rag-ingestion/.env."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Optional here so that `ingest --dry-run` works without credentials; required for any API call.
    openai_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, gt=0)

    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str = "lumora-hr-docs"
    pinecone_namespace: str = "hr-policies"
    pinecone_index_host: str | None = None
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_metric: str = "cosine"

    docs_dir: Path = Path("../agent/docs/hr")
    chunk_size: int = Field(default=1200, gt=100)
    chunk_overlap: int = Field(default=150, ge=0)

    @field_validator("docs_dir")
    @classmethod
    def _resolve_docs_dir(cls, value: Path) -> Path:
        """Resolve a relative DOCS_DIR against the rag-ingestion project directory."""
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("pinecone_index_host")
    @classmethod
    def _empty_host_is_none(cls, value: str | None) -> str | None:
        """Treat an empty PINECONE_INDEX_HOST as not set."""
        return value or None
