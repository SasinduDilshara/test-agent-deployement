"""Agent settings, loaded from environment variables / ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Agent configuration loaded from environment variables and agent/.env."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: SecretStr
    openai_chat_model: str = "gpt-5-mini"
    openai_reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = "low"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, gt=0)

    # Pinecone
    pinecone_api_key: SecretStr
    pinecone_index_name: str = "lumora-hr-docs"
    pinecone_namespace: str = "hr-policies"
    pinecone_index_host: str | None = None
    pinecone_access_mode: Literal["read", "readwrite"] = "read"
    pinecone_write_api_key: SecretStr | None = None
    pinecone_verify_write_access: bool = True

    # Documents
    hr_docs_dir: Path = Path("docs/hr")  # bundled with the agent
    company_profile_path: Path = Path("docs/company/company_profile.json")
    chunk_size: int = Field(default=1200, gt=100)
    chunk_overlap: int = Field(default=150, ge=0)
    audit_log_path: Path = Path("./logs/document_updates.jsonl")

    # Agent
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    max_query_rewrites: int = Field(default=2, ge=0, le=5)
    graph_recursion_limit: int = Field(default=30, ge=5)
    log_level: str = "INFO"

    @field_validator("hr_docs_dir", "company_profile_path", "audit_log_path")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        """Resolve relative paths against the agent project directory."""
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("pinecone_index_host", "openai_reasoning_effort", mode="before")
    @classmethod
    def _empty_is_none(cls, value):
        """Treat empty values from .env as 'not set'."""
        return value or None

    @field_validator("pinecone_write_api_key", mode="before")
    @classmethod
    def _empty_secret_is_none(cls, value):
        """Treat an empty PINECONE_WRITE_API_KEY as 'no separate write key'."""
        return value or None


@lru_cache
def get_settings() -> Settings:
    """Return the settings, loaded once and cached for the lifetime of the process."""
    return Settings()
