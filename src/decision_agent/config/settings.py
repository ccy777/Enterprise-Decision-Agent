"""Environment-backed settings with no client initialization side effects."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_EMBEDDING_MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
_RERANKER_MODEL_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"


class Environment(StrEnum):
    """Supported runtime environments."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class MilvusMetricType(StrEnum):
    """Metric supported by the M2B-2A dense retrieval adapter."""

    COSINE = "COSINE"


class MilvusIndexType(StrEnum):
    """Vector index supported by the M2B-2A Milvus adapter."""

    HNSW = "HNSW"


class Settings(BaseSettings):
    """Core process settings; future integrations own their own required secrets."""

    model_config = SettingsConfigDict(
        env_prefix="DECISION_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    app_name: str = Field(min_length=1)
    environment: Environment = Environment.DEVELOPMENT
    required_dependencies: list[str] = Field(default_factory=list)
    milvus_uri: str = Field(default="http://localhost:19530", min_length=1)
    milvus_token: SecretStr | None = None
    milvus_database: str = Field(default="default", min_length=1)
    milvus_collection: str = Field(
        default="decision_agent_chunks", min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"
    )
    milvus_dimension: int = Field(default=512, gt=0)
    milvus_metric_type: MilvusMetricType = MilvusMetricType.COSINE
    milvus_index_type: MilvusIndexType = MilvusIndexType.HNSW
    hnsw_m: int = Field(default=16, gt=0)
    hnsw_ef_construction: int = Field(default=200, gt=0)
    hnsw_ef_search: int = Field(default=64, gt=0)
    milvus_timeout_seconds: float = Field(default=10.0, gt=0)
    embedding_model_name: str = Field(default="BAAI/bge-small-zh-v1.5", min_length=1)
    embedding_model_revision: str | None = _EMBEDDING_MODEL_REVISION
    embedding_dimension: int = Field(default=512, gt=0)
    embedding_device: str = Field(default="cpu", min_length=1)
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_normalize: bool = True
    embedding_query_instruction: str = Field(
        default="为这个句子生成表示以用于检索相关文章：",  # noqa: RUF001
        min_length=1,
    )
    embedding_cache_folder: str | None = None
    embedding_local_files_only: bool = False
    embedding_trust_remote_code: bool = False
    embedding_show_progress: bool = False
    knowledge_dataset_root: Path | None = None
    reranker_model_name: str = Field(default="BAAI/bge-reranker-base", min_length=1)
    reranker_model_revision: str | None = _RERANKER_MODEL_REVISION
    reranker_device: Literal["cpu"] = "cpu"
    reranker_batch_size: int = Field(default=8, gt=0)
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = Field(default=None, min_length=1)
    llm_model_name: str | None = Field(default=None, min_length=1)
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    controlled_workflow_enabled: bool = False
    mcp_timeout_seconds: float = Field(default=10.0, gt=0)
    db_host: str = Field(default="127.0.0.1", min_length=1)
    db_port: int = Field(default=3306, gt=0, le=65535)
    db_database: str = Field(default="enterprise_operations", min_length=1)
    db_readonly_username: str = Field(default="decision_agent_readonly", min_length=1)
    db_readonly_password: SecretStr | None = None
    db_connect_timeout_seconds: int = Field(default=5, gt=0)
    db_query_timeout_seconds: float = Field(default=5.0, gt=0)
    db_max_rows: int = Field(default=200, gt=0, le=10_000)
    db_max_result_cells: int = Field(default=2_000, gt=0, le=100_000)
    memory_mode: Literal["disabled", "in_memory", "redis"] = "disabled"
    memory_redis_url: SecretStr | None = None
    memory_redis_timeout_seconds: float = Field(default=5.0, gt=0)
    memory_ttl_seconds: int = Field(default=1800, gt=0)
    memory_max_turns: int = Field(default=20, gt=0)
    memory_summary_enabled: bool = False
    memory_summary_trigger_turns: int = Field(default=6, gt=0)
    memory_summary_retain_recent_turns: int = Field(default=2, gt=0)
    memory_summary_max_source_chars: int = Field(default=8_000, gt=0)
    memory_summary_max_summary_chars: int = Field(default=2_000, gt=0)
    audit_log_path: Path | None = None

    @field_validator(
        "milvus_token",
        "embedding_cache_folder",
        "knowledge_dataset_root",
        "llm_api_key",
        "llm_base_url",
        "llm_model_name",
        "db_readonly_password",
        "memory_redis_url",
        "audit_log_path",
        mode="before",
    )
    @classmethod
    def empty_optional_value_to_none(cls, value: Any) -> Any:
        """Treat blank optional environment values as unset."""
        return None if value == "" else value

    @field_validator("embedding_model_revision", mode="before")
    @classmethod
    def blank_embedding_revision_uses_fixed_default(cls, value: Any) -> Any:
        """Keep an empty environment override from restoring Hugging Face's floating default."""
        return _EMBEDDING_MODEL_REVISION if value == "" else value

    @field_validator("reranker_model_revision", mode="before")
    @classmethod
    def blank_reranker_revision_uses_fixed_default(cls, value: Any) -> Any:
        """Keep an empty environment override from restoring Hugging Face's floating default."""
        return _RERANKER_MODEL_REVISION if value == "" else value

    @field_validator("embedding_trust_remote_code")
    @classmethod
    def reject_embedding_remote_code(cls, value: bool) -> bool:
        """Prevent model repositories from executing custom remote code."""
        if value:
            raise ValueError("embedding trust_remote_code must remain false")
        return value

    @field_validator("reranker_model_name")
    @classmethod
    def reject_blank_reranker_model_name(cls, value: str) -> str:
        """Reject whitespace-only model names without normalizing valid identifiers."""
        if not value.strip():
            raise ValueError("reranker model name cannot be blank")
        return value

    @field_validator("milvus_uri")
    @classmethod
    def validate_milvus_uri(cls, value: str) -> str:
        """Reject connection URLs that omit a scheme or embed credentials."""
        cls._validate_url_structure(value, field_name="milvus_uri")
        return value

    @field_validator("llm_base_url")
    @classmethod
    def validate_llm_base_url(cls, value: str | None) -> str | None:
        """Validate a configured LLM endpoint without making a request."""
        if value is not None:
            cls._validate_url_structure(value, field_name="llm_base_url")
        return value

    @field_validator("memory_redis_url")
    @classmethod
    def validate_memory_redis_url(cls, value: SecretStr | None) -> SecretStr | None:
        """Inspect the Redis secret only inside validation and never echo it."""
        if value is not None:
            cls._validate_url_structure(
                value.get_secret_value(),
                field_name="memory_redis_url",
            )
        return value

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> Settings:
        """Fail fast on cross-field configuration contradictions only."""
        llm_values = (self.llm_api_key, self.llm_base_url, self.llm_model_name)
        if any(value is not None for value in llm_values) and not all(
            value is not None for value in llm_values
        ):
            raise ValueError("LLM configuration must provide all three fields or none")
        if self.embedding_dimension != self.milvus_dimension:
            raise ValueError("embedding and Milvus dimensions must match")
        if self.memory_mode == "redis":
            if self.memory_redis_url is None:
                raise ValueError("redis memory mode requires memory_redis_url")
        elif self.memory_redis_url is not None:
            raise ValueError("memory_redis_url is only valid in redis memory mode")
        if self.memory_summary_enabled and self.memory_mode == "disabled":
            raise ValueError("memory summary requires an enabled memory mode")
        if self.memory_summary_retain_recent_turns >= self.memory_summary_trigger_turns:
            raise ValueError("summary retained turns must be less than trigger turns")
        if self.memory_summary_trigger_turns > self.memory_max_turns:
            raise ValueError("summary trigger turns cannot exceed memory_max_turns")
        return self

    @staticmethod
    def _validate_url_structure(value: str, *, field_name: str) -> None:
        try:
            parsed = urlsplit(value)
            has_scheme_and_host = bool(parsed.scheme and parsed.hostname)
        except ValueError as exc:
            raise ValueError(f"{field_name} has an invalid URL structure") from exc
        if not has_scheme_and_host:
            raise ValueError(f"{field_name} must include a scheme and host")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(f"{field_name} must not embed user information")
