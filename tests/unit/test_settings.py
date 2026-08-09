"""Offline tests for explicit environment configuration."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from decision_agent.config import Environment, MilvusIndexType, MilvusMetricType, Settings


@pytest.fixture(autouse=True)
def isolate_decision_agent_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore every Settings-owned environment value after each test."""
    for field_name in Settings.model_fields:
        monkeypatch.delenv(f"DECISION_AGENT_{field_name.upper()}", raising=False)


def test_settings_fail_clearly_when_app_name_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DECISION_AGENT_APP_NAME", raising=False)

    with pytest.raises(ValidationError, match="app_name"):
        Settings(_env_file=None)


def test_settings_load_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")
    monkeypatch.setenv("DECISION_AGENT_ENVIRONMENT", "test")
    monkeypatch.setenv("DECISION_AGENT_REQUIRED_DEPENDENCIES", '["vector-store"]')

    settings = Settings(_env_file=None)

    assert settings.app_name == "Test Agent"
    assert settings.environment is Environment.TEST
    assert settings.required_dependencies == ["vector-store"]


def test_milvus_settings_have_safe_local_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")

    settings = Settings(_env_file=None)

    assert settings.milvus_uri == "http://localhost:19530"
    assert settings.milvus_token is None
    assert settings.milvus_database == "default"
    assert settings.milvus_collection == "decision_agent_chunks"
    assert settings.milvus_dimension == 512
    assert settings.milvus_metric_type is MilvusMetricType.COSINE
    assert settings.milvus_index_type is MilvusIndexType.HNSW


def test_milvus_settings_load_environment_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")
    monkeypatch.setenv("DECISION_AGENT_MILVUS_URI", "https://milvus.example.invalid")
    monkeypatch.setenv("DECISION_AGENT_MILVUS_TOKEN", "test-token")
    monkeypatch.setenv("DECISION_AGENT_MILVUS_DATABASE", "analytics")
    monkeypatch.setenv("DECISION_AGENT_MILVUS_COLLECTION", "knowledge_chunks")
    monkeypatch.setenv("DECISION_AGENT_MILVUS_DIMENSION", "256")
    monkeypatch.setenv("DECISION_AGENT_EMBEDDING_DIMENSION", "256")
    monkeypatch.setenv("DECISION_AGENT_HNSW_M", "32")
    monkeypatch.setenv("DECISION_AGENT_HNSW_EF_CONSTRUCTION", "400")
    monkeypatch.setenv("DECISION_AGENT_HNSW_EF_SEARCH", "96")
    monkeypatch.setenv("DECISION_AGENT_MILVUS_TIMEOUT_SECONDS", "12.5")

    settings = Settings(_env_file=None)

    assert settings.milvus_uri == "https://milvus.example.invalid"
    assert settings.milvus_token is not None
    assert settings.milvus_token.get_secret_value() == "test-token"
    assert settings.milvus_database == "analytics"
    assert settings.milvus_collection == "knowledge_chunks"
    assert settings.milvus_dimension == 256
    assert settings.hnsw_m == 32
    assert settings.hnsw_ef_construction == 400
    assert settings.hnsw_ef_search == 96
    assert settings.milvus_timeout_seconds == 12.5


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DECISION_AGENT_MILVUS_DIMENSION", "0"),
        ("DECISION_AGENT_HNSW_M", "0"),
        ("DECISION_AGENT_HNSW_EF_CONSTRUCTION", "-1"),
        ("DECISION_AGENT_HNSW_EF_SEARCH", "0"),
        ("DECISION_AGENT_MILVUS_TIMEOUT_SECONDS", "0"),
    ],
)
def test_milvus_positive_settings_reject_non_positive_values(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_milvus_metric_rejects_unsupported_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")
    monkeypatch.setenv("DECISION_AGENT_MILVUS_METRIC_TYPE", "L2")

    with pytest.raises(ValidationError, match="milvus_metric_type"):
        Settings(_env_file=None)


def test_milvus_index_rejects_unsupported_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")
    monkeypatch.setenv("DECISION_AGENT_MILVUS_INDEX_TYPE", "IVF_FLAT")

    with pytest.raises(ValidationError, match="milvus_index_type"):
        Settings(_env_file=None)


def test_embedding_settings_have_bge_cpu_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")

    settings = Settings(_env_file=None)

    assert settings.embedding_model_name == "BAAI/bge-small-zh-v1.5"
    assert settings.embedding_model_revision == "7999e1d3359715c523056ef9478215996d62a620"
    assert settings.embedding_dimension == 512
    assert settings.embedding_device == "cpu"
    assert settings.embedding_normalize is True
    assert settings.embedding_trust_remote_code is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DECISION_AGENT_EMBEDDING_DIMENSION", "0"),
        ("DECISION_AGENT_EMBEDDING_BATCH_SIZE", "0"),
    ],
)
def test_embedding_positive_settings_reject_non_positive_values(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_embedding_trust_remote_code_true_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")
    monkeypatch.setenv("DECISION_AGENT_EMBEDDING_TRUST_REMOTE_CODE", "true")

    with pytest.raises(ValidationError, match="trust_remote_code"):
        Settings(_env_file=None)


def test_embedding_settings_load_environment_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")
    monkeypatch.setenv("DECISION_AGENT_EMBEDDING_MODEL_NAME", "local/test-model")
    monkeypatch.setenv("DECISION_AGENT_EMBEDDING_MODEL_REVISION", "revision-1")
    monkeypatch.setenv("DECISION_AGENT_EMBEDDING_DIMENSION", "256")
    monkeypatch.setenv("DECISION_AGENT_MILVUS_DIMENSION", "256")
    monkeypatch.setenv("DECISION_AGENT_EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("DECISION_AGENT_EMBEDDING_BATCH_SIZE", "7")
    monkeypatch.setenv("DECISION_AGENT_EMBEDDING_NORMALIZE", "false")
    monkeypatch.setenv("DECISION_AGENT_EMBEDDING_QUERY_INSTRUCTION", "query: ")
    monkeypatch.setenv("DECISION_AGENT_EMBEDDING_CACHE_FOLDER", "models/cache")
    monkeypatch.setenv("DECISION_AGENT_EMBEDDING_LOCAL_FILES_ONLY", "true")
    monkeypatch.setenv("DECISION_AGENT_EMBEDDING_SHOW_PROGRESS", "true")

    settings = Settings(_env_file=None)

    assert settings.embedding_model_name == "local/test-model"
    assert settings.embedding_model_revision == "revision-1"
    assert settings.embedding_dimension == 256
    assert settings.embedding_batch_size == 7
    assert settings.embedding_normalize is False
    assert settings.embedding_query_instruction == "query: "
    assert settings.embedding_cache_folder == "models/cache"
    assert settings.embedding_local_files_only is True
    assert settings.embedding_show_progress is True


def test_milvus_and_embedding_dimensions_default_to_same_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")

    settings = Settings(_env_file=None)

    assert settings.embedding_dimension == settings.milvus_dimension == 512


def test_empty_embedding_revision_environment_value_uses_fixed_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")
    monkeypatch.setenv("DECISION_AGENT_EMBEDDING_MODEL_REVISION", "")
    monkeypatch.setenv("DECISION_AGENT_EMBEDDING_CACHE_FOLDER", "")

    settings = Settings(_env_file=None)

    assert settings.embedding_model_revision == "7999e1d3359715c523056ef9478215996d62a620"
    assert settings.embedding_cache_folder is None


def test_empty_optional_milvus_token_becomes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")
    monkeypatch.setenv("DECISION_AGENT_MILVUS_TOKEN", "")

    settings = Settings(_env_file=None)

    assert settings.milvus_token is None


def test_llm_settings_load_without_exposing_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")
    monkeypatch.setenv("DECISION_AGENT_LLM_API_KEY", "test-secret")
    monkeypatch.setenv("DECISION_AGENT_LLM_BASE_URL", "https://llm.example.invalid/v1")
    monkeypatch.setenv("DECISION_AGENT_LLM_MODEL_NAME", "test-model")
    monkeypatch.setenv("DECISION_AGENT_LLM_TIMEOUT_SECONDS", "12.5")

    settings = Settings(_env_file=None)

    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "test-secret"
    assert settings.llm_base_url == "https://llm.example.invalid/v1"
    assert settings.llm_model_name == "test-model"
    assert settings.llm_timeout_seconds == 12.5


def test_database_settings_load_readonly_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")
    monkeypatch.setenv("DECISION_AGENT_DB_HOST", "mysql.example.invalid")
    monkeypatch.setenv("DECISION_AGENT_DB_PORT", "3307")
    monkeypatch.setenv("DECISION_AGENT_DB_DATABASE", "operations")
    monkeypatch.setenv("DECISION_AGENT_DB_READONLY_USERNAME", "readonly")
    monkeypatch.setenv("DECISION_AGENT_DB_READONLY_PASSWORD", "test-secret")
    monkeypatch.setenv("DECISION_AGENT_DB_CONNECT_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("DECISION_AGENT_DB_QUERY_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("DECISION_AGENT_DB_MAX_ROWS", "42")
    monkeypatch.setenv("DECISION_AGENT_DB_MAX_RESULT_CELLS", "420")

    settings = Settings(_env_file=None)

    assert settings.db_host == "mysql.example.invalid"
    assert settings.db_port == 3307
    assert settings.db_database == "operations"
    assert settings.db_readonly_username == "readonly"
    assert settings.db_readonly_password is not None
    assert settings.db_readonly_password.get_secret_value() == "test-secret"
    assert settings.db_connect_timeout_seconds == 7
    assert settings.db_query_timeout_seconds == 2.5
    assert settings.db_max_rows == 42
    assert settings.db_max_result_cells == 420


def make_settings(**overrides: Any) -> Settings:
    return Settings(app_name="Test Agent", _env_file=None, **overrides)


def test_production_contract_defaults_preserve_health_only_configuration() -> None:
    settings = make_settings()

    assert settings.knowledge_dataset_root is None
    assert settings.reranker_model_name == "BAAI/bge-reranker-base"
    assert settings.reranker_model_revision == "2cfc18c9415c912f9d8155881c133215df768a70"
    assert settings.reranker_device == "cpu"
    assert settings.reranker_batch_size == 8
    assert settings.mcp_timeout_seconds == 10.0
    assert settings.memory_mode == "disabled"
    assert settings.memory_redis_url is None
    assert settings.memory_redis_timeout_seconds == 5.0
    assert settings.memory_ttl_seconds == 1800
    assert settings.memory_max_turns == 20
    assert settings.memory_summary_enabled is False
    assert settings.memory_summary_trigger_turns == 6
    assert settings.memory_summary_retain_recent_turns == 2
    assert settings.memory_summary_max_source_chars == 8_000
    assert settings.memory_summary_max_summary_chars == 2_000
    assert settings.llm_api_key is settings.llm_base_url is settings.llm_model_name is None
    assert settings.db_readonly_password is None


@pytest.mark.parametrize(
    "model_name",
    ["   ", "\t\r\n"],
    ids=["spaces", "control-whitespace"],
)
def test_reranker_model_name_rejects_whitespace_only(model_name: str) -> None:
    with pytest.raises(ValidationError, match="reranker model name cannot be blank"):
        make_settings(reranker_model_name=model_name)


def test_production_contract_loads_prefixed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "APP_NAME": "Production Agent",
        "KNOWLEDGE_DATASET_ROOT": "data/production.jsonl",
        "RERANKER_MODEL_NAME": "local/reranker",
        "RERANKER_MODEL_REVISION": "revision-2",
        "RERANKER_DEVICE": "cpu",
        "RERANKER_BATCH_SIZE": "4",
        "MCP_TIMEOUT_SECONDS": "7.5",
        "MEMORY_MODE": "redis",
        "MEMORY_REDIS_URL": "redis://localhost:6379/0",
        "MEMORY_REDIS_TIMEOUT_SECONDS": "2.5",
        "MEMORY_TTL_SECONDS": "900",
        "MEMORY_MAX_TURNS": "12",
        "MEMORY_SUMMARY_ENABLED": "true",
        "MEMORY_SUMMARY_TRIGGER_TURNS": "5",
        "MEMORY_SUMMARY_RETAIN_RECENT_TURNS": "2",
        "MEMORY_SUMMARY_MAX_SOURCE_CHARS": "4000",
        "MEMORY_SUMMARY_MAX_SUMMARY_CHARS": "1000",
    }
    for name, value in values.items():
        monkeypatch.setenv(f"DECISION_AGENT_{name}", value)

    settings = Settings(_env_file=None)

    assert settings.knowledge_dataset_root == Path("data/production.jsonl")
    assert settings.reranker_model_name == "local/reranker"
    assert settings.reranker_model_revision == "revision-2"
    assert settings.reranker_batch_size == 4
    assert settings.mcp_timeout_seconds == 7.5
    assert settings.memory_mode == "redis"
    assert settings.memory_redis_url is not None
    assert settings.memory_redis_url.get_secret_value() == "redis://localhost:6379/0"
    assert settings.memory_redis_timeout_seconds == 2.5
    assert settings.memory_ttl_seconds == 900
    assert settings.memory_max_turns == 12
    assert settings.memory_summary_enabled is True
    assert settings.memory_summary_trigger_turns == 5
    assert settings.memory_summary_retain_recent_turns == 2
    assert settings.memory_summary_max_source_chars == 4_000
    assert settings.memory_summary_max_summary_chars == 1_000


def test_empty_reranker_revision_environment_value_uses_fixed_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECISION_AGENT_APP_NAME", "Test Agent")
    monkeypatch.setenv("DECISION_AGENT_KNOWLEDGE_DATASET_ROOT", "")
    monkeypatch.setenv("DECISION_AGENT_RERANKER_MODEL_REVISION", "")
    monkeypatch.setenv("DECISION_AGENT_MEMORY_REDIS_URL", "")

    settings = Settings(_env_file=None)

    assert settings.knowledge_dataset_root is None
    assert settings.reranker_model_revision == "2cfc18c9415c912f9d8155881c133215df768a70"
    assert settings.memory_redis_url is None


def test_secret_fields_are_typed_and_masked_in_settings_repr() -> None:
    marker = "settings-secret-marker"
    settings = make_settings(
        milvus_token=marker,
        memory_mode="redis",
        memory_redis_url=f"redis://localhost:6379/{marker}",
    )

    assert isinstance(settings.milvus_token, SecretStr)
    assert isinstance(settings.memory_redis_url, SecretStr)
    assert marker not in repr(settings)


@pytest.mark.parametrize(
    "partial",
    [
        {"llm_api_key": "key"},
        {"llm_base_url": "https://llm.example.invalid/v1"},
        {"llm_model_name": "model"},
        {"llm_api_key": "key", "llm_base_url": "https://llm.example.invalid/v1"},
        {"llm_api_key": "key", "llm_model_name": "model"},
        {"llm_base_url": "https://llm.example.invalid/v1", "llm_model_name": "model"},
    ],
)
def test_incomplete_llm_trio_is_rejected(partial: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="all three fields or none"):
        make_settings(**partial)


def test_complete_llm_trio_is_valid() -> None:
    settings = make_settings(
        llm_api_key="key",
        llm_base_url="https://llm.example.invalid/v1",
        llm_model_name="model",
    )

    assert settings.llm_api_key is not None
    assert settings.llm_model_name == "model"


def test_mismatched_embedding_and_milvus_dimensions_are_rejected() -> None:
    with pytest.raises(ValidationError, match="dimensions must match"):
        make_settings(embedding_dimension=256, milvus_dimension=512)


@pytest.mark.parametrize(
    ("mode", "redis_url"),
    [
        ("disabled", None),
        ("in_memory", None),
        ("redis", "redis://localhost:6379/0"),
    ],
)
def test_memory_modes_accept_only_matching_redis_configuration(
    mode: str,
    redis_url: str | None,
) -> None:
    settings = make_settings(memory_mode=mode, memory_redis_url=redis_url)

    assert settings.memory_mode == mode


@pytest.mark.parametrize(
    ("mode", "redis_url", "message"),
    [
        ("redis", None, "requires memory_redis_url"),
        ("disabled", "redis://localhost:6379/0", "only valid"),
        ("in_memory", "redis://localhost:6379/0", "only valid"),
    ],
)
def test_memory_modes_reject_conflicting_redis_configuration(
    mode: str,
    redis_url: str | None,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        make_settings(memory_mode=mode, memory_redis_url=redis_url)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"memory_mode": "disabled", "memory_summary_enabled": True},
            "enabled memory mode",
        ),
        (
            {
                "memory_mode": "in_memory",
                "memory_summary_enabled": True,
                "memory_summary_retain_recent_turns": 6,
            },
            "less than trigger",
        ),
        (
            {
                "memory_mode": "in_memory",
                "memory_summary_enabled": True,
                "memory_summary_trigger_turns": 21,
            },
            "cannot exceed",
        ),
    ],
)
def test_invalid_summary_combinations_are_rejected(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        make_settings(**overrides)


@pytest.mark.parametrize(
    "field_name",
    [
        "reranker_batch_size",
        "mcp_timeout_seconds",
        "memory_redis_timeout_seconds",
        "memory_ttl_seconds",
        "memory_max_turns",
        "memory_summary_trigger_turns",
        "memory_summary_retain_recent_turns",
        "memory_summary_max_source_chars",
        "memory_summary_max_summary_chars",
    ],
)
def test_new_positive_numeric_settings_reject_zero(field_name: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{field_name: 0})


def test_invalid_memory_mode_and_reranker_device_are_rejected() -> None:
    with pytest.raises(ValidationError, match="memory_mode"):
        make_settings(memory_mode="fallback")
    with pytest.raises(ValidationError, match="reranker_device"):
        make_settings(reranker_device="cuda")


@pytest.mark.parametrize(
    ("overrides", "field_name"),
    [
        ({"milvus_uri": "localhost:19530"}, "milvus_uri"),
        (
            {
                "llm_api_key": "key",
                "llm_base_url": "https://user:password@llm.example.invalid/v1",
                "llm_model_name": "model",
            },
            "llm_base_url",
        ),
        (
            {
                "memory_mode": "redis",
                "memory_redis_url": "redis://user:password@localhost:6379/0",
            },
            "memory_redis_url",
        ),
    ],
)
def test_connection_urls_reject_missing_scheme_or_embedded_user_info(
    overrides: dict[str, Any],
    field_name: str,
) -> None:
    with pytest.raises(ValidationError, match=field_name):
        make_settings(**overrides)


def test_validation_errors_never_echo_secret_inputs() -> None:
    marker = "validation-secret-marker"

    with pytest.raises(ValidationError) as raised:
        make_settings(
            milvus_token=marker,
            memory_mode="redis",
            memory_redis_url=f"redis://{marker}@localhost:6379/0",
        )

    assert marker not in str(raised.value)
    assert marker not in repr(raised.value)


def test_settings_does_not_access_dataset_or_external_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_accessed(*_args: Any, **_kwargs: Any) -> bool:
        raise AssertionError("Settings must not access the filesystem or an external service")

    monkeypatch.setattr(Path, "exists", fail_if_accessed)
    root = Path("definitely/not/present/knowledge.jsonl")

    settings = make_settings(knowledge_dataset_root=root)

    assert settings.knowledge_dataset_root == root
