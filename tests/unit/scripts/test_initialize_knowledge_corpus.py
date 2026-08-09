"""Public release tests for the formal knowledge-corpus initialization command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.initialize_knowledge_corpus as command

from decision_agent.config import Settings


class _Pipeline:
    def __init__(
        self,
        *,
        parent_count: int = 12,
        child_count: int = 101,
        ingestion: object | None = None,
    ) -> None:
        self.parent_count = parent_count
        self.child_count = child_count
        self.last_ingestion_result = ingestion or SimpleNamespace(
            attempted_count=child_count,
            inserted_count=child_count,
            updated_count=0,
        )


class _Runtime:
    def __init__(
        self,
        *,
        pipeline: _Pipeline | None = None,
        initialize_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.pipeline = pipeline or _Pipeline()
        self.initialize_error = initialize_error
        self.close_error = close_error
        self.initialize_calls = 0
        self.close_calls = 0

    async def initialize_for_ingestion(self) -> None:
        self.initialize_calls += 1
        if self.initialize_error is not None:
            raise self.initialize_error

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _settings(dataset_root: Path) -> Settings:
    return Settings(
        app_name="Public Test Agent",
        knowledge_dataset_root=dataset_root,
        _env_file=None,
    )


@pytest.mark.asyncio
async def test_success_uses_formal_ingestion_and_closes_runtime(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    runtime = _Runtime()

    result = await command._run(
        argparse.Namespace(dataset_root=None),
        settings_factory=lambda: _settings(dataset),
        runtime_factory=lambda settings: runtime,  # type: ignore[return-value]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert runtime.initialize_calls == runtime.close_calls == 1
    assert payload == {
        "child_count": 101,
        "inserted_count": 101,
        "parent_count": 12,
        "processed_count": 101,
        "status": "initialized",
        "updated_count": 0,
    }


@pytest.mark.asyncio
async def test_missing_dataset_fails_before_runtime_construction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    constructed = False

    def runtime_factory(settings: Settings) -> _Runtime:
        nonlocal constructed
        constructed = True
        return _Runtime()

    result = await command._run(
        argparse.Namespace(dataset_root=tmp_path / "missing"),
        settings_factory=lambda: _settings(tmp_path),
        runtime_factory=runtime_factory,  # type: ignore[arg-type]
    )

    assert result == 2
    assert constructed is False
    assert json.loads(capsys.readouterr().err) == {
        "error_code": "knowledge_dataset_missing",
        "status": "failed",
    }


@pytest.mark.asyncio
async def test_ingestion_exception_is_redacted_and_runtime_is_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    runtime = _Runtime(initialize_error=RuntimeError("api_key=must-not-leak"))

    result = await command._run(
        argparse.Namespace(dataset_root=None),
        settings_factory=lambda: _settings(dataset),
        runtime_factory=lambda settings: runtime,  # type: ignore[return-value]
    )

    output = capsys.readouterr().err
    assert result == 2
    assert runtime.close_calls == 1
    assert json.loads(output) == {
        "error_code": "knowledge_corpus_initialization_failed",
        "status": "failed",
    }
    assert "api_key" not in output


@pytest.mark.asyncio
async def test_invalid_settings_fail_closed_without_runtime(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def invalid_settings() -> Settings:
        raise ValueError("secret configuration body")

    result = await command._run(
        argparse.Namespace(dataset_root=None),
        settings_factory=invalid_settings,
        runtime_factory=lambda settings: _Runtime(),  # type: ignore[arg-type,return-value]
    )

    output = capsys.readouterr().err
    assert result == 2
    assert json.loads(output) == {"error_code": "configuration_invalid", "status": "failed"}
    assert "secret configuration" not in output


@pytest.mark.asyncio
async def test_incomplete_ingestion_result_fails_verification(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    runtime = _Runtime(
        pipeline=_Pipeline(
            child_count=101,
            ingestion=SimpleNamespace(
                attempted_count=100,
                inserted_count=100,
                updated_count=0,
            ),
        )
    )

    result = await command._run(
        argparse.Namespace(dataset_root=None),
        settings_factory=lambda: _settings(dataset),
        runtime_factory=lambda settings: runtime,  # type: ignore[return-value]
    )

    assert result == 2
    assert runtime.close_calls == 1
    assert json.loads(capsys.readouterr().err) == {
        "error_code": "knowledge_corpus_verification_failed",
        "status": "failed",
    }


@pytest.mark.asyncio
async def test_cleanup_failure_replaces_success_with_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    runtime = _Runtime(close_error=RuntimeError("close detail"))

    result = await command._run(
        argparse.Namespace(dataset_root=None),
        settings_factory=lambda: _settings(dataset),
        runtime_factory=lambda settings: runtime,  # type: ignore[return-value]
    )

    assert result == 2
    assert runtime.initialize_calls == runtime.close_calls == 1
    assert json.loads(capsys.readouterr().err) == {
        "error_code": "resource_cleanup_failed",
        "status": "failed",
    }


def test_command_reuses_formal_runtime_without_reimplementing_retrieval() -> None:
    source = Path(command.__file__).read_text(encoding="utf-8")

    assert "initialize_for_ingestion()" in source
    for forbidden in (
        "ParentChildChunker",
        "SentenceTransformerEmbeddingProvider",
        "MilvusVectorStore",
        "create_schema",
        "embed_documents",
        "fixed-window-v1",
    ):
        assert forbidden not in source


def test_public_environment_example_closes_local_bootstrap_configuration() -> None:
    example = Path(".env.example").read_text(encoding="utf-8")

    for required in (
        "MYSQL_ROOT_PASSWORD=change-me-for-local-demo",
        "DECISION_AGENT_DB_READONLY_PASSWORD=change-me-for-local-demo-readonly",
        "DECISION_AGENT_DB_HOST=127.0.0.1",
        "DECISION_AGENT_MILVUS_URI=http://127.0.0.1:19530",
        "DECISION_AGENT_KNOWLEDGE_DATASET_ROOT=./datasets/enterprise_kb/m2c1",
    ):
        assert required in example
