"""Opt-in integration lifecycle for a real, explicitly provisioned Milvus."""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from pymilvus import MilvusClient

from decision_agent.config import Settings
from decision_agent.retrieval import (
    MilvusVectorStore,
    VectorRecord,
    VectorSearchFilter,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_MILVUS_INTEGRATION") != "1",
        reason="set RUN_MILVUS_INTEGRATION=1 to use a provisioned Milvus service",
    ),
]


@pytest.mark.asyncio
async def test_real_milvus_lifecycle() -> None:
    """Create an isolated collection and exercise CRUD plus filtered search."""
    uri = os.getenv("DECISION_AGENT_MILVUS_URI")
    if not uri:
        pytest.skip("DECISION_AGENT_MILVUS_URI must identify the provisioned test service")
    settings = Settings(
        app_name="Milvus Integration",
        milvus_uri=uri,
        milvus_token=os.getenv("DECISION_AGENT_MILVUS_TOKEN") or None,
        milvus_database=os.getenv("DECISION_AGENT_MILVUS_DATABASE", "default"),
        _env_file=None,
    )
    collection_name = f"decision_agent_it_{uuid.uuid4().hex}"
    client = MilvusClient(
        uri=settings.milvus_uri,
        token=settings.milvus_token or "",
        db_name=settings.milvus_database,
        timeout=settings.milvus_timeout_seconds,
    )
    store = MilvusVectorStore(
        dimension=2,
        uri=settings.milvus_uri,
        token=settings.milvus_token,
        database=settings.milvus_database,
        collection_name=collection_name,
        timeout_seconds=settings.milvus_timeout_seconds,
        client=client,
    )
    original_error: BaseException | None = None
    try:
        await store.initialize()
        records = [
            VectorRecord(
                record_id="it-a",
                parent_id="parent-a",
                document_id="doc-a",
                document_version="v1",
                content="east sales decline",
                vector=[1.0, 0.0],
                metadata={"kind": "integration"},
            ),
            VectorRecord(
                record_id="it-b",
                parent_id="parent-b",
                document_id="doc-b",
                document_version="v1",
                content="employee policy",
                vector=[0.0, 1.0],
                metadata={"kind": "integration"},
            ),
        ]
        inserted = await store.upsert(records)
        assert inserted.inserted_count == 2
        assert await store.count() == 2
        hits = await store.search([1.0, 0.0], 2, VectorSearchFilter(document_id="doc-a"))
        assert [hit.record_id for hit in hits] == ["it-a"]
        assert await store.delete_by_document("doc-a") == 1
    except BaseException as exc:
        original_error = exc
        raise
    finally:
        cleanup_errors: list[Exception] = []
        try:
            await asyncio.to_thread(
                client.drop_collection,
                collection_name=collection_name,
                timeout=settings.milvus_timeout_seconds,
            )
        except Exception as exc:
            cleanup_errors.append(exc)
        try:
            await store.close()
        except Exception as exc:
            cleanup_errors.append(exc)
        if cleanup_errors and original_error is not None:
            for cleanup_error in cleanup_errors:
                original_error.add_note(f"Milvus integration cleanup failed: {cleanup_error!r}")
        elif cleanup_errors:
            first_error = cleanup_errors[0]
            for cleanup_error in cleanup_errors[1:]:
                first_error.add_note(f"Additional cleanup failure: {cleanup_error!r}")
            raise first_error
