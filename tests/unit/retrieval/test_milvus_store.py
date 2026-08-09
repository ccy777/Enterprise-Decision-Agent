"""Offline contract tests for the direct pymilvus vector-store adapter."""

from __future__ import annotations

import copy
import math
import threading
import traceback
from typing import Any

import pytest
from pydantic import SecretStr
from pymilvus import DataType, MilvusClient

from decision_agent.config import Settings
from decision_agent.domain import ChildChunk
from decision_agent.exceptions import (
    RetrievalValidationError,
    VectorStoreConnectionError,
    VectorStoreOperationError,
    VectorStoreSchemaError,
)
from decision_agent.retrieval import (
    DenseIndexer,
    DenseRetriever,
    DeterministicHashEmbeddingProvider,
    MilvusFieldLimits,
    MilvusVectorStore,
    VectorRecord,
    VectorSearchFilter,
    VectorStore,
)


@pytest.fixture(autouse=True)
def isolate_decision_agent_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore every Settings-owned environment value after each test."""
    for field_name in Settings.model_fields:
        monkeypatch.delenv(f"DECISION_AGENT_{field_name.upper()}", raising=False)


class FakeSchema:
    def __init__(self, **kwargs: Any) -> None:
        self.options = kwargs
        self.fields: list[dict[str, Any]] = []

    def add_field(self, *, field_name: str, datatype: DataType, **kwargs: Any) -> None:
        self.fields.append({"name": field_name, "datatype": datatype, **kwargs})


class FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict[str, Any]] = []

    def add_index(self, **kwargs: Any) -> None:
        self.indexes.append(copy.deepcopy(kwargs))


class FakeQueryIterator:
    def __init__(self, client: FakeMilvusClient, batches: list[Any]) -> None:
        self._client = client
        self._batches = copy.deepcopy(batches)
        self.next_calls = 0
        self.closed = False

    def next(self) -> Any:
        self._client._call("query_iterator_next")
        self.next_calls += 1
        if self._batches:
            return self._batches.pop(0)
        return []

    def close(self) -> None:
        self._client._call("query_iterator_close")
        self.closed = True


class FakeMilvusClient:
    def __init__(self, *, collection_exists: bool = False, dimension: int = 2) -> None:
        self.collection_exists = collection_exists
        self.dimension = dimension
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.call_threads: list[int] = []
        self.fail_methods: set[str] = set()
        self.rows: dict[str, dict[str, Any]] = {}
        self.search_results: list[list[dict[str, Any]]] | None = None
        self.query_response: list[dict[str, Any]] | None = None
        self.query_iterator_batches: list[Any] | None = None
        self.query_iterators: list[FakeQueryIterator] = []
        self.upsert_response: dict[str, Any] | None = None
        self.delete_response: dict[str, Any] | list[str] | None = None
        self.stats_response: dict[str, Any] | None = None
        self.index_names = ["vector"]
        self.closed = False
        self.schema_description = "decision-agent-vector-schema-v1"
        self.primary_name = "record_id"
        self.primary_type: Any = DataType.VARCHAR
        self.index_description: dict[str, Any] = {
            "field_name": "vector",
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }

    def _call(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, copy.deepcopy(kwargs)))
        self.call_threads.append(threading.get_ident())
        if name in self.fail_methods:
            raise RuntimeError(f"fake {name} failure")

    def has_collection(self, **kwargs: Any) -> bool:
        self._call("has_collection", **kwargs)
        return self.collection_exists

    def create_schema(self, **kwargs: Any) -> FakeSchema:
        self._call("create_schema", **kwargs)
        return FakeSchema(**kwargs)

    def prepare_index_params(self) -> FakeIndexParams:
        self._call("prepare_index_params")
        return FakeIndexParams()

    def create_collection(self, **kwargs: Any) -> None:
        self._call("create_collection", **kwargs)
        self.collection_exists = True

    def create_index(self, **kwargs: Any) -> None:
        self._call("create_index", **kwargs)

    def load_collection(self, **kwargs: Any) -> None:
        self._call("load_collection", **kwargs)

    def describe_collection(self, **kwargs: Any) -> dict[str, Any]:
        self._call("describe_collection", **kwargs)
        return {
            "description": self.schema_description,
            "auto_id": False,
            "fields": [
                {
                    "name": self.primary_name,
                    "type": self.primary_type,
                    "is_primary": True,
                    "auto_id": False,
                    "params": {"max_length": 256},
                },
                {
                    "name": "vector",
                    "type": DataType.FLOAT_VECTOR,
                    "params": {"dim": self.dimension},
                },
                {"name": "parent_id", "type": DataType.VARCHAR, "params": {}},
                {"name": "document_id", "type": DataType.VARCHAR, "params": {}},
                {"name": "document_version", "type": DataType.VARCHAR, "params": {}},
                {"name": "content", "type": DataType.VARCHAR, "params": {}},
                {"name": "source", "type": DataType.VARCHAR, "params": {}},
                {"name": "page_number", "type": DataType.INT64, "params": {}},
                {"name": "metadata", "type": DataType.JSON, "params": {}},
                {"name": "schema_version", "type": DataType.VARCHAR, "params": {}},
            ],
        }

    def list_indexes(self, **kwargs: Any) -> list[str]:
        self._call("list_indexes", **kwargs)
        return list(self.index_names)

    def describe_index(self, **kwargs: Any) -> dict[str, Any]:
        self._call("describe_index", **kwargs)
        description = copy.deepcopy(self.index_description)
        description["index_name"] = kwargs["index_name"]
        return description

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._call("query", **kwargs)
        if self.query_response is not None:
            return copy.deepcopy(self.query_response)
        ids = kwargs.get("filter_params", {}).get("record_ids")
        if ids is None:
            return list(copy.deepcopy(self.rows).values())
        return [{"record_id": record_id} for record_id in ids if record_id in self.rows]

    def query_iterator(self, **kwargs: Any) -> FakeQueryIterator:
        self._call("query_iterator", **kwargs)
        if self.query_iterator_batches is None:
            record_ids = sorted(self.rows)
            batch_size = kwargs["batch_size"]
            batches = [
                [{"record_id": record_id} for record_id in record_ids[offset : offset + batch_size]]
                for offset in range(0, len(record_ids), batch_size)
            ]
        else:
            batches = self.query_iterator_batches
        iterator = FakeQueryIterator(self, batches)
        self.query_iterators.append(iterator)
        return iterator

    def upsert(self, **kwargs: Any) -> dict[str, int]:
        self._call("upsert", **kwargs)
        for row in kwargs["data"]:
            self.rows[row["record_id"]] = copy.deepcopy(row)
        if self.upsert_response is not None:
            return copy.deepcopy(self.upsert_response)
        return {"upsert_count": len(kwargs["data"])}

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self._call("search", **kwargs)
        if self.search_results is not None:
            return copy.deepcopy(self.search_results)
        hits = [
            {"id": row["record_id"], "distance": 1.0, "entity": copy.deepcopy(row)}
            for row in self.rows.values()
        ]
        return [hits[: kwargs["limit"]]]

    def delete(self, **kwargs: Any) -> dict[str, Any] | list[str]:
        self._call("delete", **kwargs)
        document_id = kwargs.get("filter_params", {}).get("document_id")
        ids = [key for key, value in self.rows.items() if value["document_id"] == document_id]
        for key in ids:
            del self.rows[key]
        if self.delete_response is not None:
            return copy.deepcopy(self.delete_response)
        return {"delete_count": len(ids)}

    def get_collection_stats(self, **kwargs: Any) -> dict[str, Any]:
        self._call("get_collection_stats", **kwargs)
        if self.stats_response is not None:
            return copy.deepcopy(self.stats_response)
        return {"row_count": len(self.rows)}

    def close(self) -> None:
        self._call("close")
        self.closed = True


def make_store(client: FakeMilvusClient, **kwargs: Any) -> MilvusVectorStore:
    return MilvusVectorStore(
        dimension=2,
        uri="http://localhost:19530",
        collection_name="test_chunks",
        client=client,
        **kwargs,
    )


def make_record(
    record_id: str,
    vector: list[float] | None = None,
    *,
    document_id: str = "doc-1",
    metadata: dict[str, Any] | None = None,
) -> VectorRecord:
    return VectorRecord(
        record_id=record_id,
        parent_id=f"parent-{record_id}",
        document_id=document_id,
        document_version="v1",
        content=f"content-{record_id}",
        vector=vector if vector is not None else [1.0, 0.0],
        source="reports/q2.txt",
        page_number=2,
        metadata=metadata or {"department": "sales"},
    )


async def initialized_store(
    client: FakeMilvusClient | None = None, **kwargs: Any
) -> tuple[MilvusVectorStore, FakeMilvusClient]:
    resolved_client = client or FakeMilvusClient()
    store = make_store(resolved_client, **kwargs)
    await store.initialize()
    return store, resolved_client


def calls(client: FakeMilvusClient, name: str) -> list[dict[str, Any]]:
    return [kwargs for call_name, kwargs in client.calls if call_name == name]


@pytest.mark.asyncio
async def test_constructor_does_not_create_client_or_perform_io() -> None:
    created = 0

    def factory(**kwargs: Any) -> FakeMilvusClient:
        nonlocal created
        created += 1
        return FakeMilvusClient()

    store = MilvusVectorStore(
        dimension=2,
        uri="http://localhost:19530",
        collection_name="test_chunks",
        client_factory=factory,
    )
    assert created == 0
    await store.initialize()
    assert created == 1


@pytest.mark.asyncio
async def test_from_settings_holds_secret_until_client_constructor_boundary() -> None:
    marker = "milvus-secret-marker"
    settings = Settings(app_name="Test Agent", milvus_token=marker, _env_file=None)
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeMilvusClient:
        captured.update(kwargs)
        return FakeMilvusClient(dimension=512)

    store = MilvusVectorStore.from_settings(settings, client_factory=factory)

    assert captured == {}
    assert isinstance(store._token, SecretStr)
    assert marker not in repr(store)
    assert marker not in repr(store.__dict__)

    await store.initialize()

    assert captured["token"] == marker
    assert isinstance(captured["token"], str)


@pytest.mark.asyncio
async def test_programmatic_plain_token_is_wrapped_before_initialize() -> None:
    marker = "programmatic-secret-marker"
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeMilvusClient:
        captured.update(kwargs)
        return FakeMilvusClient()

    store = MilvusVectorStore(
        dimension=2,
        uri="http://localhost:19530",
        collection_name="test_chunks",
        token=marker,
        client_factory=factory,
    )

    assert isinstance(store._token, SecretStr)
    assert marker not in repr(store.__dict__)
    await store.initialize()
    assert captured["token"] == marker


@pytest.mark.asyncio
async def test_none_token_stays_none_at_client_constructor_boundary() -> None:
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeMilvusClient:
        captured.update(kwargs)
        return FakeMilvusClient()

    store = MilvusVectorStore(
        dimension=2,
        uri="http://localhost:19530",
        collection_name="test_chunks",
        token=None,
        client_factory=factory,
    )

    await store.initialize()

    assert store._token is None
    assert captured["token"] is None


@pytest.mark.asyncio
async def test_client_constructor_failure_has_stable_secret_free_error() -> None:
    marker = "constructor-secret-marker"

    def factory(**_kwargs: Any) -> FakeMilvusClient:
        raise RuntimeError(f"fake constructor failure: {marker}")

    store = MilvusVectorStore(
        dimension=2,
        uri="http://localhost:19530",
        collection_name="test_chunks",
        token=marker,
        client_factory=factory,
    )

    with pytest.raises(VectorStoreConnectionError) as raised:
        await store.initialize()

    assert str(raised.value) == "failed to create Milvus client"
    assert marker not in str(raised.value)
    assert marker not in repr(raised.value)
    assert marker not in "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None
    assert store._client is None


@pytest.mark.asyncio
async def test_initialize_creates_missing_collection_with_explicit_schema() -> None:
    store, client = await initialized_store()
    create = calls(client, "create_collection")[0]
    schema = create["schema"]
    assert schema.options["auto_id"] is False
    assert schema.options["enable_dynamic_field"] is False
    assert schema.options["description"] == store.schema_description
    assert {field["name"] for field in schema.fields} == set(store.required_field_names)
    primary = next(field for field in schema.fields if field["name"] == "record_id")
    assert primary["is_primary"] is True


@pytest.mark.asyncio
async def test_initialize_creates_hnsw_cosine_index() -> None:
    _, client = await initialized_store()
    index = calls(client, "create_index")[0]["index_params"].indexes[0]
    assert index == {
        "field_name": "vector",
        "index_name": "vector",
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    }


@pytest.mark.asyncio
async def test_initialize_loads_collection() -> None:
    _, client = await initialized_store()
    assert len(calls(client, "load_collection")) == 1


@pytest.mark.asyncio
async def test_initialize_uses_create_then_index_then_load_order() -> None:
    _, client = await initialized_store()
    call_names = [name for name, _ in client.calls]
    assert call_names.index("create_collection") < call_names.index("create_index")
    assert call_names.index("create_index") < call_names.index("load_collection")


@pytest.mark.asyncio
async def test_initialize_is_idempotent() -> None:
    store, client = await initialized_store()
    call_count = len(client.calls)
    await store.initialize()
    assert len(client.calls) == call_count


@pytest.mark.asyncio
async def test_existing_compatible_collection_is_reused() -> None:
    client = FakeMilvusClient(collection_exists=True)
    _, client = await initialized_store(client)
    assert not calls(client, "create_collection")
    assert len(calls(client, "describe_collection")) == 1
    assert calls(client, "list_indexes")[0]["field_name"] == "vector"
    assert len(calls(client, "describe_index")) == 1


@pytest.mark.asyncio
async def test_existing_collection_accepts_nested_integer_hnsw_parameters() -> None:
    client = FakeMilvusClient(collection_exists=True)
    client.index_description["params"] = {"M": 16, "efConstruction": 200}
    await make_store(client).initialize()


@pytest.mark.asyncio
async def test_existing_collection_accepts_nested_string_hnsw_parameters() -> None:
    client = FakeMilvusClient(collection_exists=True)
    client.index_description["params"] = {"M": "16", "efConstruction": "200"}
    await make_store(client).initialize()


@pytest.mark.asyncio
async def test_existing_collection_accepts_flat_string_hnsw_parameters() -> None:
    client = FakeMilvusClient(collection_exists=True)
    client.index_description = {
        "field_name": "vector",
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "M": "16",
        "efConstruction": "200",
    }
    await make_store(client).initialize()


@pytest.mark.asyncio
async def test_existing_collection_falls_back_to_flat_parameter_when_nested_field_missing() -> None:
    client = FakeMilvusClient(collection_exists=True)
    client.index_description["params"] = {"M": 16}
    client.index_description["efConstruction"] = "200"
    await make_store(client).initialize()


@pytest.mark.asyncio
async def test_existing_collection_rejects_mismatched_hnsw_m() -> None:
    client = FakeMilvusClient(collection_exists=True)
    client.index_description["params"] = {"M": 32, "efConstruction": 200}
    with pytest.raises(
        VectorStoreSchemaError,
        match=r"^Milvus HNSW index parameters are incompatible$",
    ):
        await make_store(client).initialize()


@pytest.mark.asyncio
async def test_existing_collection_rejects_mismatched_hnsw_ef_construction() -> None:
    client = FakeMilvusClient(collection_exists=True)
    client.index_description["params"] = {"M": 16, "efConstruction": 400}
    with pytest.raises(
        VectorStoreSchemaError,
        match=r"^Milvus HNSW index parameters are incompatible$",
    ):
        await make_store(client).initialize()


@pytest.mark.asyncio
async def test_existing_collection_rejects_missing_hnsw_parameters() -> None:
    client = FakeMilvusClient(collection_exists=True)
    client.index_description.pop("params")
    with pytest.raises(
        VectorStoreSchemaError,
        match=r"^Milvus HNSW index parameters are incompatible$",
    ):
        await make_store(client).initialize()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_m", ["sixteen", None, True, 16.0])
async def test_existing_collection_rejects_invalid_hnsw_parameter_types(
    invalid_m: Any,
) -> None:
    client = FakeMilvusClient(collection_exists=True)
    client.index_description["params"] = {"M": invalid_m, "efConstruction": 200}
    with pytest.raises(
        VectorStoreSchemaError,
        match=r"^Milvus HNSW index parameters are incompatible$",
    ):
        await make_store(client).initialize()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("index_type", "IVF_FLAT", "Milvus vector index type is incompatible"),
        ("metric_type", "L2", "Milvus vector index metric is incompatible"),
    ],
)
async def test_existing_collection_still_rejects_incompatible_index_contract(
    field_name: str,
    invalid_value: str,
    message: str,
) -> None:
    client = FakeMilvusClient(collection_exists=True)
    client.index_description[field_name] = invalid_value
    with pytest.raises(VectorStoreSchemaError, match=f"^{message}$"):
        await make_store(client).initialize()


@pytest.mark.asyncio
async def test_existing_collection_describes_actual_listed_index_name() -> None:
    client = FakeMilvusClient(collection_exists=True)
    client.index_names = ["custom-vector-index"]
    await make_store(client).initialize()
    assert calls(client, "describe_index")[0]["index_name"] == "custom-vector-index"


@pytest.mark.asyncio
async def test_existing_collection_without_vector_index_is_rejected() -> None:
    client = FakeMilvusClient(collection_exists=True)
    client.index_names = []
    with pytest.raises(VectorStoreSchemaError, match="index"):
        await make_store(client).initialize()


@pytest.mark.asyncio
async def test_existing_collection_dimension_mismatch_is_rejected() -> None:
    client = FakeMilvusClient(collection_exists=True, dimension=3)
    with pytest.raises(VectorStoreSchemaError, match="dimension"):
        await make_store(client).initialize()


@pytest.mark.asyncio
async def test_existing_collection_primary_key_mismatch_is_rejected() -> None:
    client = FakeMilvusClient(collection_exists=True)
    client.primary_name = "id"
    with pytest.raises(VectorStoreSchemaError, match="primary"):
        await make_store(client).initialize()


@pytest.mark.asyncio
async def test_existing_collection_schema_version_mismatch_is_rejected() -> None:
    client = FakeMilvusClient(collection_exists=True)
    client.schema_description = "old-schema"
    with pytest.raises(VectorStoreSchemaError, match="schema version"):
        await make_store(client).initialize()


@pytest.mark.asyncio
async def test_existing_collection_key_field_type_mismatch_is_rejected() -> None:
    client = FakeMilvusClient(collection_exists=True)
    original_describe = client.describe_collection

    def describe_with_wrong_metadata_type(**kwargs: Any) -> dict[str, Any]:
        description = original_describe(**kwargs)
        next(field for field in description["fields"] if field["name"] == "metadata")["type"] = (
            DataType.VARCHAR
        )
        return description

    client.describe_collection = describe_with_wrong_metadata_type  # type: ignore[method-assign]
    with pytest.raises(VectorStoreSchemaError, match="metadata"):
        await make_store(client).initialize()


@pytest.mark.asyncio
async def test_initialize_converts_client_failure_and_preserves_cause() -> None:
    client = FakeMilvusClient()
    client.fail_methods.add("has_collection")
    with pytest.raises(VectorStoreConnectionError) as raised:
        await make_store(client).initialize()
    assert isinstance(raised.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_sync_client_calls_run_off_event_loop_thread() -> None:
    main_thread = threading.get_ident()
    _, client = await initialized_store()
    assert client.call_threads
    assert all(thread_id != main_thread for thread_id in client.call_threads)


@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    store, client = await initialized_store()
    await store.close()
    await store.close()
    assert len(calls(client, "close")) == 1


@pytest.mark.asyncio
async def test_uninitialized_operation_fails_clearly() -> None:
    with pytest.raises(VectorStoreConnectionError, match="initialize"):
        await make_store(FakeMilvusClient()).count()


@pytest.mark.asyncio
async def test_empty_upsert_returns_zero_without_remote_call() -> None:
    store, client = await initialized_store()
    result = await store.upsert([])
    assert result.attempted_count == result.inserted_count == result.updated_count == 0
    assert not calls(client, "upsert")


@pytest.mark.asyncio
async def test_upsert_uses_one_batch_query_and_one_batch_write() -> None:
    store, client = await initialized_store()
    result = await store.upsert([make_record("a"), make_record("b")])
    assert result.inserted_count == 2
    assert len(calls(client, "query")) == 1
    assert len(calls(client, "upsert")) == 1
    assert len(calls(client, "upsert")[0]["data"]) == 2


@pytest.mark.asyncio
async def test_cross_batch_same_id_is_reported_as_update() -> None:
    store, _ = await initialized_store()
    await store.upsert([make_record("a")])
    result = await store.upsert([make_record("a", [0.0, 1.0])])
    assert result.inserted_count == 0
    assert result.updated_count == 1


@pytest.mark.asyncio
async def test_duplicate_id_in_same_batch_is_rejected_before_remote_call() -> None:
    store, client = await initialized_store()
    with pytest.raises(RetrievalValidationError, match="duplicate record_id"):
        await store.upsert([make_record("a"), make_record("a")])
    assert not calls(client, "query")
    assert not calls(client, "upsert")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([1.0], "dimension"),
        ([math.nan, 0.0], "model validation"),
        ([math.inf, 0.0], "model validation"),
        ([0.0, 0.0], "zero vector"),
        ([], "model validation"),
    ],
)
async def test_invalid_upsert_vector_never_calls_remote(vector: list[float], message: str) -> None:
    store, client = await initialized_store()
    invalid = make_record("a").model_copy(update={"vector": vector})
    with pytest.raises(RetrievalValidationError, match=message):
        await store.upsert([invalid])
    assert not calls(client, "query")
    assert not calls(client, "upsert")


@pytest.mark.asyncio
async def test_overlong_field_is_rejected_without_truncation() -> None:
    limits = MilvusFieldLimits(content=5)
    store, client = await initialized_store(field_limits=limits)
    with pytest.raises(RetrievalValidationError, match="content"):
        await store.upsert([make_record("a")])
    assert not calls(client, "upsert")


@pytest.mark.asyncio
async def test_multibyte_field_limit_is_measured_as_utf8_bytes() -> None:
    limits = MilvusFieldLimits(content=5)
    store, client = await initialized_store(field_limits=limits)
    record = make_record("a").model_copy(update={"content": "中文"})
    with pytest.raises(RetrievalValidationError, match="content"):
        await store.upsert([record])
    assert not calls(client, "upsert")


@pytest.mark.asyncio
async def test_invalid_metadata_is_rejected_before_remote_call() -> None:
    store, client = await initialized_store()
    invalid = make_record("a").model_copy(update={"metadata": {"bad": object()}})
    with pytest.raises(RetrievalValidationError, match="model validation"):
        await store.upsert([invalid])
    assert not calls(client, "query")
    assert not calls(client, "upsert")


@pytest.mark.asyncio
async def test_metadata_is_copied_before_remote_write() -> None:
    metadata = {"department": "sales"}
    record = make_record("a", metadata=metadata)
    store, client = await initialized_store()
    await store.upsert([record])
    metadata["department"] = "changed"
    record.metadata["department"] = "also-changed"
    assert client.rows["a"]["metadata"] == {"department": "sales"}


@pytest.mark.asyncio
async def test_nullable_source_and_page_number_are_stored_explicitly() -> None:
    record = make_record("a").model_copy(update={"source": None, "page_number": None})
    store, client = await initialized_store()
    await store.upsert([record])
    assert client.rows["a"]["source"] is None
    assert client.rows["a"]["page_number"] is None


@pytest.mark.asyncio
async def test_upsert_remote_failure_does_not_return_success() -> None:
    store, client = await initialized_store()
    client.fail_methods.add("upsert")
    with pytest.raises(VectorStoreOperationError) as raised:
        await store.upsert([make_record("a")])
    assert isinstance(raised.value.__cause__, RuntimeError)


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [{}, {"upsert_count": 1}, {"upsert_count": "bad"}])
async def test_upsert_rejects_unconfirmed_or_mismatched_sdk_count(response: dict[str, Any]) -> None:
    store, client = await initialized_store()
    client.upsert_response = response
    with pytest.raises(VectorStoreOperationError, match="upsert_count"):
        await store.upsert([make_record("a"), make_record("b")])


@pytest.mark.asyncio
async def test_malformed_existing_id_query_is_converted() -> None:
    store, client = await initialized_store()
    client.query_response = [{"wrong_key": "a"}]
    with pytest.raises(VectorStoreOperationError, match="existing-ID") as raised:
        await store.upsert([make_record("a")])
    assert isinstance(raised.value.__cause__, KeyError)
    assert not calls(client, "upsert")


@pytest.mark.asyncio
async def test_search_empty_collection_returns_empty_list() -> None:
    store, _ = await initialized_store()
    assert await store.search([1.0, 0.0], 5) == []


@pytest.mark.asyncio
async def test_search_passes_cosine_hnsw_parameters_and_top_k() -> None:
    store, client = await initialized_store(hnsw_ef_search=77)
    await store.search([1.0, 0.0], 4)
    search = calls(client, "search")[0]
    assert search["limit"] == 4
    assert search["anns_field"] == "vector"
    assert search["search_params"] == {"metric_type": "COSINE", "params": {"ef": 77}}


@pytest.mark.asyncio
async def test_search_uses_parameterized_single_document_filter() -> None:
    store, client = await initialized_store()
    await store.search([1.0, 0.0], 3, VectorSearchFilter(document_id='doc" or true'))
    search = calls(client, "search")[0]
    assert search["filter"] == "document_id == {document_id}"
    assert search["filter_params"] == {"document_id": 'doc" or true'}
    assert 'doc" or true' not in search["filter"]


@pytest.mark.asyncio
async def test_search_uses_parameterized_multiple_document_filter() -> None:
    store, client = await initialized_store()
    await store.search([1.0, 0.0], 3, VectorSearchFilter(document_ids=("doc-a", "doc-b")))
    search = calls(client, "search")[0]
    assert search["filter"] == "document_id in {document_ids}"
    assert search["filter_params"] == {"document_ids": ["doc-a", "doc-b"]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([1.0], "dimension"),
        ([math.nan, 0.0], "finite"),
        ([math.inf, 0.0], "finite"),
        ([0.0, 0.0], "zero vector"),
        ([], "empty"),
    ],
)
async def test_invalid_search_vector_never_calls_remote(vector: list[float], message: str) -> None:
    store, client = await initialized_store()
    with pytest.raises(RetrievalValidationError, match=message):
        await store.search(vector, 1)
    assert not calls(client, "search")


@pytest.mark.asyncio
async def test_search_rejects_invalid_top_k_before_remote_call() -> None:
    store, client = await initialized_store()
    with pytest.raises(RetrievalValidationError, match="top_k"):
        await store.search([1.0, 0.0], 0)
    assert not calls(client, "search")


@pytest.mark.asyncio
async def test_search_maps_typed_results_and_stably_sorts_ties() -> None:
    client = FakeMilvusClient()
    client.search_results = [
        [
            {
                "id": "b",
                "distance": 0.75,
                "entity": make_store_row("b", metadata={"rank": 2}),
            },
            {
                "id": "a",
                "distance": 0.75,
                "entity": make_store_row("a", metadata={"rank": 1}),
            },
        ]
    ]
    store, _ = await initialized_store(client)
    results = await store.search([1.0, 0.0], 2)
    assert [result.record_id for result in results] == ["a", "b"]
    assert results[0].document_version == "v1"
    assert results[0].page_number == 2


def make_store_row(record_id: str, *, metadata: dict[str, Any]) -> dict[str, Any]:
    record = make_record(record_id, metadata=metadata)
    return {
        **record.model_dump(mode="python", exclude={"vector"}),
        "vector": record.vector,
        "schema_version": "v1",
    }


@pytest.mark.asyncio
async def test_search_results_do_not_expose_client_metadata_reference() -> None:
    client = FakeMilvusClient()
    client.search_results = [
        [{"id": "a", "distance": 1.0, "entity": make_store_row("a", metadata={"rank": 1})}]
    ]
    original = client.search_results
    store, _ = await initialized_store(client)
    result = (await store.search([1.0, 0.0], 1))[0]
    result.metadata["rank"] = 99
    assert original[0][0]["entity"]["metadata"] == {"rank": 1}


@pytest.mark.asyncio
async def test_search_remote_failure_is_converted() -> None:
    store, client = await initialized_store()
    client.fail_methods.add("search")
    with pytest.raises(VectorStoreOperationError) as raised:
        await store.search([1.0, 0.0], 1)
    assert isinstance(raised.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_malformed_search_response_is_converted() -> None:
    client = FakeMilvusClient()
    client.search_results = [[{"id": "a", "distance": 1.0, "entity": {}}]]
    store, _ = await initialized_store(client)
    with pytest.raises(VectorStoreOperationError, match="invalid search result") as raised:
        await store.search([1.0, 0.0], 1)
    assert isinstance(raised.value.__cause__, KeyError)


@pytest.mark.asyncio
async def test_delete_by_document_uses_parameterized_filter_and_returns_count() -> None:
    store, client = await initialized_store()
    await store.upsert(
        [make_record("a", document_id="doc-a"), make_record("b", document_id="doc-b")]
    )
    deleted = await store.delete_by_document('doc-a" or true')
    delete = calls(client, "delete")[0]
    assert delete["filter"] == "document_id == {document_id}"
    assert delete["filter_params"] == {"document_id": 'doc-a" or true'}
    assert deleted == 0
    assert "b" in client.rows


@pytest.mark.asyncio
async def test_delete_existing_document_returns_confirmed_count() -> None:
    store, _ = await initialized_store()
    await store.upsert([make_record("a", document_id="doc-a")])
    assert await store.delete_by_document("doc-a") == 1


@pytest.mark.asyncio
async def test_delete_zero_count_omitted_by_sdk_returns_zero() -> None:
    store, client = await initialized_store()
    client.delete_response = {}
    assert await store.delete_by_document("missing-doc") == 0


@pytest.mark.asyncio
async def test_legacy_delete_primary_key_list_returns_confirmed_count() -> None:
    store, client = await initialized_store()
    client.delete_response = ["a", "b"]
    assert await store.delete_by_document("doc-a") == 2


@pytest.mark.asyncio
async def test_delete_rejects_blank_document_id() -> None:
    store, client = await initialized_store()
    with pytest.raises(RetrievalValidationError, match="document_id"):
        await store.delete_by_document("  ")
    assert not calls(client, "delete")


@pytest.mark.asyncio
async def test_list_record_ids_uses_strong_primary_key_iterator_and_dedupes() -> None:
    store, client = await initialized_store()
    client.query_iterator_batches = [
        [{"record_id": "a"}, {"record_id": "a"}, {"record_id": "b"}],
        [],
    ]

    assert await store.list_record_ids() == frozenset({"a", "b"})

    query = calls(client, "query_iterator")[0]
    assert query["output_fields"] == ["record_id"]
    assert query["consistency_level"] == "Strong"
    assert query["batch_size"] == store.RECORD_ID_BATCH_SIZE
    assert query["limit"] == -1
    assert client.query_iterators[0].closed is True
    assert not calls(client, "get_collection_stats")


@pytest.mark.asyncio
async def test_list_record_ids_reads_every_iterator_batch() -> None:
    store, client = await initialized_store()
    record_ids = {f"record-{index:04d}" for index in range(store.RECORD_ID_BATCH_SIZE + 1)}
    client.rows = {record_id: {"record_id": record_id} for record_id in record_ids}

    assert await store.list_record_ids() == frozenset(record_ids)
    assert client.query_iterators[0].next_calls == 3
    assert client.query_iterators[0].closed is True


@pytest.mark.asyncio
async def test_list_record_ids_empty_collection_returns_empty_set() -> None:
    store, client = await initialized_store()
    assert await store.list_record_ids() == frozenset()
    assert client.query_iterators[0].closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [None, {}, "invalid"])
async def test_list_record_ids_rejects_invalid_query_response(response: Any) -> None:
    store, client = await initialized_store()
    client.query_iterator_batches = [response]
    with pytest.raises(VectorStoreOperationError, match="query response"):
        await store.list_record_ids()
    assert client.query_iterators[0].closed is True


@pytest.mark.asyncio
async def test_list_record_ids_rejects_missing_record_id() -> None:
    store, client = await initialized_store()
    client.query_iterator_batches = [[{}]]
    with pytest.raises(VectorStoreOperationError, match="record_id"):
        await store.list_record_ids()
    assert client.query_iterators[0].closed is True


@pytest.mark.asyncio
async def test_list_record_ids_rejects_non_string_record_id() -> None:
    store, client = await initialized_store()
    client.query_iterator_batches = [[{"record_id": 7}]]
    with pytest.raises(VectorStoreOperationError, match="record_id"):
        await store.list_record_ids()
    assert client.query_iterators[0].closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("record_id", ["", "   "])
async def test_list_record_ids_rejects_blank_record_id(record_id: str) -> None:
    store, client = await initialized_store()
    client.query_iterator_batches = [[{"record_id": record_id}]]
    with pytest.raises(VectorStoreOperationError, match="record_id"):
        await store.list_record_ids()
    assert client.query_iterators[0].closed is True


@pytest.mark.asyncio
async def test_count_uses_logical_record_ids_instead_of_physical_stats() -> None:
    store, client = await initialized_store()
    client.rows = {f"record-{index}": {"record_id": f"record-{index}"} for index in range(7)}
    client.stats_response = {"row_count": 522}

    assert await store.count() == 7
    assert not calls(client, "get_collection_stats")
    assert len(calls(client, "query_iterator")) == 1


@pytest.mark.asyncio
async def test_count_client_failure_is_converted() -> None:
    store, client = await initialized_store()
    client.fail_methods.add("query_iterator")
    with pytest.raises(VectorStoreOperationError) as raised:
        await store.count()
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_milvus_store_satisfies_vector_store_protocol() -> None:
    assert isinstance(make_store(FakeMilvusClient()), VectorStore)


def test_installed_sdk_accepts_collection_schema_and_hnsw_contract() -> None:
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        description="decision-agent-vector-schema-v1",
    )
    schema.add_field(
        field_name="record_id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        auto_id=False,
        max_length=256,
    )
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=2)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=2048, nullable=True)
    schema.add_field(field_name="page_number", datatype=DataType.INT64, nullable=True)
    schema.add_field(field_name="metadata", datatype=DataType.JSON)
    schema.add_field(field_name="schema_version", datatype=DataType.VARCHAR, max_length=16)
    schema_dict = schema.to_dict()

    assert schema_dict["auto_id"] is False
    assert schema_dict["enable_dynamic_field"] is False
    fields = {field["name"]: field for field in schema_dict["fields"]}
    assert fields["record_id"]["params"]["max_length"] == 256
    assert fields["record_id"]["is_primary"] is True
    assert fields["vector"]["params"]["dim"] == 2
    assert fields["source"]["nullable"] is True
    assert fields["page_number"]["nullable"] is True
    assert fields["metadata"]["type"] == DataType.JSON

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    assert [index.to_dict() for index in index_params] == [
        {
            "field_name": "vector",
            "index_name": "vector",
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "M": 16,
            "efConstruction": 200,
        }
    ]


def make_chunk(chunk_id: str, content: str) -> ChildChunk:
    return ChildChunk(
        chunk_id=chunk_id,
        parent_id=f"parent-{chunk_id}",
        document_id="doc-flow",
        document_version="v1",
        content=content,
        source="flow.txt",
        page_number=1,
        start_offset=0,
        end_offset=len(content),
        metadata={"flow": True},
    )


@pytest.mark.asyncio
async def test_dense_indexer_and_retriever_use_milvus_without_business_changes() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=2)
    client = FakeMilvusClient(dimension=2)
    store = make_store(client)
    await store.initialize()
    await DenseIndexer(provider, store).index([make_chunk("child-1", "east sales")])
    results = await DenseRetriever(provider, store).retrieve("east sales", top_k=1)
    assert results[0].record_id == "child-1"


def test_dense_services_reject_milvus_dimension_mismatch_before_calls() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=4)
    store = make_store(FakeMilvusClient())
    with pytest.raises(RetrievalValidationError, match="dimensions must match"):
        DenseIndexer(provider, store)
    with pytest.raises(RetrievalValidationError, match="dimensions must match"):
        DenseRetriever(provider, store)
