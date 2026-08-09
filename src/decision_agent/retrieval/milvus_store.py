"""Async Milvus adapter for the typed vector-store contract."""

from __future__ import annotations

import asyncio
import copy
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, ClassVar, TypeVar

from pydantic import SecretStr, ValidationError
from pymilvus import DataType, MilvusClient

from decision_agent.config import Settings
from decision_agent.exceptions import (
    RetrievalValidationError,
    VectorStoreConnectionError,
    VectorStoreOperationError,
    VectorStoreSchemaError,
)
from decision_agent.retrieval.models import (
    VectorRecord,
    VectorSearchFilter,
    VectorSearchResult,
    VectorUpsertResult,
)

_T = TypeVar("_T")
_ClientFactory = Callable[..., Any]
_HNSW_PARAMETERS_INCOMPATIBLE = "Milvus HNSW index parameters are incompatible"


def _parse_hnsw_index_parameter(index: Mapping[str, Any], name: str) -> int:
    params = index.get("params")
    if "params" in index and not isinstance(params, Mapping):
        raise VectorStoreSchemaError(_HNSW_PARAMETERS_INCOMPATIBLE)

    if isinstance(params, Mapping) and name in params:
        value = params[name]
    elif name in index:
        value = index[name]
    else:
        raise VectorStoreSchemaError(_HNSW_PARAMETERS_INCOMPATIBLE)

    if isinstance(value, bool):
        raise VectorStoreSchemaError(_HNSW_PARAMETERS_INCOMPATIBLE)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    raise VectorStoreSchemaError(_HNSW_PARAMETERS_INCOMPATIBLE)


@dataclass(frozen=True, slots=True)
class MilvusFieldLimits:
    """Explicit client-side VARCHAR and JSON size limits."""

    record_id: int = 256
    parent_id: int = 256
    document_id: int = 256
    document_version: int = 128
    content: int = 65_535
    source: int = 2_048
    metadata_bytes: int = 65_536

    def __post_init__(self) -> None:
        if any(getattr(self, field.name) <= 0 for field in fields(self)):
            raise RetrievalValidationError("Milvus field limits must be greater than zero")


class MilvusVectorStore:
    """Direct pymilvus adapter with explicit lifecycle and validation boundaries.

    Client-side batch validation prevents avoidable partial requests, but it is
    not a Milvus server-side transaction. The inserted/updated split is observed
    by a pre-query and can race with concurrent writers. Stable record IDs plus
    Milvus upsert provide an idempotent-write foundation, not exactly-once delivery.
    """

    SCHEMA_VERSION = "v1"
    SCHEMA_DESCRIPTION = "decision-agent-vector-schema-v1"
    RECORD_ID_BATCH_SIZE = 1_000
    REQUIRED_FIELD_NAMES = (
        "record_id",
        "vector",
        "parent_id",
        "document_id",
        "document_version",
        "content",
        "source",
        "page_number",
        "metadata",
        "schema_version",
    )
    OUTPUT_FIELDS: ClassVar[list[str]] = [
        "record_id",
        "parent_id",
        "document_id",
        "document_version",
        "content",
        "source",
        "page_number",
        "metadata",
    ]

    def __init__(
        self,
        *,
        dimension: int,
        uri: str,
        collection_name: str,
        token: SecretStr | str | None = None,
        database: str = "default",
        metric_type: str = "COSINE",
        index_type: str = "HNSW",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        timeout_seconds: float = 10.0,
        field_limits: MilvusFieldLimits | None = None,
        client: Any | None = None,
        client_factory: _ClientFactory | None = None,
    ) -> None:
        if dimension <= 0:
            raise RetrievalValidationError("store dimension must be greater than zero")
        if not uri.strip():
            raise RetrievalValidationError("Milvus URI cannot be empty")
        if not collection_name.strip():
            raise RetrievalValidationError("Milvus collection name cannot be empty")
        if metric_type != "COSINE":
            raise RetrievalValidationError("Milvus metric_type must be COSINE")
        if index_type != "HNSW":
            raise RetrievalValidationError("Milvus index_type must be HNSW")
        if min(hnsw_m, hnsw_ef_construction, hnsw_ef_search) <= 0:
            raise RetrievalValidationError("HNSW parameters must be greater than zero")
        if timeout_seconds <= 0:
            raise RetrievalValidationError("Milvus timeout must be greater than zero")
        if client is not None and client_factory is not None:
            raise RetrievalValidationError("provide either client or client_factory, not both")

        self._dimension = dimension
        self._uri = uri
        self._collection_name = collection_name
        self._token = SecretStr(token) if isinstance(token, str) else token
        self._database = database
        self._metric_type = metric_type
        self._index_type = index_type
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search
        self._timeout_seconds = timeout_seconds
        self._field_limits = field_limits or MilvusFieldLimits()
        self._client = client
        self._client_factory = client_factory or MilvusClient
        self._initialized = False
        self._closed = False

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        client: Any | None = None,
        client_factory: _ClientFactory | None = None,
        field_limits: MilvusFieldLimits | None = None,
    ) -> MilvusVectorStore:
        """Build an uninitialized adapter from validated process settings."""
        return cls(
            dimension=settings.milvus_dimension,
            uri=settings.milvus_uri,
            collection_name=settings.milvus_collection,
            token=settings.milvus_token,
            database=settings.milvus_database,
            metric_type=settings.milvus_metric_type.value,
            index_type=settings.milvus_index_type.value,
            hnsw_m=settings.hnsw_m,
            hnsw_ef_construction=settings.hnsw_ef_construction,
            hnsw_ef_search=settings.hnsw_ef_search,
            timeout_seconds=settings.milvus_timeout_seconds,
            field_limits=field_limits,
            client=client,
            client_factory=client_factory,
        )

    @property
    def dimension(self) -> int:
        """Return the vector dimension accepted by the collection."""
        return self._dimension

    @property
    def schema_description(self) -> str:
        """Return the collection-level schema/version marker."""
        return self.SCHEMA_DESCRIPTION

    @property
    def required_field_names(self) -> tuple[str, ...]:
        """Return fields required for a compatible collection."""
        return self.REQUIRED_FIELD_NAMES

    async def initialize(self) -> None:
        """Create or validate, index, and load the configured collection once."""
        if self._initialized:
            return
        if self._closed:
            raise VectorStoreConnectionError("closed Milvus store cannot be initialized")
        if self._client is None:
            try:
                self._client = await asyncio.to_thread(
                    self._client_factory,
                    uri=self._uri,
                    token=(self._token.get_secret_value() if self._token is not None else None),
                    db_name=self._database,
                    timeout=self._timeout_seconds,
                )
            except Exception:
                raise VectorStoreConnectionError("failed to create Milvus client") from None

        exists = await self._connection_call(
            "check collection existence",
            self._client.has_collection,
            collection_name=self._collection_name,
            timeout=self._timeout_seconds,
        )
        if exists:
            description = await self._connection_call(
                "describe collection",
                self._client.describe_collection,
                collection_name=self._collection_name,
                timeout=self._timeout_seconds,
            )
            self._validate_existing_schema(description)
            index_names = await self._connection_call(
                "list vector indexes",
                self._client.list_indexes,
                collection_name=self._collection_name,
                field_name="vector",
            )
            if len(index_names) != 1:
                raise VectorStoreSchemaError("Milvus collection must have exactly one vector index")
            index = await self._connection_call(
                "describe vector index",
                self._client.describe_index,
                collection_name=self._collection_name,
                index_name=index_names[0],
                timeout=self._timeout_seconds,
            )
            self._validate_existing_index(index)
        else:
            await self._create_collection_and_index()

        await self._connection_call(
            "load collection",
            self._client.load_collection,
            collection_name=self._collection_name,
            timeout=self._timeout_seconds,
        )
        self._initialized = True

    async def close(self) -> None:
        """Release the client once; repeated calls are no-ops."""
        if self._closed:
            return
        if self._client is not None:
            await self._operation_call("close Milvus client", self._client.close)
        self._closed = True
        self._initialized = False

    async def upsert(self, records: Sequence[VectorRecord]) -> VectorUpsertResult:
        """Validate locally, observe existing IDs once, then issue one batch upsert."""
        if not records:
            return VectorUpsertResult(attempted_count=0, inserted_count=0, updated_count=0)
        self._require_initialized()
        validated = self._validate_records(records)
        record_ids = [record.record_id for record in validated]
        existing_rows = await self._operation_call(
            "query existing vector IDs",
            self._client.query,
            collection_name=self._collection_name,
            filter="record_id in {record_ids}",
            filter_params={"record_ids": record_ids},
            output_fields=["record_id"],
            timeout=self._timeout_seconds,
        )
        try:
            existing_ids = {str(row["record_id"]) for row in existing_rows}
        except (KeyError, TypeError) as exc:
            raise VectorStoreOperationError(
                "Milvus returned an invalid existing-ID query response"
            ) from exc
        rows = [self._record_to_row(record) for record in validated]
        upsert_result = await self._operation_call(
            "upsert vector batch",
            self._client.upsert,
            collection_name=self._collection_name,
            data=rows,
            timeout=self._timeout_seconds,
        )
        try:
            upsert_count = int(upsert_result["upsert_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VectorStoreOperationError("Milvus returned an invalid upsert_count") from exc
        if upsert_count != len(rows):
            raise VectorStoreOperationError(
                f"Milvus upsert_count {upsert_count} does not match batch size {len(rows)}"
            )
        updated_count = sum(record_id in existing_ids for record_id in record_ids)
        return VectorUpsertResult(
            attempted_count=len(rows),
            inserted_count=len(rows) - updated_count,
            updated_count=updated_count,
        )

    async def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """Run a parameterized COSINE search and map typed, copied results."""
        self._require_initialized()
        if top_k <= 0:
            raise RetrievalValidationError("top_k must be greater than zero")
        query = [float(value) for value in query_vector]
        self._validate_vector(query, field_name="query vector")
        filter_expression, filter_params = self._build_search_filter(filters)
        response = await self._operation_call(
            "search vectors",
            self._client.search,
            collection_name=self._collection_name,
            data=[query],
            anns_field="vector",
            filter=filter_expression,
            filter_params=filter_params,
            limit=top_k,
            output_fields=self.OUTPUT_FIELDS,
            search_params={
                "metric_type": self._metric_type,
                "params": {"ef": self._hnsw_ef_search},
            },
            timeout=self._timeout_seconds,
        )
        hits = response[0] if response else []
        try:
            results = [self._hit_to_result(hit) for hit in hits]
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise VectorStoreOperationError("Milvus returned an invalid search result") from exc
        results.sort(key=lambda result: (-result.score, result.record_id))
        return results[:top_k]

    async def delete_by_document(self, document_id: str) -> int:
        """Delete only one parameterized document scope and return confirmed count."""
        self._require_initialized()
        if not document_id.strip():
            raise RetrievalValidationError("document_id cannot be empty or whitespace")
        result = await self._operation_call(
            "delete document vectors",
            self._client.delete,
            collection_name=self._collection_name,
            filter="document_id == {document_id}",
            filter_params={"document_id": document_id},
            timeout=self._timeout_seconds,
        )
        if isinstance(result, list):
            return len(result)
        if not isinstance(result, Mapping):
            raise VectorStoreOperationError("Milvus returned an invalid delete response")
        try:
            return int(result.get("delete_count", 0))
        except (TypeError, ValueError) as exc:
            raise VectorStoreOperationError("Milvus returned an invalid delete_count") from exc

    async def count(self) -> int:
        """Return the queryable logical record count."""
        return len(await self.list_record_ids())

    async def list_record_ids(self) -> frozenset[str]:
        """Read every queryable logical primary key through a strong iterator snapshot."""
        self._require_initialized()
        iterator = await self._operation_call(
            "create record-ID query iterator",
            self._client.query_iterator,
            collection_name=self._collection_name,
            batch_size=self.RECORD_ID_BATCH_SIZE,
            limit=-1,
            filter="",
            output_fields=["record_id"],
            consistency_level="Strong",
            timeout=self._timeout_seconds,
        )
        next_batch = getattr(iterator, "next", None)
        close_iterator = getattr(iterator, "close", None)
        if not callable(next_batch) or not callable(close_iterator):
            raise VectorStoreOperationError("Milvus returned an invalid record-ID query iterator")

        record_ids: set[str] = set()
        primary_error: BaseException | None = None
        try:
            while True:
                batch = await self._operation_call(
                    "read record-ID query batch",
                    next_batch,
                )
                if not isinstance(batch, list):
                    raise VectorStoreOperationError(
                        "Milvus returned an invalid record-ID query response"
                    )
                if not batch:
                    break
                for row in batch:
                    if not isinstance(row, Mapping):
                        raise VectorStoreOperationError(
                            "Milvus returned an invalid record-ID query response"
                        )
                    record_id = row.get("record_id")
                    if not isinstance(record_id, str) or not record_id.strip():
                        raise VectorStoreOperationError("Milvus returned an invalid record_id")
                    record_ids.add(record_id)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                await self._operation_call(
                    "close record-ID query iterator",
                    close_iterator,
                )
            except VectorStoreOperationError as close_error:
                if primary_error is None:
                    raise
                primary_error.add_note("Milvus record-ID query iterator cleanup also failed")
                close_error.add_note("Primary record-ID query failure remains authoritative")
        return frozenset(record_ids)

    async def _create_collection_and_index(self) -> None:
        schema = await self._connection_call(
            "create collection schema",
            self._client.create_schema,
            auto_id=False,
            enable_dynamic_field=False,
            description=self.SCHEMA_DESCRIPTION,
        )
        limits = self._field_limits
        schema.add_field(
            field_name="record_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=limits.record_id,
        )
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
        schema.add_field(
            field_name="parent_id", datatype=DataType.VARCHAR, max_length=limits.parent_id
        )
        schema.add_field(
            field_name="document_id", datatype=DataType.VARCHAR, max_length=limits.document_id
        )
        schema.add_field(
            field_name="document_version",
            datatype=DataType.VARCHAR,
            max_length=limits.document_version,
        )
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=limits.content)
        schema.add_field(
            field_name="source",
            datatype=DataType.VARCHAR,
            max_length=limits.source,
            nullable=True,
        )
        schema.add_field(field_name="page_number", datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        schema.add_field(
            field_name="schema_version",
            datatype=DataType.VARCHAR,
            max_length=16,
        )
        await self._connection_call(
            "create collection",
            self._client.create_collection,
            collection_name=self._collection_name,
            schema=schema,
            timeout=self._timeout_seconds,
        )
        index_params = await self._connection_call(
            "prepare vector index", self._client.prepare_index_params
        )
        index_params.add_index(
            field_name="vector",
            index_name="vector",
            index_type=self._index_type,
            metric_type=self._metric_type,
            params={"M": self._hnsw_m, "efConstruction": self._hnsw_ef_construction},
        )
        await self._connection_call(
            "create vector index",
            self._client.create_index,
            collection_name=self._collection_name,
            index_params=index_params,
            timeout=self._timeout_seconds,
        )

    def _validate_existing_schema(self, description: Mapping[str, Any]) -> None:
        if description.get("description") != self.SCHEMA_DESCRIPTION:
            raise VectorStoreSchemaError("Milvus collection schema version is incompatible")
        fields = {str(field.get("name")): field for field in description.get("fields", [])}
        primary_names = {name for name, field in fields.items() if bool(field.get("is_primary"))}
        if primary_names != {"record_id"}:
            raise VectorStoreSchemaError("Milvus collection primary key is incompatible")
        missing = set(self.REQUIRED_FIELD_NAMES).difference(fields)
        if missing:
            raise VectorStoreSchemaError(
                f"Milvus collection is missing required fields: {sorted(missing)}"
            )
        primary = fields["record_id"]
        if not primary.get("is_primary") or not self._same_type(
            primary.get("type"), DataType.VARCHAR
        ):
            raise VectorStoreSchemaError("Milvus collection primary key is incompatible")
        if description.get("auto_id") or primary.get("auto_id"):
            raise VectorStoreSchemaError("Milvus record_id primary key must disable auto_id")
        vector = fields["vector"]
        if not self._same_type(vector.get("type"), DataType.FLOAT_VECTOR):
            raise VectorStoreSchemaError("Milvus vector field type is incompatible")
        actual_dimension = int(vector.get("params", {}).get("dim", 0))
        if actual_dimension != self.dimension:
            raise VectorStoreSchemaError(
                f"Milvus vector dimension {actual_dimension} does not match {self.dimension}"
            )
        expected_types = {
            "parent_id": DataType.VARCHAR,
            "document_id": DataType.VARCHAR,
            "document_version": DataType.VARCHAR,
            "content": DataType.VARCHAR,
            "source": DataType.VARCHAR,
            "page_number": DataType.INT64,
            "metadata": DataType.JSON,
            "schema_version": DataType.VARCHAR,
        }
        for field_name, expected_type in expected_types.items():
            if not self._same_type(fields[field_name].get("type"), expected_type):
                raise VectorStoreSchemaError(f"Milvus field {field_name} type is incompatible")

    def _validate_existing_index(self, index: Mapping[str, Any]) -> None:
        if index.get("field_name") != "vector":
            raise VectorStoreSchemaError("Milvus vector index field is incompatible")
        if index.get("index_type") != self._index_type:
            raise VectorStoreSchemaError("Milvus vector index type is incompatible")
        if index.get("metric_type") != self._metric_type:
            raise VectorStoreSchemaError("Milvus vector index metric is incompatible")
        if (
            _parse_hnsw_index_parameter(index, "M") != self._hnsw_m
            or _parse_hnsw_index_parameter(index, "efConstruction") != self._hnsw_ef_construction
        ):
            raise VectorStoreSchemaError(_HNSW_PARAMETERS_INCOMPATIBLE)

    def _validate_records(self, records: Sequence[VectorRecord]) -> list[VectorRecord]:
        validated: list[VectorRecord] = []
        seen: set[str] = set()
        for record in records:
            try:
                current = VectorRecord.model_validate(
                    record.model_dump(mode="python", warnings="none")
                )
            except ValidationError as exc:
                raise RetrievalValidationError("vector record failed model validation") from exc
            if current.record_id in seen:
                raise RetrievalValidationError(
                    f"duplicate record_id in one upsert batch: {current.record_id}"
                )
            self._validate_vector(current.vector, field_name="record vector")
            self._validate_record_fields(current)
            seen.add(current.record_id)
            validated.append(current.model_copy(deep=True))
        return validated

    def _validate_record_fields(self, record: VectorRecord) -> None:
        limits = self._field_limits
        values = {
            "record_id": (record.record_id, limits.record_id),
            "parent_id": (record.parent_id, limits.parent_id),
            "document_id": (record.document_id, limits.document_id),
            "document_version": (record.document_version, limits.document_version),
            "content": (record.content, limits.content),
        }
        if record.source is not None:
            values["source"] = (record.source, limits.source)
        for field_name, (value, maximum) in values.items():
            if len(value.encode("utf-8")) > maximum:
                raise RetrievalValidationError(
                    f"{field_name} exceeds Milvus maximum UTF-8 byte length {maximum}"
                )
        try:
            metadata_json = json.dumps(
                record.metadata,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise RetrievalValidationError("metadata must be JSON-safe") from exc
        if len(metadata_json.encode("utf-8")) > limits.metadata_bytes:
            raise RetrievalValidationError("metadata exceeds Milvus JSON size limit")

    def _validate_vector(self, vector: Sequence[float], *, field_name: str) -> float:
        if not vector:
            raise RetrievalValidationError(f"{field_name} cannot be empty")
        if len(vector) != self.dimension:
            raise RetrievalValidationError(
                f"{field_name} dimension {len(vector)} does not match store dimension "
                f"{self.dimension}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise RetrievalValidationError(f"{field_name} elements must be finite")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            raise RetrievalValidationError(f"{field_name} cannot be a zero vector")
        return norm

    def _record_to_row(self, record: VectorRecord) -> dict[str, Any]:
        return {
            "record_id": record.record_id,
            "vector": list(record.vector),
            "parent_id": record.parent_id,
            "document_id": record.document_id,
            "document_version": record.document_version,
            "content": record.content,
            "source": record.source,
            "page_number": record.page_number,
            "metadata": copy.deepcopy(record.metadata),
            "schema_version": self.SCHEMA_VERSION,
        }

    @staticmethod
    def _build_search_filter(
        filters: VectorSearchFilter | None,
    ) -> tuple[str, dict[str, Any]]:
        if filters is None:
            return "", {}
        if filters.document_id is not None:
            return "document_id == {document_id}", {"document_id": filters.document_id}
        if filters.document_ids:
            return "document_id in {document_ids}", {"document_ids": list(filters.document_ids)}
        return "", {}

    @staticmethod
    def _hit_to_result(hit: Mapping[str, Any]) -> VectorSearchResult:
        entity = copy.deepcopy(dict(hit.get("entity", {})))
        record_id = str(entity.get("record_id", hit.get("id", "")))
        score = float(hit.get("distance", hit.get("score")))
        return VectorSearchResult(
            record_id=record_id,
            score=score,
            content=entity["content"],
            parent_id=entity["parent_id"],
            document_id=entity["document_id"],
            document_version=entity["document_version"],
            source=entity.get("source"),
            page_number=entity.get("page_number"),
            metadata=copy.deepcopy(entity.get("metadata", {})),
        )

    def _require_initialized(self) -> None:
        if not self._initialized or self._client is None:
            raise VectorStoreConnectionError("Milvus store must initialize before use")

    async def _connection_call(
        self, operation: str, function: Callable[..., _T], **kwargs: Any
    ) -> _T:
        try:
            return await asyncio.to_thread(function, **kwargs)
        except Exception as exc:
            raise VectorStoreConnectionError(f"Milvus failed to {operation}") from exc

    async def _operation_call(
        self, operation: str, function: Callable[..., _T], **kwargs: Any
    ) -> _T:
        try:
            return await asyncio.to_thread(function, **kwargs)
        except Exception as exc:
            raise VectorStoreOperationError(f"Milvus failed to {operation}") from exc

    @staticmethod
    def _same_type(actual: Any, expected: DataType) -> bool:
        try:
            return int(actual) == int(expected)
        except (TypeError, ValueError):
            return actual == expected
