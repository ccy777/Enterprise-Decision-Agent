"""Typed application exceptions shared across architectural layers."""


class DecisionAgentError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(DecisionAgentError):
    """Raised when required application configuration is invalid or absent."""


class DomainValidationError(DecisionAgentError):
    """Raised when a domain invariant cannot be satisfied."""


class DependencyUnavailableError(DecisionAgentError):
    """Raised when a required external dependency is unavailable."""


class DocumentIngestionError(DecisionAgentError):
    """Base exception for expected document ingestion failures."""


class InvalidDocumentSourceError(DocumentIngestionError):
    """Raised when a document source is missing, unsafe, or not a regular file."""


class UnsupportedDocumentTypeError(DocumentIngestionError):
    """Raised when no parser supports a document suffix."""


class DocumentTooLargeError(DocumentIngestionError):
    """Raised before a document larger than the configured limit is fully read."""


class DocumentDecodingError(DocumentIngestionError):
    """Raised when text cannot be decoded without replacing damaged characters."""


class DocumentParsingError(DocumentIngestionError):
    """Raised when a supported document cannot be parsed safely."""


class RetrievalValidationError(DomainValidationError):
    """Raised when retrieval input violates a typed contract or vector invariant."""


class EmbeddingError(DecisionAgentError):
    """Base exception for expected local embedding provider failures."""


class EmbeddingModelLoadError(EmbeddingError, DependencyUnavailableError):
    """Raised when a local embedding model cannot be initialized."""


class EmbeddingInferenceError(EmbeddingError):
    """Raised when local embedding inference or output validation fails."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when configured, model, and output dimensions disagree."""


class RerankerError(DecisionAgentError):
    """Base exception for expected cross-encoder reranking failures."""


class RerankerModelLoadError(RerankerError, DependencyUnavailableError):
    """Raised when the configured cross-encoder cannot be initialized."""


class RerankerInferenceError(RerankerError):
    """Raised when cross-encoder scoring fails or returns invalid scores."""


class EvaluationError(DecisionAgentError):
    """Base exception for expected offline evaluation failures."""


class EvaluationValidationError(EvaluationError, DomainValidationError):
    """Raised when an evaluation dataset or metric input is invalid."""


class VectorStoreError(DecisionAgentError):
    """Base exception for expected external vector-store failures."""


class VectorStoreConnectionError(VectorStoreError, DependencyUnavailableError):
    """Raised when a vector-store client cannot connect or initialize."""


class VectorStoreSchemaError(VectorStoreError):
    """Raised when an existing vector collection is incompatible."""


class VectorStoreOperationError(VectorStoreError):
    """Raised when a remote vector-store operation fails."""


class SafeQueryError(DecisionAgentError):
    """Base exception for expected safe-query integration failures."""
