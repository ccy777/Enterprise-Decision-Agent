"""Deterministic, character-budgeted context assembly from parent evidence."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from decision_agent.domain.models import Metadata
from decision_agent.exceptions import RetrievalValidationError
from decision_agent.retrieval.parent_expansion import MatchedChild, ParentExpansionResult


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvidenceItem(EvidenceModel):
    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    final_rank: int = Field(gt=0)
    parent_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    original_content_length: int = Field(gt=0)
    included_content_length: int = Field(gt=0)
    truncated: bool
    matched_child_count: int = Field(gt=0)
    best_child_rank: int = Field(gt=0)
    metadata: Metadata = Field(default_factory=dict)
    provenance: Metadata = Field(default_factory=dict)
    matched_children: tuple[MatchedChild, ...] = Field(min_length=1)


class EvidenceReference(EvidenceModel):
    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    parent_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)


class EvidenceContext(EvidenceModel):
    rendered_context: str
    evidence_items: tuple[EvidenceItem, ...]
    references: tuple[EvidenceReference, ...]
    included_evidence_count: int = Field(ge=0)
    omitted_evidence_count: int = Field(ge=0)
    total_original_chars: int = Field(ge=0)
    total_included_chars: int = Field(ge=0)
    truncated: bool


class EvidenceContextBuilder:
    """Preserve parent order while assigning citations within character budgets."""

    def __init__(
        self,
        *,
        max_total_chars: int,
        max_evidence_chars: int,
        max_evidence_count: int,
        truncation_marker: str = "…[截断]",
    ) -> None:
        limits = (max_total_chars, max_evidence_chars, max_evidence_count)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in limits
        ):
            raise RetrievalValidationError("evidence context limits must be positive integers")
        if not isinstance(truncation_marker, str) or not truncation_marker:
            raise RetrievalValidationError("truncation marker must be a nonempty string")
        self.max_total_chars = max_total_chars
        self.max_evidence_chars = max_evidence_chars
        self.max_evidence_count = max_evidence_count
        self.truncation_marker = truncation_marker

    def build(self, parents: Sequence[ParentExpansionResult]) -> EvidenceContext:
        copied = self._validate_parents(parents)
        original_chars = sum(len(parent.parent_content) for parent in copied)
        items: list[EvidenceItem] = []
        references: list[EvidenceReference] = []
        blocks: list[str] = []
        included_chars = 0

        for parent in copied:
            if len(items) >= self.max_evidence_count:
                break
            evidence_id = f"E{len(items) + 1}"
            reference = self._reference(evidence_id, parent)
            header = self._header(reference)
            separator = "\n\n" if blocks else ""
            available = self.max_total_chars - len("".join(blocks)) - len(separator) - len(header)
            content = self._fit_content(parent.parent_content, available)
            if content is None:
                break
            truncated = content != parent.parent_content
            block = separator + header + content
            if len("".join(blocks)) + len(block) > self.max_total_chars:
                raise RetrievalValidationError("evidence context exceeded its total budget")
            blocks.append(block)
            included_chars += len(content)
            items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    final_rank=parent.final_rank,
                    parent_id=parent.parent_id,
                    document_id=parent.document_id,
                    content=content,
                    original_content_length=len(parent.parent_content),
                    included_content_length=len(content),
                    truncated=truncated,
                    matched_child_count=parent.matched_child_count,
                    best_child_rank=parent.best_child_rank,
                    metadata=parent.model_copy(deep=True).metadata,
                    provenance=parent.model_copy(deep=True).provenance,
                    matched_children=parent.model_copy(deep=True).matched_children,
                )
            )
            references.append(reference)

        rendered = "".join(blocks)
        return EvidenceContext(
            rendered_context=rendered,
            evidence_items=tuple(items),
            references=tuple(references),
            included_evidence_count=len(items),
            omitted_evidence_count=len(copied) - len(items),
            total_original_chars=original_chars,
            total_included_chars=included_chars,
            truncated=len(items) != len(copied) or any(item.truncated for item in items),
        )

    def _fit_content(self, content: str, available: int) -> str | None:
        budget = min(self.max_evidence_chars, available)
        if budget <= 0:
            return None
        if len(content) <= budget:
            return content
        if budget <= len(self.truncation_marker):
            return None
        return content[: budget - len(self.truncation_marker)] + self.truncation_marker

    @staticmethod
    def _reference(evidence_id: str, parent: ParentExpansionResult) -> EvidenceReference:
        source = parent.provenance.get("source")
        page = parent.metadata.get("page_number")
        start = parent.metadata.get("start_offset")
        end = parent.metadata.get("end_offset")
        return EvidenceReference(
            evidence_id=evidence_id,
            parent_id=parent.parent_id,
            document_id=parent.document_id,
            source=source if isinstance(source, str) and source.strip() else None,
            page_number=page if isinstance(page, int) and not isinstance(page, bool) else None,
            start_offset=start if isinstance(start, int) and not isinstance(start, bool) else None,
            end_offset=end if isinstance(end, int) and not isinstance(end, bool) else None,
        )

    @staticmethod
    def _header(reference: EvidenceReference) -> str:
        lines = [f"[{reference.evidence_id}]", f"文档ID：{reference.document_id}"]  # noqa: RUF001
        if reference.source is not None:
            lines.append(f"来源：{reference.source}")  # noqa: RUF001
        if reference.page_number is not None:
            lines.append(f"页码：{reference.page_number}")  # noqa: RUF001
        if reference.start_offset is not None and reference.end_offset is not None:
            lines.append(f"偏移：{reference.start_offset}-{reference.end_offset}")  # noqa: RUF001
        lines.append("内容：")  # noqa: RUF001
        return "\n".join(lines)

    @staticmethod
    def _validate_parents(parents: Sequence[ParentExpansionResult]) -> list[ParentExpansionResult]:
        if parents is None or isinstance(parents, (str, bytes)):
            raise RetrievalValidationError("evidence parents must be a sequence")
        try:
            copied = [
                ParentExpansionResult.model_validate(parent.model_dump(mode="python"))
                for parent in parents
            ]
        except (AttributeError, ValidationError) as exc:
            raise RetrievalValidationError("evidence parents are invalid") from exc
        if [parent.final_rank for parent in copied] != list(range(1, len(copied) + 1)):
            raise RetrievalValidationError("parent ranks must be consecutive and start at one")
        if len({parent.parent_id for parent in copied}) != len(copied):
            raise RetrievalValidationError("evidence parent IDs must be unique")
        for parent in copied:
            if not parent.parent_content.strip():
                raise RetrievalValidationError("parent evidence content cannot be blank")
            if parent.matched_child_count != len(parent.matched_children):
                raise RetrievalValidationError("matched child count is inconsistent")
        return copied
