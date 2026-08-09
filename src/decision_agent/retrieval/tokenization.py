"""Deterministic tokenization for Chinese sparse retrieval."""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol, runtime_checkable

from decision_agent.exceptions import RetrievalValidationError

_TOKEN_PATTERN = re.compile(
    r"(?P<ascii>[a-z0-9]+(?:[-_./][a-z0-9]+)*)|"
    r"(?P<cjk>[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+)"
)


@runtime_checkable
class TextTokenizer(Protocol):
    """Convert text into an ordered list of searchable tokens."""

    def tokenize(self, text: str) -> list[str]:
        """Return deterministic tokens without modifying the caller's input."""
        ...


class DeterministicChineseTokenizer:
    """Tokenize ASCII identifiers and Chinese character unigrams/bigrams.

    Text is normalized with Unicode NFKC and lowercased. Contiguous ASCII
    alphanumeric identifiers may retain ``-``, ``_``, ``.``, and ``/`` when
    those separators join nonempty alphanumeric segments. Each contiguous CJK
    run emits its characters first and then its adjacent character bigrams.
    """

    name = "deterministic-chinese-char-ngram"
    version = "1.0"
    chinese_ngram_sizes = (1, 2)

    def tokenize(self, text: str) -> list[str]:
        """Return NFKC-normalized, lowercase, deterministic lexical tokens."""
        if not isinstance(text, str) or not text.strip():
            raise RetrievalValidationError("tokenizer input cannot be empty or whitespace")

        normalized = unicodedata.normalize("NFKC", text).lower()
        tokens: list[str] = []
        for match in _TOKEN_PATTERN.finditer(normalized):
            ascii_token = match.group("ascii")
            if ascii_token is not None:
                tokens.append(ascii_token)
                continue
            cjk_run = match.group("cjk")
            tokens.extend(cjk_run)
            tokens.extend(cjk_run[index : index + 2] for index in range(len(cjk_run) - 1))

        if not tokens:
            raise RetrievalValidationError("tokenizer input produced no searchable tokens")
        return tokens
