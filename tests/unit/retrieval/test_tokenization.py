"""Tests for deterministic Chinese and business-identifier tokenization."""

import pytest

from decision_agent.exceptions import RetrievalValidationError
from decision_agent.retrieval.tokenization import DeterministicChineseTokenizer, TextTokenizer


@pytest.fixture
def tokenizer() -> DeterministicChineseTokenizer:
    return DeterministicChineseTokenizer()


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_empty_or_whitespace_text_is_rejected(
    tokenizer: DeterministicChineseTokenizer, text: str
) -> None:
    with pytest.raises(RetrievalValidationError, match="empty or whitespace"):
        tokenizer.tokenize(text)


def test_punctuation_only_text_is_rejected(tokenizer: DeterministicChineseTokenizer) -> None:
    with pytest.raises(RetrievalValidationError, match="no searchable tokens"):
        tokenizer.tokenize(",.!?---")


def test_nfkc_normalizes_full_width_business_identifiers(
    tokenizer: DeterministicChineseTokenizer,
) -> None:
    assert tokenizer.tokenize("\uff31\uff12 \uff12\uff10\uff12\uff16") == ["q2", "2026"]


def test_english_is_lowercased(tokenizer: DeterministicChineseTokenizer) -> None:
    assert tokenizer.tokenize("Product WARRANTY") == ["product", "warranty"]


def test_numbers_are_preserved(tokenizer: DeterministicChineseTokenizer) -> None:
    assert tokenizer.tokenize("2026 5000") == ["2026", "5000"]


def test_quarter_identifiers_remain_distinct(tokenizer: DeterministicChineseTokenizer) -> None:
    assert tokenizer.tokenize("Q1 Q2") == ["q1", "q2"]


def test_product_identifiers_remain_distinct(tokenizer: DeterministicChineseTokenizer) -> None:
    tokens = tokenizer.tokenize("产品A 产品B")
    assert "a" in tokens
    assert "b" in tokens
    assert tokens.count("a") == 1
    assert tokens.count("b") == 1


def test_stable_ascii_business_identifier_is_one_token(
    tokenizer: DeterministicChineseTokenizer,
) -> None:
    assert tokenizer.tokenize("SKU-A_2026/Q2") == ["sku-a_2026/q2"]


def test_chinese_uses_ordered_unigrams_and_bigrams(
    tokenizer: DeterministicChineseTokenizer,
) -> None:
    assert tokenizer.tokenize("电池") == ["电", "池", "电池"]


def test_longer_chinese_run_preserves_deterministic_order(
    tokenizer: DeterministicChineseTokenizer,
) -> None:
    assert tokenizer.tokenize("华东区") == ["华", "东", "区", "华东", "东区"]


def test_repeated_calls_are_identical(tokenizer: DeterministicChineseTokenizer) -> None:
    text = "产品A在华东区2026年Q2销售"
    assert tokenizer.tokenize(text) == tokenizer.tokenize(text)


def test_tokens_never_contain_empty_strings(tokenizer: DeterministicChineseTokenizer) -> None:
    assert all(token for token in tokenizer.tokenize("产品 A / Q2"))


def test_tokenization_does_not_modify_input(tokenizer: DeterministicChineseTokenizer) -> None:
    original = "\uff30\uff52\uff4f\uff44\uff55\uff43\uff54 A 电池"
    tokenizer.tokenize(original)
    assert original == "\uff30\uff52\uff4f\uff44\uff55\uff43\uff54 A 电池"


def test_default_tokenizer_satisfies_protocol(tokenizer: DeterministicChineseTokenizer) -> None:
    assert isinstance(tokenizer, TextTokenizer)
