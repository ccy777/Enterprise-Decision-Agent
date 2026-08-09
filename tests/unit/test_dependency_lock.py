# ruff: noqa: I001

from pathlib import Path

import pytest

from scripts.validate_dependency_lock import validate_lock


_ROOT = Path(__file__).resolve().parents[2]


def test_formal_dependency_lock_matches_pyproject() -> None:
    validate_lock(_ROOT / "pyproject.toml", _ROOT / "requirements.lock")


def test_lock_validation_rejects_local_paths_and_unpinned_entries(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname="sample"\nversion="1"\ndependencies=["demo>=1"]\n',
        encoding="utf-8",
    )
    lock = tmp_path / "requirements.lock"
    lock.write_text("demo>=1\n# C:\\Users\\local\\cache\n", encoding="utf-8")

    with pytest.raises(ValueError):
        validate_lock(pyproject, lock)
