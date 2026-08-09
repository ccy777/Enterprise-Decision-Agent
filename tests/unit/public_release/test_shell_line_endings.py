"""Protect executable shell assets from Windows line-ending drift."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
READONLY_INITIALIZER = REPO_ROOT / "docker/mysql/init/03-create-readonly-user.sh"


def _shell_files() -> tuple[Path, ...]:
    shell_files = tuple(sorted(REPO_ROOT.rglob("*.sh")))
    assert shell_files, "the public repository must retain its shell initializer"
    return shell_files


def test_all_public_shell_scripts_have_no_crlf() -> None:
    offenders = [
        path.relative_to(REPO_ROOT) for path in _shell_files() if b"\r\n" in path.read_bytes()
    ]
    assert offenders == []


def test_all_public_shell_scripts_have_no_bare_cr() -> None:
    offenders = [
        path.relative_to(REPO_ROOT) for path in _shell_files() if b"\r" in path.read_bytes()
    ]
    assert offenders == []


def test_readonly_initializer_shebang_is_exact() -> None:
    first_line = READONLY_INITIALIZER.read_bytes().splitlines(keepends=True)[0]
    assert first_line == b"#!/bin/sh\n"


def test_public_gitattributes_forces_shell_lf() -> None:
    rules = {
        line.strip()
        for line in (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "*.sh text eol=lf" in rules


def test_readonly_initializer_is_plain_posix_text() -> None:
    content = READONLY_INITIALIZER.read_bytes()
    assert not content.startswith(b"\xef\xbb\xbf")
    assert b"\x00" not in content
    assert content.endswith(b"\n")
