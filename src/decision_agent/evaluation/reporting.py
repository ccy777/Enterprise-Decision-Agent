"""Shared safe report-writing utilities for offline evaluations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def _stage_text_file(output_path: Path, content: str) -> Path:
    """Write and sync one same-directory temporary file without replacing its target."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write(content)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    return temporary_path


def write_text_atomically(output_path: Path, content: str) -> None:
    """Replace a UTF-8 text file only after its complete content reaches disk."""
    temporary_path: Path | None = None
    try:
        temporary_path = _stage_text_file(output_path, content)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_text_files_atomically(files: dict[Path, str]) -> None:
    """Replace a small related set of text files with rollback on commit failure."""
    if not files:
        return
    if len(files) != len({path.resolve() for path in files}):
        raise ValueError("output paths must be unique")

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    installed: set[Path] = set()
    try:
        for output_path, content in files.items():
            staged[output_path] = _stage_text_file(output_path, content)

        for output_path in files:
            if output_path.exists():
                with NamedTemporaryFile(
                    dir=output_path.parent,
                    prefix=f".{output_path.name}.",
                    suffix=".backup",
                    delete=False,
                ) as backup_file:
                    backup_path = Path(backup_file.name)
                backup_path.unlink()
                os.replace(output_path, backup_path)
                backups[output_path] = backup_path

        for output_path, temporary_path in staged.items():
            os.replace(temporary_path, output_path)
            installed.add(output_path)
        staged.clear()
    except BaseException:
        for output_path in installed:
            output_path.unlink(missing_ok=True)
        for output_path, backup_path in backups.items():
            if backup_path.exists():
                os.replace(backup_path, output_path)
        raise
    finally:
        for temporary_path in staged.values():
            temporary_path.unlink(missing_ok=True)
        for backup_path in backups.values():
            backup_path.unlink(missing_ok=True)


def write_json_report_atomically(output_path: Path, payload: dict[str, Any]) -> None:
    """Replace a report only after complete UTF-8 JSON reaches disk."""
    write_text_atomically(
        output_path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
