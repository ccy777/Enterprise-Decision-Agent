"""Validate that every declared dependency is represented by a portable exact pin."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==[^;\s]+(?:\s*;\s*.+)?$")
_LOCAL_PATH = re.compile(r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]", re.I)


def validate_lock(pyproject_path: Path, lock_path: Path) -> None:
    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    declared = [*project["project"]["dependencies"]]
    for dependencies in project["project"].get("optional-dependencies", {}).values():
        declared.extend(dependencies)
    required = {_dependency_name(item) for item in declared}
    text = lock_path.read_text(encoding="utf-8")
    if _LOCAL_PATH.search(text):
        raise ValueError("lock file contains a local user path")
    pins: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        match = _PIN.fullmatch(line)
        if match is None:
            raise ValueError(f"lock entry is not an exact pin: {line}")
        pins.add(_normalize_name(match.group(1)))
    missing = sorted(required - pins)
    if missing:
        raise ValueError(f"declared dependencies missing from lock: {', '.join(missing)}")


def _dependency_name(requirement: str) -> str:
    match = _NAME.match(requirement)
    if match is None:
        raise ValueError("invalid dependency declaration")
    return _normalize_name(match.group(1))


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    validate_lock(root / "pyproject.toml", root / "requirements.lock")
    print("dependency lock validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
