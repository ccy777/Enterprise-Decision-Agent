"""Pure reader for an already persisted Runtime preflight evidence envelope."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from decision_agent.application.runtime_preflight_supervisor import VerifiedEnvelope


def verify(evidence_dir: Path) -> tuple[VerifiedEnvelope | None, int]:
    try:
        payload = json.loads((evidence_dir / "verified-envelope.json").read_text(encoding="utf-8"))
        envelope = VerifiedEnvelope(**payload)
    except (OSError, TypeError, json.JSONDecodeError):
        return None, 20
    if envelope.evidence_status != "verified":
        return envelope, 20
    return envelope, 0 if envelope.runtime_status == "passed" else 10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    envelope, code = verify(arguments.evidence_dir)
    if envelope is not None:
        sys.stdout.write(envelope.safe_json() + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
