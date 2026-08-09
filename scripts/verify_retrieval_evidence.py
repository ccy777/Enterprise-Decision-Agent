"""Verify the committed M2C2-A retrieval evidence without loading models."""

from __future__ import annotations

import json
from pathlib import Path

from decision_agent.evaluation.retrieval_evidence import verify_retrieval_evidence


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    result = verify_retrieval_evidence(repository)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
