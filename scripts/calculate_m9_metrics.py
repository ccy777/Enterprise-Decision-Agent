"""Strictly derive v1.0.2 M9 metrics from the immutable redacted run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from decision_agent.evaluation.final_system_evaluation import (
    derive_final_metrics_from_frozen_sources,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--manifest-output", type=Path, required=True)
    arguments = parser.parse_args()
    source_manifest = arguments.source_manifest or arguments.records.parent / "manifest.json"
    report, manifest = derive_final_metrics_from_frozen_sources(
        records_path=arguments.records,
        dataset_path=arguments.dataset,
        adjudications_path=arguments.adjudications,
        source_manifest_path=source_manifest,
    )
    rendered = report.model_dump_json(indent=2) + "\n"
    manifest["derived_metrics_sha256"] = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered, encoding="utf-8")
    arguments.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
