"""Execute the 28 case-bound security checks and write their safe observations."""

from __future__ import annotations

import argparse
from pathlib import Path

from decision_agent.evaluation.security_evaluation import (
    evaluate_security_cases,
    load_security_cases,
    write_security_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("datasets/security/m8c_d_security_cases.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = evaluate_security_cases(load_security_cases(arguments.cases))
    write_security_report(report, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
