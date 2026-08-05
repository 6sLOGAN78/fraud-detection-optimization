"""Pipeline script to execute Part 15 — System Documentation Strategy & ADR Validation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.utils.docs_generator import DocumentationStrategyGate, DocumentationSuiteGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 15 Documentation Validation Pipeline")
    args = parser.parse_args()

    gate = DocumentationStrategyGate()
    gate.verify()

    generator = DocumentationSuiteGenerator()
    report = generator.validate_documentation_suite(".")

    out_file = Path("reports/docs/documentation_suite_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Part 15 Documentation Validation Pipeline completed. Report saved to {out_file}")


if __name__ == "__main__":
    main()
