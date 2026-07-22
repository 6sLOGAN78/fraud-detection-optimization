"""Pipeline orchestration script for Transaction Feature Analysis (Part 3.8).

Execution gate verifies all upstream DVC artifacts before running analysis.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Upstream Artifact Dependency Gate
# ---------------------------------------------------------------------------

_REQUIRED_ARTIFACTS: list[Path] = [
    Path("data/interim/train_merged.parquet"),
    Path("data/interim/test_merged.parquet"),
]


def _verify_dependencies() -> None:
    """Validates that all required upstream artifacts exist.

    Raises:
        SystemExit: If any expected artifact is missing.
    """
    missing = [str(p) for p in _REQUIRED_ARTIFACTS if not p.exists()]
    if missing:
        logger.error(
            "DEPENDENCY GATE FAILED. Missing required upstream artifacts:\n%s",
            "\n".join(f"  - {m}" for m in missing),
        )
        logger.error(
            "Run `dvc repro` to generate all preceding pipeline stages "
            "before executing transaction_analysis."
        )
        sys.exit(1)
    logger.info("Dependency gate passed — all upstream artifacts present.")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for transaction feature analysis pipeline stage."""
    _verify_dependencies()

    # Import here to avoid circular issues during dependency checks
    from src.eda.transaction import TransactionFeatureAnalyzer

    project_root = Path(__file__).resolve().parents[2]
    train_path = project_root / "data" / "interim" / "train_merged.parquet"
    test_path = project_root / "data" / "interim" / "test_merged.parquet"
    report_dir = project_root / "reports" / "eda" / "transaction"

    logger.info("Loading training dataset from %s", train_path)
    df_train = pd.read_parquet(train_path)

    logger.info("Loading test dataset from %s", test_path)
    df_test = pd.read_parquet(test_path)

    logger.info("Initializing Transaction Feature Analyzer...")
    analyzer = TransactionFeatureAnalyzer(
        df_train=df_train,
        df_test=df_test,
        target_col="isFraud",
    )

    analyzer.analyze_all(report_dir=report_dir)
    logger.info(
        "Transaction Feature Analysis pipeline stage completed. "
        "Reports saved to %s",
        report_dir,
    )


if __name__ == "__main__":
    main()
