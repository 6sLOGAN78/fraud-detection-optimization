"""Pipeline orchestration script for running Target Variable Analysis."""

import logging
from pathlib import Path

import pandas as pd

from src.eda.target import TargetVariableAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Orchestrates overall class distribution, imbalance, and time behaviors."""
    logger.info("Initializing Target Variable Analysis pipeline stage...")

    # Load merged data outputs
    train_path = Path("data/interim/train_merged.parquet")
    test_path = Path("data/interim/test_merged.parquet")

    if not train_path.exists():
        raise FileNotFoundError(
            f"Required train dataset {train_path} does not exist."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"Required test dataset {test_path} does not exist."
        )

    logger.info("Loading training dataset from %s", train_path)
    df_train = pd.read_parquet(train_path)

    logger.info("Loading testing dataset from %s", test_path)
    df_test = pd.read_parquet(test_path)

    report_dir = Path("reports/eda/target")
    report_dir.mkdir(parents=True, exist_ok=True)

    analyzer = TargetVariableAnalyzer(
        df_train=df_train,
        df_test=df_test,
    )

    logger.info("Running Target Variable Analysis diagnostics...")
    analyzer.analyze_all(report_dir=report_dir)

    logger.info("Target Variable Analysis pipeline stage completed.")


if __name__ == "__main__":
    main()
