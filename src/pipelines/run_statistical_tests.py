"""Pipeline script to run Statistical Tests — Part 3.14."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.eda.statistical_tests import StatisticalTestsAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    # 1. Dependency gate: check upstream merged parquet files exist
    logger.info("Executing Pre-Execution Pipeline Verification Gate...")

    train_path = Path("data/interim/train_merged.parquet")
    test_path = Path("data/interim/test_merged.parquet")

    if not train_path.exists() or not test_path.exists():
        logger.error(
            "Dependency verification failed! Missing intermediate datasets: train_merged=%s, test_merged=%s",
            train_path.exists(),
            test_path.exists(),
        )
        sys.exit(1)

    logger.info("Dependency gate passed — all upstream datasets are present.")

    # Load configuration
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        target_col = config.get("data", {}).get("target_col", "isFraud")
    else:
        target_col = "isFraud"

    # Define paths
    report_dir = Path("reports/eda/statistical_tests")
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load parquets
    logger.info("Loading train and test parquets...")
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    # Initialize and run StatisticalTestsAnalyzer
    logger.info("Initializing Statistical Tests Analyzer...")
    analyzer = StatisticalTestsAnalyzer(
        df_train=df_train,
        df_test=df_test,
        target_col=target_col,
    )

    logger.info("Executing Statistical Analysis pipeline stages...")
    analyzer.analyze_all(report_dir=report_dir)

    logger.info("Statistical Analysis pipeline stage completed successfully.")


if __name__ == "__main__":
    main()
