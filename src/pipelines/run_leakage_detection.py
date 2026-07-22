"""Pipeline script to run Data Leakage Detection — Part 3.16."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.eda.leakage import DataLeakageDetector

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

    # Define paths
    report_dir = Path("reports/eda/leakage")
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load parquets
    logger.info("Loading train and test parquets...")
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    # Initialize and run DataLeakageDetector
    logger.info("Initializing Data Leakage Detector...")
    detector = DataLeakageDetector(
        df_train=df_train,
        df_test=df_test,
    )

    logger.info("Executing Data Leakage Detection pipeline stages...")
    detector.analyze_all(report_dir=report_dir)

    logger.info("Data Leakage Detection pipeline stage completed successfully.")


if __name__ == "__main__":
    main()
