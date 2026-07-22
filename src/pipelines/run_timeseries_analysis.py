"""Pipeline orchestration script for Part 3.10 Time Series Feature Analysis."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.eda.timeseries import TimeSeriesFeatureAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    # 1. Dependency Gate Check
    root_dir = Path(__file__).resolve().parents[2]
    train_path = root_dir / "data/interim/train_merged.parquet"
    test_path = root_dir / "data/interim/test_merged.parquet"
    report_dir = root_dir / "reports/eda/timeseries"

    missing_artifacts = []
    if not train_path.exists():
        missing_artifacts.append(str(train_path))
    if not test_path.exists():
        missing_artifacts.append(str(test_path))

    if missing_artifacts:
        logger.error("Dependency gate failed — missing upstream artifacts:")
        for art in missing_artifacts:
            logger.error("  Missing: %s", art)
        sys.exit(1)

    logger.info("Dependency gate passed — all upstream artifacts present.")

    # 2. Load Parquet Data
    logger.info("Loading training dataset from %s", train_path)
    df_train = pd.read_parquet(train_path, columns=["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"])

    logger.info("Loading test dataset from %s", test_path)
    df_test = pd.read_parquet(test_path, columns=["TransactionID", "TransactionDT", "TransactionAmt"])

    # 3. Execution of Analyzer
    logger.info("Initializing Time Series Feature Analyzer...")
    analyzer = TimeSeriesFeatureAnalyzer(df_train, df_test, target_col="isFraud")
    analyzer.analyze_all(report_dir=report_dir)

    logger.info("Time Series Feature Analysis pipeline stage completed. Reports saved to %s", report_dir)


if __name__ == "__main__":
    main()
