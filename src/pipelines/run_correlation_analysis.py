"""Pipeline orchestration script for Part 3.12 Correlation Analysis."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.eda.correlation import CorrelationAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    # 1. Dependency Gate Check
    root_dir = Path(__file__).resolve().parents[2]
    train_path = root_dir / "data/interim/train_merged.parquet"
    test_path = root_dir / "data/interim/test_merged.parquet"
    report_dir = root_dir / "reports/eda/correlation"

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

    # 2. Load data
    logger.info("Loading training and test datasets...")
    # Because correlation can run on many features, we load the merged intermediate files.
    # To keep memory usage low, we'll exclude device-specific high-cardinality text columns
    # but load all numeric/anonymous columns.
    
    import pyarrow.parquet as pq
    train_schema = pq.read_schema(train_path)
    ignore_sets = {"DeviceInfo", "DeviceType", "id_30", "id_31", "id_33"}
    load_cols = [c for c in train_schema.names if c not in ignore_sets]
    
    df_train = pd.read_parquet(train_path, columns=load_cols)
    
    test_schema = pq.read_schema(test_path)
    load_cols_test = [c for c in load_cols if c in test_schema.names]
    df_test = pd.read_parquet(test_path, columns=load_cols_test)

    # 3. Execution of Analyzer
    logger.info("Initializing Correlation Analyzer...")
    analyzer = CorrelationAnalyzer(
        df_train=df_train,
        df_test=df_test,
        target_col="isFraud",
        threshold=0.90,
    )
    analyzer.analyze_all(report_dir=report_dir)

    logger.info("Correlation Analysis pipeline stage completed. Reports saved to %s", report_dir)


if __name__ == "__main__":
    main()
