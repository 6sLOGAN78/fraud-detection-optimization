"""Pipeline orchestration script for Part 3.11 Anonymous Feature Analysis."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.eda.anonymous import AnonymousFeatureAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    # 1. Dependency Gate Check
    root_dir = Path(__file__).resolve().parents[2]
    train_path = root_dir / "data/interim/train_merged.parquet"
    test_path = root_dir / "data/interim/test_merged.parquet"
    report_dir = root_dir / "reports/eda/anonymous"

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

    # 2. Extract columns for anonymous analysis to optimize memory footprint
    logger.info("Scanning parquet schema for V, C, D, M anonymous features...")
    
    import pyarrow.parquet as pq
    
    # Read just the schema to get column names
    train_schema = pq.read_schema(train_path)
    all_cols = train_schema.names
    
    v_cols = [c for c in all_cols if c.startswith("V") and c[1:].isdigit()]
    c_cols = [c for c in all_cols if c.startswith("C") and c[1:].isdigit()]
    d_cols = [c for c in all_cols if c.startswith("D") and c[1:].isdigit()]
    m_cols = [c for c in all_cols if c.startswith("M") and c[1:].isdigit()]
    
    selected_cols = ["TransactionID", "isFraud"] + v_cols + c_cols + d_cols + m_cols
    selected_cols_test = ["TransactionID"] + v_cols + c_cols + d_cols + m_cols

    # 3. Load Parquet Data
    logger.info("Loading training dataset matching columns...")
    df_train = pd.read_parquet(train_path, columns=selected_cols)

    logger.info("Loading test dataset matching columns...")
    # Sieve columns that are present in test (some might be missing or different)
    test_schema = pq.read_schema(test_path)
    present_cols_test = [c for c in selected_cols_test if c in test_schema.names]
    df_test = pd.read_parquet(test_path, columns=present_cols_test)

    # 4. Execution of Analyzer
    logger.info("Initializing Anonymous Feature Analyzer...")
    analyzer = AnonymousFeatureAnalyzer(df_train, df_test, target_col="isFraud")
    analyzer.analyze_all(report_dir=report_dir)

    logger.info("Anonymous Feature Analysis pipeline stage completed. Reports saved to %s", report_dir)


if __name__ == "__main__":
    main()
