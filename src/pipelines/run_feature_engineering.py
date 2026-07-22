"""Pipeline script to run Feature Engineering — Part 4.1."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import mlflow

from src.feature_engineering.pipeline import FeatureEngineeringPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    # 1. Pipeline Verification Gate
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

    # Load parquets
    logger.info("Loading train and test parquet datasets...")
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    # Initialize and execute pipeline
    logger.info("Initializing Feature Engineering Pipeline...")
    pipeline = FeatureEngineeringPipeline()

    # Track within active MLflow run if any
    logger.info("Executing feature engineering fit_transform operations...")
    pipeline.fit_transform(df_train, df_test, version="v1")

    logger.info("Feature Engineering pipeline completed successfully.")


if __name__ == "__main__":
    main()
