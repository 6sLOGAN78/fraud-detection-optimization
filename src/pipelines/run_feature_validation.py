"""Pipeline script to execute comprehensive feature validation gates, check constraints, and log data health metrics."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import mlflow

from src.feature_engineering.validation import FeatureValidationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    logger.info("Executing Pre-Execution Pipeline Verification Gate...")

    train_miss_in = Path("data/feature_store_engineered/v1/train_missing_features.parquet")
    test_miss_in = Path("data/feature_store_engineered/v1/test_missing_features.parquet")

    missing_deps = []
    for path in [train_miss_in, test_miss_in]:
        if not path.exists():
            missing_deps.append(str(path))

    if missing_deps:
        msg = f"Dependency verification failed! Missing prior artifacts: {', '.join(missing_deps)}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Prior stage verification checks passed.")

    # Load inputs
    logger.info("Loading inputs...")
    df_train = pd.read_parquet(train_miss_in)
    df_test = pd.read_parquet(test_miss_in)

    # Establish validation configurations
    expected_types = {
        "TransactionID": "numeric",
        "missing_count": "numeric",
        "missing_ratio": "numeric",
        "completeness_score": "numeric",
        "identity_missing_pattern": "numeric",
        "card_missing_pattern": "numeric",
        "address_missing_pattern": "numeric",
    }
    
    # All columns ending in _isna are expected to be numeric
    for col in df_train.columns:
        if col.endswith("_isna"):
            expected_types[col] = "numeric"

    range_bounds = {
        "missing_ratio": (0.0, 1.0),
        "completeness_score": (0.0, 1.0),
        "missing_count": (0.0, 1000.0),
    }

    # Engineered missingness metrics themselves MUST have 0% missing values
    missing_thresholds = {col: 0.00 for col in df_train.columns if col != "TransactionID"}

    validator_pn = FeatureValidationPipeline()

    logger.info("Validating train missingness features...")
    train_report = validator_pn.run_validation(
        df_train,
        expected_types=expected_types,
        range_bounds=range_bounds,
        missing_thresholds=missing_thresholds,
    )

    logger.info("Validating test missingness features...")
    test_report = validator_pn.run_validation(
        df_test,
        expected_types=expected_types,
        range_bounds=range_bounds,
        missing_thresholds=missing_thresholds,
    )

    # Save validation reports
    store_dir = Path("data/feature_store_engineered/v1")
    train_report_path = store_dir / "train_validation_report.json"
    test_report_path = store_dir / "test_validation_report.json"

    validator_pn.save_report(train_report, train_report_path)
    validator_pn.save_report(test_report, test_report_path)

    # MLflow Tracking
    logger.info("Logging validation metrics to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="feature_validation_pipeline")
        started = True

    try:
        mlflow.log_params({
            "pipeline_stage": "feature_validation",
            "version": "v1",
            "train_overall_status": train_report["overall_status"],
            "test_overall_status": test_report["overall_status"],
            "train_schema_status": train_report["schema_validation"]["status"],
            "train_missingness_status": train_report["missingness_validation"]["status"],
            "train_statistical_status": train_report["statistical_validation"]["status"],
        })
    except Exception as e:
        logger.warning("MLflow tracking logging encountered warning: %s", e)
    finally:
        if started:
            mlflow.end_run()

    # Raise exception if vital schema validation fails
    if train_report["schema_validation"]["status"] == "FAIL" or test_report["schema_validation"]["status"] == "FAIL":
        msg = "Feature validation pipeline failed due to schema integrity mismatches!"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Feature validation pipeline completed successfully.")


if __name__ == "__main__":
    main()
