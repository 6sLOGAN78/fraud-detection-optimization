"""Pipeline script to execute comprehensive feature validation gates, drift metrics, target leakage checks, redundancy, and log data health metrics."""

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
    interim_train = Path("data/interim/train_merged.parquet")
    interim_test = Path("data/interim/test_merged.parquet")

    missing_deps = []
    for path in [train_miss_in, test_miss_in, interim_train, interim_test]:
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
    
    # Load targets and reference datasets for drift/leakage/importance calculations
    df_orig_train = pd.read_parquet(interim_train)
    
    target = None
    if "isFraud" in df_orig_train.columns:
        target = df_orig_train["isFraud"]

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

    logger.info("Validating train missingness features with drift, leakage, redundancy, and surrogate importances...")
    train_report = validator_pn.run_validation(
        df_train,
        expected_types=expected_types,
        range_bounds=range_bounds,
        missing_thresholds=missing_thresholds,
        df_ref=df_train,       # Baseline reference for train selfcheck (PSI is 0.0)
        target=target,
        drift_threshold=0.25,
        leakage_threshold=0.90,
        redundancy_threshold=0.85,
        importance_threshold=0.01,
    )

    logger.info("Validating test missingness features with drift checks against training set...")
    test_report = validator_pn.run_validation(
        df_test,
        expected_types=expected_types,
        range_bounds=range_bounds,
        missing_thresholds=missing_thresholds,
        df_ref=df_train,       # Detect drift of test features relative to train features
        target=None,          # Test target is unobserved
        drift_threshold=0.25,
        leakage_threshold=0.90,
        redundancy_threshold=0.85,
        importance_threshold=0.01,
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
            "version": "v1.1",
            "train_overall_status": train_report["overall_status"],
            "test_overall_status": test_report["overall_status"],
            "train_schema_status": train_report["schema_validation"]["status"],
            "train_drift_status": train_report["drift_validation"]["status"],
            "test_drift_status": test_report["drift_validation"]["status"],
            "train_leakage_status": train_report["leakage_validation"]["status"],
            "train_redundancy_status": train_report["correlation_validation"]["status"],
            "train_importance_status": train_report["importance_validation"]["status"],
        })
        
        # Log numeric metrics
        mlflow.log_metrics({
            "train_drifted_columns_count": train_report["drift_validation"].get("drifted_columns_count", 0),
            "test_drifted_columns_count": test_report["drift_validation"].get("drifted_columns_count", 0),
            "train_leakage_columns_count": train_report["leakage_validation"].get("leakage_columns_count", 0),
            "train_redundancies_count": train_report["correlation_validation"].get("redundancies_count", 0),
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
