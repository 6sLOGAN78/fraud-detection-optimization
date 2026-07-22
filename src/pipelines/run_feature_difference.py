"""Pipeline script to execute difference calculations matching numerical columns with aggregated metrics."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

from src.feature_engineering.differences import (
    VectorizedDifferenceEngine,
    AutomaticDifferenceDiscoveryEngine,
    DifferenceValidationGate,
    DifferenceRegistry,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    logger.info("Executing Pre-Execution Pipeline Verification Gate...")

    train_agg_in = Path("data/feature_store_engineered/v1/train_aggregated_features.parquet")
    test_agg_in = Path("data/feature_store_engineered/v1/test_aggregated_features.parquet")
    train_ratio_in = Path("data/feature_store_engineered/v1/train_ratio_features.parquet")
    test_ratio_in = Path("data/feature_store_engineered/v1/test_ratio_features.parquet")
    interim_train = Path("data/interim/train_merged.parquet")
    interim_test = Path("data/interim/test_merged.parquet")

    missing = []
    for path in [train_agg_in, test_agg_in, train_ratio_in, test_ratio_in, interim_train, interim_test]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        msg = f"Dependency verification failed! Missing prior artifacts: {', '.join(missing)}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Prior stage verification checks passed.")

    # Load inputs
    logger.info("Loading inputs...")
    df_train_agg = pd.read_parquet(train_agg_in)
    df_test_agg = pd.read_parquet(test_agg_in)

    # Load numerical metricsTransactionAmt and dist1 from interim
    req_cols = ["TransactionID", "TransactionAmt", "dist1"]
    
    df_train_raw = pd.read_parquet(interim_train, columns=[c for c in req_cols if c in pd.read_parquet(interim_train).columns])
    df_test_raw = pd.read_parquet(interim_test, columns=[c for c in req_cols if c in pd.read_parquet(interim_test).columns])

    # Merge
    df_train = pd.merge(df_train_agg, df_train_raw, on="TransactionID", how="left")
    df_test = pd.merge(df_test_agg, df_test_raw, on="TransactionID", how="left")

    df_diff_train = pd.DataFrame(index=df_train.index)
    df_diff_train["TransactionID"] = df_train["TransactionID"]

    df_diff_test = pd.DataFrame(index=df_test.index)
    df_diff_test["TransactionID"] = df_test["TransactionID"]

    # Discover pairings
    discovery_engine = AutomaticDifferenceDiscoveryEngine(target_numerators=["TransactionAmt", "dist1"])
    pairings = discovery_engine.discover_pairings(df_train.columns.tolist())

    logger.info("Automatically discovered %d difference pairings", len(pairings))

    diff_engine = VectorizedDifferenceEngine(default_val=0.0)
    registry = DifferenceRegistry()
    validator = DifferenceValidationGate()

    for feat_name, numer_col, denom_col in pairings:
        logger.info("Computing difference feature: %s = %s - %s", feat_name, numer_col, denom_col)
        
        df_diff_train[feat_name] = diff_engine.compute_difference(df_train[numer_col], df_train[denom_col])
        df_diff_test[feat_name] = diff_engine.compute_difference(df_test[numer_col], df_test[denom_col])
        registry.register(feat_name, numer_col, denom_col)

    # Validation checks
    val_train_report = validator.validate(df_diff_train)
    val_test_report = validator.validate(df_diff_test)

    # Save
    store_dir = Path("data/feature_store_engineered/v1")
    store_dir.mkdir(parents=True, exist_ok=True)
    
    registry.save_catalog(store_dir)
    
    train_diff_out = store_dir / "train_difference_features.parquet"
    test_diff_out = store_dir / "test_difference_features.parquet"
    
    df_diff_train.to_parquet(train_diff_out, index=False)
    df_diff_test.to_parquet(test_diff_out, index=False)
    
    logger.info("Saved difference features to %s and %s", train_diff_out, test_diff_out)

    # MLflow Tracking
    logger.info("Logging difference metrics to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="feature_difference_pipeline")
        started = True

    try:
        mlflow.log_params({
            "pipeline_stage": "feature_difference",
            "version": "v1",
            "differences_count": len(registry.metadata),
            "train_validation_gate_status": val_train_report["status"],
            "test_validation_gate_status": val_test_report["status"],
        })
    except Exception as e:
        logger.warning("MLflow tracking logging encountered warning: %s", e)
    finally:
        if started:
            mlflow.end_run()

    logger.info("Feature difference pipeline completed successfully.")


if __name__ == "__main__":
    main()
