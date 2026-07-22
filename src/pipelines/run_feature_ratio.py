"""Pipeline script to execute ratio calculations matching numerical columns with aggregated metrics."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

from src.feature_engineering.ratios import (
    VectorizedRatioEngine,
    AutomaticRatioDiscoveryEngine,
    RatioValidationGate,
    RatioRegistry,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    logger.info("Executing Pre-Execution Pipeline Verification Gate...")

    train_agg_in = Path("data/feature_store_engineered/v1/train_aggregated_features.parquet")
    test_agg_in = Path("data/feature_store_engineered/v1/test_aggregated_features.parquet")
    interim_train = Path("data/interim/train_merged.parquet")
    interim_test = Path("data/interim/test_merged.parquet")

    missing = []
    for path in [train_agg_in, test_agg_in, interim_train, interim_test]:
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

    df_ratio_train = pd.DataFrame(index=df_train.index)
    df_ratio_train["TransactionID"] = df_train["TransactionID"]

    df_ratio_test = pd.DataFrame(index=df_test.index)
    df_ratio_test["TransactionID"] = df_test["TransactionID"]

    # Discover pairings
    discovery_engine = AutomaticRatioDiscoveryEngine(target_numerators=["TransactionAmt", "dist1"])
    pairings = discovery_engine.discover_pairings(df_train.columns.tolist())

    logger.info("Automatically discovered %d ratio pairings", len(pairings))

    ratio_engine = VectorizedRatioEngine(epsilon=1e-5, default_val=1.0)
    registry = RatioRegistry()
    validator = RatioValidationGate()

    for feat_name, numer_col, denom_col in pairings:
        logger.info("Computing ratio feature: %s = %s / %s", feat_name, numer_col, denom_col)
        
        df_ratio_train[feat_name] = ratio_engine.compute_ratio(df_train[numer_col], df_train[denom_col])
        df_ratio_test[feat_name] = ratio_engine.compute_ratio(df_test[numer_col], df_test[denom_col])
        registry.register(feat_name, numer_col, denom_col)

    # Validation checks
    val_train_report = validator.validate(df_ratio_train)
    val_test_report = validator.validate(df_ratio_test)

    # Save
    store_dir = Path("data/feature_store_engineered/v1")
    store_dir.mkdir(parents=True, exist_ok=True)
    
    registry.save_catalog(store_dir)
    
    train_ratio_out = store_dir / "train_ratio_features.parquet"
    test_ratio_out = store_dir / "test_ratio_features.parquet"
    
    df_ratio_train.to_parquet(train_ratio_out, index=False)
    df_ratio_test.to_parquet(test_ratio_out, index=False)
    
    logger.info("Saved ratio features to %s and %s", train_ratio_out, test_ratio_out)

    # MLflow Tracking
    logger.info("Logging ratio metrics to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="feature_ratio_pipeline")
        started = True

    try:
        mlflow.log_params({
            "pipeline_stage": "feature_ratio",
            "version": "v1",
            "ratios_count": len(registry.metadata),
            "train_validation_gate_status": val_train_report["status"],
            "test_validation_gate_status": val_test_report["status"],
        })
    except Exception as e:
        logger.warning("MLflow tracking logging encountered warning: %s", e)
    finally:
        if started:
            mlflow.end_run()

    logger.info("Feature ratio pipeline completed successfully.")


if __name__ == "__main__":
    main()
