"""Pipeline script to execute feature calculations, fitting summary statistics, rolling windows, and registries."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

from src.feature_engineering.aggregations import (
    AggregationGroupBuilder,
    VectorizedAggregationEngine,
    RollingAggregationEngine,
    AggregationValidationGate,
    AggregationRegistry,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    logger.info("Executing Pre-Execution Pipeline Verification Gate...")

    train_in = Path("data/feature_store_engineered/v1/train_encoded_features.parquet")
    test_in = Path("data/feature_store_engineered/v1/test_encoded_features.parquet")
    interim_train = Path("data/interim/train_merged.parquet")
    interim_test = Path("data/interim/test_merged.parquet")

    missing = []
    for path in [train_in, test_in, interim_train, interim_test]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        msg = f"Dependency verification failed! Missing prior artifacts: {', '.join(missing)}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Prior stage verification checks passed.")

    # Load inputs
    logger.info("Loading inputs...")
    df_train_encoded = pd.read_parquet(train_in)
    df_test_encoded = pd.read_parquet(test_in)

    # Let's rebuild the group columns and aggregate values from the interim dataset
    # We aggregate TransactionAmt and dist1
    agg_cols = ["TransactionAmt", "dist1"]
    
    builder = AggregationGroupBuilder()
    group_keys = builder.get_group_keys()

    # Load necessary raw columns from interim
    req_cols = ["TransactionID"] + group_keys + [c for c in agg_cols]
    
    df_train_raw = pd.read_parquet(interim_train, columns=[c for c in req_cols if c in pd.read_parquet(interim_train).columns])
    df_test_raw = pd.read_parquet(interim_test, columns=[c for c in req_cols if c in pd.read_parquet(interim_test).columns])

    # Merge
    df_train = pd.merge(df_train_encoded, df_train_raw, on="TransactionID", how="left")
    df_test = pd.merge(df_test_encoded, df_test_raw, on="TransactionID", how="left")

    df_agg_train = pd.DataFrame(index=df_train.index)
    df_agg_train["TransactionID"] = df_train["TransactionID"]

    df_agg_test = pd.DataFrame(index=df_test.index)
    df_agg_test["TransactionID"] = df_test["TransactionID"]

    registry = AggregationRegistry()
    validator = AggregationValidationGate()

    # Run summary statistics aggregations
    # To reduce size and complexity, we can aggregate TransactionAmt by card1, addr1, ProductCD, P_emaildomain
    pairs = [
        ("card1", "TransactionAmt"),
        ("addr1", "TransactionAmt"),
        ("ProductCD", "TransactionAmt"),
        ("P_emaildomain", "TransactionAmt"),
        ("card1", "dist1"),
    ]

    for group_col, agg_col in pairs:
        if group_col not in df_train.columns or agg_col not in df_train.columns:
            logger.warning("Feature %s or agg column %s missing; skipping", group_col, agg_col)
            continue
            
        logger.info("Aggregating %s grouped by %s", agg_col, group_col)
        engine = VectorizedAggregationEngine(group_col, agg_col)
        engine.fit(df_train)

        for stat in ["mean", "median", "std", "min", "max", "count"]:
            feat_name = f"{group_col}_{agg_col}_{stat}"
            df_agg_train[feat_name] = engine.transform(df_train, stat_type=stat)
            df_agg_test[feat_name] = engine.transform(df_test, stat_type=stat)
            registry.register(feat_name, stat, agg_col, group_col)

    # Run rolling integrations (temporal expanding / rolling window)
    rolling_pairs = [
        ("card1", "TransactionAmt"),
        ("addr1", "TransactionAmt"),
    ]
    
    # Merge train and test temporarily to compute chronological rolling averages across entire history, 
    # but make sure we sort and filter correctly to avoid target lookahead validation leaks.
    df_all = pd.concat([df_train[["TransactionID", "card1", "addr1", "TransactionAmt"]], 
                        df_test[["TransactionID", "card1", "addr1", "TransactionAmt"]]], 
                       ignore_index=True)
    df_all = df_all.sort_values("TransactionID").reset_index(drop=True)

    for group_col, agg_col in rolling_pairs:
        if group_col not in df_all.columns or agg_col not in df_all.columns:
            continue
            
        logger.info("Calculating chronological rolling stats for %s by %s", agg_col, group_col)
        r_engine = RollingAggregationEngine(group_col, agg_col, window_size=5)
        rolling_df = r_engine.compute_rolling(df_all)
        
        # Merge back to train/test via TransactionID index
        rolling_df["TransactionID"] = df_all["TransactionID"]
        
        # Split back
        train_roll = pd.merge(df_train[["TransactionID"]], rolling_df, on="TransactionID", how="left").drop(columns=["TransactionID"])
        test_roll = pd.merge(df_test[["TransactionID"]], rolling_df, on="TransactionID", how="left").drop(columns=["TransactionID"])
        
        for col in train_roll.columns:
            df_agg_train[col] = train_roll[col]
            df_agg_test[col] = test_roll[col]
            registry.register(col, "rolling", agg_col, group_col, window="5-transactions")

    # Validation checks
    val_train_report = validator.validate(df_agg_train)
    val_test_report = validator.validate(df_agg_test)

    # Save
    store_dir = Path("data/feature_store_engineered/v1")
    store_dir.mkdir(parents=True, exist_ok=True)
    
    registry.save_catalog(store_dir)
    
    train_agg_out = store_dir / "train_aggregated_features.parquet"
    test_agg_out = store_dir / "test_aggregated_features.parquet"
    
    df_agg_train.to_parquet(train_agg_out, index=False)
    df_agg_test.to_parquet(test_agg_out, index=False)
    
    logger.info("Saved aggregation features to %s and %s", train_agg_out, test_agg_out)

    # MLflow Tracking
    logger.info("Logging aggregation metrics to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="feature_aggregation_pipeline")
        started = True

    try:
        mlflow.log_params({
            "pipeline_stage": "feature_aggregation",
            "version": "v1",
            "aggregations_count": len(registry.metadata),
            "train_validation_gate_status": val_train_report["status"],
            "test_validation_gate_status": val_test_report["status"],
        })
    except Exception as e:
        logger.warning("MLflow tracking logging encountered warning: %s", e)
    finally:
        if started:
            mlflow.end_run()

    logger.info("Feature aggregation pipeline completed successfully.")


if __name__ == "__main__":
    main()
