"""Pipeline script to execute missingness feature calculation, pattern hashing, and registry logging."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import mlflow

from src.feature_engineering.missing import (
    VectorizedMissingEngine,
    AutomaticMissingDiscoveryEngine,
    MissingPatternBuilder,
    MissingValidationGate,
    MissingRegistry,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    logger.info("Executing Pre-Execution Pipeline Verification Gate...")

    train_inter_in = Path("data/feature_store_engineered/v1/train_interaction_features.parquet")
    test_inter_in = Path("data/feature_store_engineered/v1/test_interaction_features.parquet")
    interim_train = Path("data/interim/train_merged.parquet")
    interim_test = Path("data/interim/test_merged.parquet")

    missing_deps = []
    for path in [train_inter_in, test_inter_in, interim_train, interim_test]:
        if not path.exists():
            missing_deps.append(str(path))

    if missing_deps:
        msg = f"Dependency verification failed! Missing prior artifacts: {', '.join(missing_deps)}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Prior stage verification checks passed.")

    # Load inputs
    logger.info("Loading inputs...")
    df_train = pd.read_parquet(interim_train)
    df_test = pd.read_parquet(interim_test)

    df_miss_train = pd.DataFrame(index=df_train.index)
    df_miss_train["TransactionID"] = df_train["TransactionID"]

    df_miss_test = pd.DataFrame(index=df_test.index)
    df_miss_test["TransactionID"] = df_test["TransactionID"]

    # Discover features with missing values dynamically on training set
    discovery_engine = AutomaticMissingDiscoveryEngine(min_missing_ratio=0.05, max_missing_ratio=0.95)
    cols_with_missing = discovery_engine.discover_missing_columns(df_train)

    logger.info("Automatically discovered %d features with missing rates between 0.05 and 0.95", len(cols_with_missing))

    missing_engine = VectorizedMissingEngine()
    pattern_builder = MissingPatternBuilder()
    registry = MissingRegistry()
    validator = MissingValidationGate()

    # 1. Compute binary indicators for discovered features
    df_ind_train = missing_engine.compute_indicators(df_train, cols_with_missing)
    df_ind_test = missing_engine.compute_indicators(df_test, cols_with_missing)

    for col in df_ind_train.columns:
        df_miss_train[col] = df_ind_train[col]
        df_miss_test[col] = df_ind_test[col]
        
        # Register base column mapping
        base_col = col.replace("_isna", "")
        registry.register(col, base_col, "binary_indicator")

    # 2. Compute missing counts, ratios and completeness metrics
    df_stats_train = missing_engine.compute_row_stats(df_train, cols_with_missing)
    df_stats_test = missing_engine.compute_row_stats(df_test, cols_with_missing)

    for col in df_stats_train.columns:
        df_miss_train[col] = df_stats_train[col]
        df_miss_test[col] = df_stats_test[col]
        registry.register(col, "multiple_columns", f"row_missingness_{col}")

    # 3. Formulate binary missing pattern hashes for distinct logical feature groups
    # Group A: Identity columns (id_01 to id_38)
    id_cols = [f"id_{str(i).zfill(2)}" for i in range(1, 39)]
    # Group B: Card columns (card1 to card6)
    card_cols = [f"card{i}" for i in range(1, 7)]
    # Group C: Address columns (addr1, addr2)
    addr_cols = ["addr1", "addr2"]
    
    groups = {
        "identity_missing_pattern": id_cols,
        "card_missing_pattern": card_cols,
        "address_missing_pattern": addr_cols,
    }

    for pattern_name, base_cols in groups.items():
        logger.info("Computing missing pattern hash: %s", pattern_name)
        df_miss_train[pattern_name] = pattern_builder.build_pattern_hashes(df_train, base_cols)
        df_miss_test[pattern_name] = pattern_builder.build_pattern_hashes(df_test, base_cols)
        registry.register(pattern_name, ",".join(base_cols), "missingness_pattern_hash")

    # Validation checks
    val_train_report = validator.validate(df_miss_train)
    val_test_report = validator.validate(df_miss_test)

    # Save outputs
    store_dir = Path("data/feature_store_engineered/v1")
    store_dir.mkdir(parents=True, exist_ok=True)

    registry.save_catalog(store_dir)

    train_miss_out = store_dir / "train_missing_features.parquet"
    test_miss_out = store_dir / "test_missing_features.parquet"

    df_miss_train.to_parquet(train_miss_out, index=False)
    df_miss_test.to_parquet(test_miss_out, index=False)

    logger.info("Saved missing features to %s and %s", train_miss_out, test_miss_out)

    # MLflow Tracking
    logger.info("Logging missing feature metrics to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="feature_missing_pipeline")
        started = True

    try:
        mlflow.log_params({
            "pipeline_stage": "feature_missing",
            "version": "v1",
            "missing_features_count": len(registry.metadata),
            "train_validation_gate_status": val_train_report["status"],
            "test_validation_gate_status": val_test_report["status"],
        })
    except Exception as e:
        logger.warning("MLflow tracking logging encountered warning: %s", e)
    finally:
        if started:
            mlflow.end_run()

    logger.info("Feature missing pipeline completed successfully.")


if __name__ == "__main__":
    main()
