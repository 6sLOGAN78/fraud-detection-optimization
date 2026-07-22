"""Pipeline script to execute interaction calculations, feature explosion screening, and registry recording."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import mlflow

from src.feature_engineering.interactions import (
    VectorizedInteractionEngine,
    AutomaticInteractionDiscoveryEngine,
    FeatureExplosionController,
    InteractionValidationGate,
    InteractionRegistry,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    logger.info("Executing Pre-Execution Pipeline Verification Gate...")

    train_agg_in = Path("data/feature_store_engineered/v1/train_aggregated_features.parquet")
    test_agg_in = Path("data/feature_store_engineered/v1/test_aggregated_features.parquet")
    train_diff_in = Path("data/feature_store_engineered/v1/train_difference_features.parquet")
    test_diff_in = Path("data/feature_store_engineered/v1/test_difference_features.parquet")
    interim_train = Path("data/interim/train_merged.parquet")
    interim_test = Path("data/interim/test_merged.parquet")

    missing = []
    for path in [train_agg_in, test_agg_in, train_diff_in, test_diff_in, interim_train, interim_test]:
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

    df_inter_train = pd.DataFrame(index=df_train.index)
    df_inter_train["TransactionID"] = df_train["TransactionID"]

    df_inter_test = pd.DataFrame(index=df_test.index)
    df_inter_test["TransactionID"] = df_test["TransactionID"]

    # Discover pairings
    discovery_engine = AutomaticInteractionDiscoveryEngine(target_cols=["TransactionAmt", "dist1"])
    pairings = discovery_engine.discover_pairings(df_train.columns.tolist())

    logger.info("Automatically discovered %d interaction pairings", len(pairings))

    inter_engine = VectorizedInteractionEngine(default_val=0.0)
    registry = InteractionRegistry()
    validator = InteractionValidationGate()
    explorer = FeatureExplosionController(variance_threshold=0.01)

    # Compute discovered candidates
    temp_train = pd.DataFrame(index=df_train.index)
    temp_test = pd.DataFrame(index=df_test.index)
    
    candidate_meta = []
    for feat_name, col_a, col_b, op in pairings:
        logger.info("Computing interaction candidate: %s = %s %s %s", feat_name, col_a, op, col_b)
        temp_train[feat_name] = inter_engine.compute_interaction(df_train[col_a], df_train[col_b], op)
        temp_test[feat_name] = inter_engine.compute_interaction(df_test[col_a], df_test[col_b], op)
        candidate_meta.append((feat_name, col_a, col_b, op))

    # Apply Explosion Control screening on train variance
    valid_cols = explorer.filter_features(temp_train)
    logger.info("Pruned candidate list from %d down to %d using variance screening", len(candidate_meta), len(valid_cols) - 1)

    for feat_name, col_a, col_b, op in candidate_meta:
        if feat_name in valid_cols:
            df_inter_train[feat_name] = temp_train[feat_name]
            df_inter_test[feat_name] = temp_test[feat_name]
            registry.register(feat_name, col_a, col_b, op)

    # Validation checks
    val_train_report = validator.validate(df_inter_train)
    val_test_report = validator.validate(df_inter_test)

    # Save
    store_dir = Path("data/feature_store_engineered/v1")
    store_dir.mkdir(parents=True, exist_ok=True)
    
    registry.save_catalog(store_dir)
    
    train_inter_out = store_dir / "train_interaction_features.parquet"
    test_inter_out = store_dir / "test_interaction_features.parquet"
    
    df_inter_train.to_parquet(train_inter_out, index=False)
    df_inter_test.to_parquet(test_inter_out, index=False)
    
    logger.info("Saved interaction features to %s and %s", train_inter_out, test_inter_out)

    # MLflow Tracking
    logger.info("Logging interaction metrics to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="feature_interaction_pipeline")
        started = True

    try:
        mlflow.log_params({
            "pipeline_stage": "feature_interaction",
            "version": "v1",
            "interactions_count": len(registry.metadata),
            "train_validation_gate_status": val_train_report["status"],
            "test_validation_gate_status": val_test_report["status"],
        })
    except Exception as e:
        logger.warning("MLflow tracking logging encountered warning: %s", e)
    finally:
        if started:
            mlflow.end_run()

    logger.info("Feature interaction pipeline completed successfully.")


if __name__ == "__main__":
    main()
