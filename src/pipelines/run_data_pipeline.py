"""Orchestrator pipeline executing all 11 stages of the Data Engineering Pipeline."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.data.ingestion import load_dataset
from src.data.metadata import generate_metadata_reports
from src.data.schema import generate_schema, save_schema, validate_schema
from src.data.store import register_features_to_store, save_processed_dataset
from src.monitoring.drift import generate_drift_report
from src.preprocessing.cleaning import (
    clean_categorical_columns,
    impute_numerical_columns,
)
from src.preprocessing.leakage import run_leakage_checks
from src.preprocessing.memory import optimize_memory
from src.preprocessing.merge import merge_datasets
from src.preprocessing.quality import run_quality_checks
from src.utils.logging import setup_logger
from src.utils.mlflow_helper import MLflowTracker

logger = setup_logger("data_pipeline")


def load_config() -> Any:
    """Loads default Hydra configuration file with defaults composed."""
    config_dir = Path("configs").resolve()
    if not GlobalHydra.instance().is_initialized():
        hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3")
    return hydra.compose(config_name="config")


def run_load_data(config: Any, max_rows: int | None = None) -> None:
    """Stage 1: Load and validation raw data."""
    logger.info("--- Stage 1: Load Data ---")
    raw_dir = Path(config.paths.raw_dir)
    interim_dir = Path(config.paths.interim_dir)

    train_trans = load_dataset(raw_dir / config.data.train_transaction_file)
    train_ident = load_dataset(raw_dir / config.data.train_identity_file)
    test_trans = load_dataset(raw_dir / config.data.test_transaction_file)
    test_ident = load_dataset(raw_dir / config.data.test_identity_file)

    if max_rows:
        logger.info("Sampling dataset to %d rows for fast run.", max_rows)
        train_trans = train_trans.head(max_rows)
        train_ident = train_ident[
            train_ident["TransactionID"].isin(train_trans["TransactionID"])
        ]
        test_trans = test_trans.head(max_rows)
        test_ident = test_ident[
            test_ident["TransactionID"].isin(test_trans["TransactionID"])
        ]

    # Save as interim parquet files
    interim_dir.mkdir(parents=True, exist_ok=True)
    train_trans.to_parquet(interim_dir / "train_transaction.parquet", index=False)
    train_ident.to_parquet(interim_dir / "train_identity.parquet", index=False)
    test_trans.to_parquet(interim_dir / "test_transaction.parquet", index=False)
    test_ident.to_parquet(interim_dir / "test_identity.parquet", index=False)
    logger.info("Stage 1 completed successfully.")


def run_validate_schema(config: Any) -> None:
    """Stage 2: Schema extraction and validation."""
    logger.info("--- Stage 2: Validate Schema ---")
    interim_dir = Path(config.paths.interim_dir)
    metadata_dir = Path(config.paths.metadata_dir)

    train_trans = pd.read_parquet(interim_dir / "train_transaction.parquet")
    schema = generate_schema(train_trans)
    save_schema(schema, metadata_dir / "schema.json")

    # Mismatch check
    errors = validate_schema(train_trans, schema)
    if errors:
        logger.error("Schema mismatches detected: %s", errors)
        raise ValueError(f"Schema validation failed: {errors}")
    logger.info("Stage 2 completed successfully.")


def run_memory_optimization(config: Any) -> None:
    """Stage 3: Compress data sizes."""
    logger.info("--- Stage 3: Memory Optimization ---")
    interim_dir = Path(config.paths.interim_dir)
    metadata_dir = Path(config.paths.metadata_dir)

    train_trans = pd.read_parquet(interim_dir / "train_transaction.parquet")
    train_ident = pd.read_parquet(interim_dir / "train_identity.parquet")
    test_trans = pd.read_parquet(interim_dir / "test_transaction.parquet")
    test_ident = pd.read_parquet(interim_dir / "test_identity.parquet")

    # Run downcasting
    train_trans, t_rep = optimize_memory(
        train_trans, metadata_dir / "memory_report_train.json"
    )
    train_ident, _ = optimize_memory(train_ident)
    test_trans, _ = optimize_memory(test_trans)
    test_ident, _ = optimize_memory(test_ident)

    # Save compressed versions
    train_trans.to_parquet(interim_dir / "train_transaction_opt.parquet", index=False)
    train_ident.to_parquet(interim_dir / "train_identity_opt.parquet", index=False)
    test_trans.to_parquet(interim_dir / "test_transaction_opt.parquet", index=False)
    test_ident.to_parquet(interim_dir / "test_identity_opt.parquet", index=False)

    # MLflow track
    tracker = MLflowTracker(config.mlflow.experiment_name, config.mlflow.tracking_uri)
    tracker.start_run("Memory_Optimization_Stage")
    tracker.log_metrics({"memory_reduction_pct": t_rep["reduction_pct"]})
    tracker.log_artifact(str(metadata_dir / "memory_report_train.json"), "metadata")
    tracker.end_run()
    logger.info("Stage 3 completed successfully.")


def run_merge_identity(config: Any) -> None:
    """Stage 4: Join transaction and identity tables."""
    logger.info("--- Stage 4: Merge Identity ---")
    interim_dir = Path(config.paths.interim_dir)

    train_trans = pd.read_parquet(interim_dir / "train_transaction_opt.parquet")
    train_ident = pd.read_parquet(interim_dir / "train_identity_opt.parquet")
    test_trans = pd.read_parquet(interim_dir / "test_transaction_opt.parquet")
    test_ident = pd.read_parquet(interim_dir / "test_identity_opt.parquet")

    train_merged = merge_datasets(train_trans, train_ident)
    test_merged = merge_datasets(test_trans, test_ident)

    train_merged.to_parquet(interim_dir / "train_merged.parquet", index=False)
    test_merged.to_parquet(interim_dir / "test_merged.parquet", index=False)
    logger.info("Stage 4 completed successfully.")


def run_quality_checks_stage(config: Any) -> None:
    """Stage 5: Data quality scans."""
    logger.info("--- Stage 5: Quality Checks ---")
    interim_dir = Path(config.paths.interim_dir)
    metadata_dir = Path(config.paths.metadata_dir)

    train_merged = pd.read_parquet(interim_dir / "train_merged.parquet")
    # Clean infs, duplicates, constant columns
    train_cleaned, q_rep = run_quality_checks(
        train_merged,
        near_const_threshold=config.data.features.correlation_threshold,
        report_path=metadata_dir / "quality_report.json",
    )

    train_cleaned.to_parquet(interim_dir / "train_quality.parquet", index=False)

    tracker = MLflowTracker(config.mlflow.experiment_name, config.mlflow.tracking_uri)
    tracker.start_run("Quality_Checks_Stage")
    tracker.log_metrics(
        {
            "duplicate_rows_pct": q_rep["duplicate_rows_pct"],
            "dropped_constant_columns_count": len(q_rep["dropped_constant_columns"]),
            "duplicate_columns_count": len(q_rep["dropped_duplicate_columns"]),
        }
    )
    tracker.log_artifact(str(metadata_dir / "quality_report.json"), "metadata")
    tracker.end_run()
    logger.info("Stage 5 completed successfully.")


def run_missing_analysis(config: Any) -> None:
    """Stage 6: Analyze missing features and create ratios."""
    logger.info("--- Stage 6: Missing Analysis ---")
    interim_dir = Path(config.paths.interim_dir)
    metadata_dir = Path(config.paths.metadata_dir)

    train_df = pd.read_parquet(interim_dir / "train_quality.parquet")

    missing_count = train_df.isna().sum()
    missing_pct = (missing_count / len(train_df)) * 100

    missing_df = pd.DataFrame(
        {"missing_count": missing_count, "missing_pct": missing_pct}
    )
    missing_df.to_csv(metadata_dir / "missing_report.csv")

    # Create engineered missing indicators
    train_df["missing_count"] = train_df.isna().sum(axis=1).astype(int)
    train_df["missing_ratio"] = (
        train_df["missing_count"] / len(train_df.columns)
    ).astype(float)

    train_df.to_parquet(interim_dir / "train_missing.parquet", index=False)

    tracker = MLflowTracker(config.mlflow.experiment_name, config.mlflow.tracking_uri)
    tracker.start_run("Missing_Analysis_Stage")
    tracker.log_metrics({"overall_missing_avg_pct": float(missing_pct.mean())})
    tracker.log_artifact(str(metadata_dir / "missing_report.csv"), "metadata")
    tracker.end_run()
    logger.info("Stage 6 completed successfully.")


def run_clean_data(config: Any) -> None:
    """Stage 7: Standard categories cleaning."""
    logger.info("--- Stage 7: Clean Data ---")
    interim_dir = Path(config.paths.interim_dir)

    train_df = pd.read_parquet(interim_dir / "train_missing.parquet")
    train_cleaned = clean_categorical_columns(train_df)
    train_cleaned = impute_numerical_columns(train_cleaned, strategy="none")

    train_cleaned.to_parquet(interim_dir / "train_cleaned.parquet", index=False)
    logger.info("Stage 7 completed successfully.")


def run_metadata_generation(config: Any) -> None:
    """Stage 8: Profiling features."""
    logger.info("--- Stage 8: Metadata Generation ---")
    interim_dir = Path(config.paths.interim_dir)
    metadata_dir = Path(config.paths.metadata_dir)

    train_df = pd.read_parquet(interim_dir / "train_cleaned.parquet")
    generate_metadata_reports(
        train_df,
        dict_path=metadata_dir / "feature_dictionary.json",
        groups_path=metadata_dir / "column_groups.json",
    )
    logger.info("Stage 8 completed successfully.")


def run_drift_analysis(config: Any) -> None:
    """Stage 9: Evaluate split drift."""
    logger.info("--- Stage 9: Drift Analysis ---")
    interim_dir = Path(config.paths.interim_dir)
    metadata_dir = Path(config.paths.metadata_dir)

    train_df = pd.read_parquet(interim_dir / "train_cleaned.parquet")
    # For drift comparison, load / mock a test split
    test_trans = pd.read_parquet(interim_dir / "test_transaction_opt.parquet")
    test_ident = pd.read_parquet(interim_dir / "test_identity_opt.parquet")
    test_df = merge_datasets(test_trans, test_ident)
    test_df = clean_categorical_columns(test_df)

    drift_report = generate_drift_report(train_df, test_df, metadata_dir)

    tracker = MLflowTracker(config.mlflow.experiment_name, config.mlflow.tracking_uri)
    tracker.start_run("Drift_Analysis_Stage")
    tracker.log_metrics(
        {
            "severe_drift_features": drift_report["summary"]["severe_drift_count"],
            "moderate_drift_features": drift_report["summary"]["moderate_drift_count"],
        }
    )
    tracker.log_artifact(str(metadata_dir / "drift_report.html"), "metadata")
    tracker.log_artifact(str(metadata_dir / "psi_scores.csv"), "metadata")
    tracker.end_run()
    logger.info("Stage 9 completed successfully.")


def run_feature_store_registration(config: Any) -> None:
    """Stage 10: Store parsed families independently."""
    logger.info("--- Stage 10: Feature Store Registration ---")
    interim_dir = Path(config.paths.interim_dir)
    metadata_dir = Path(config.paths.metadata_dir)
    feature_store_dir = Path(config.paths.feature_store_dir)

    train_df = pd.read_parquet(interim_dir / "train_cleaned.parquet")

    # Load column classifications
    with Path(metadata_dir / "column_groups.json").open(encoding="utf-8") as f:
        groups = json.load(f)

    register_features_to_store(train_df, groups, feature_store_dir)
    logger.info("Stage 10 completed successfully.")


def run_save_processed_data(config: Any) -> None:
    """Stage 11: Export final versioned artifacts."""
    logger.info("--- Stage 11: Save Processed Data ---")
    interim_dir = Path(config.paths.interim_dir)
    processed_dir = Path(config.paths.processed_dir)

    train_df = pd.read_parquet(interim_dir / "train_cleaned.parquet")

    # Run leakage checks on the final set
    metadata_dir = Path(config.paths.metadata_dir)
    run_leakage_checks(
        train_df, target_col="isFraud", report_path=metadata_dir / "leakage_report.md"
    )

    save_processed_dataset(train_df, processed_dir)
    logger.info("Stage 11 completed successfully.")


def main() -> None:
    """Main orchestrator entrypoint."""
    parser = argparse.ArgumentParser(
        description="Orchestrate Data Engineering Pipeline."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "all",
            "load_data",
            "validate_schema",
            "memory_optimization",
            "merge_identity",
            "quality_checks",
            "missing_analysis",
            "clean_data",
            "metadata_generation",
            "drift_analysis",
            "feature_store_registration",
            "save_processed_data",
        ],
        help="Pipeline phase execution target.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional row sampling count.",
    )
    args = parser.parse_args()

    config = load_config()

    try:
        if args.stage == "all":
            run_load_data(config, max_rows=args.sample)
            run_validate_schema(config)
            run_memory_optimization(config)
            run_merge_identity(config)
            run_quality_checks_stage(config)
            run_missing_analysis(config)
            run_clean_data(config)
            run_metadata_generation(config)
            run_drift_analysis(config)
            run_feature_store_registration(config)
            run_save_processed_data(config)
        elif args.stage == "load_data":
            run_load_data(config, max_rows=args.sample)
        elif args.stage == "validate_schema":
            run_validate_schema(config)
        elif args.stage == "memory_optimization":
            run_memory_optimization(config)
        elif args.stage == "merge_identity":
            run_merge_identity(config)
        elif args.stage == "quality_checks":
            run_quality_checks_stage(config)
        elif args.stage == "missing_analysis":
            run_missing_analysis(config)
        elif args.stage == "clean_data":
            run_clean_data(config)
        elif args.stage == "metadata_generation":
            run_metadata_generation(config)
        elif args.stage == "drift_analysis":
            run_drift_analysis(config)
        elif args.stage == "feature_store_registration":
            run_feature_store_registration(config)
        elif args.stage == "save_processed_data":
            run_save_processed_data(config)

        logger.info("Pipeline execution of stage '%s' completed.", args.stage)

    except Exception as e:
        logger.critical("Data Engineering Pipeline crashed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
