"""Pipeline orchestration script executing automated feature selection stages, pruning redundancy, and logging selection results in MLflow. Optimized to prevent OOM."""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path

import pandas as pd
import mlflow

from src.feature_selection.selectors import NullSelector, VarianceSelector, CorrelationSelector, ImportanceSelector, MutualInformationSelector, SHAPSelector, PermutationImportanceSelector, RFESelector
from src.feature_selection.pipeline import FeatureSelectionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    logger.info("Executing Pre-Execution Pipeline Verification Gate...")

    # Define paths
    train_miss_in = Path("data/feature_store_engineered/v1/train_missing_features.parquet")
    test_miss_in = Path("data/feature_store_engineered/v1/test_missing_features.parquet")
    interim_train = Path("data/interim/train_merged.parquet")

    missing_deps = []
    for p in [train_miss_in, test_miss_in, interim_train]:
        if not p.exists():
            missing_deps.append(str(p))

    if missing_deps:
        msg = f"Dependency verification failed! Missing prior artifacts: {', '.join(missing_deps)}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Prior stage verification checks passed. Loading training features...")
    df_train = pd.read_parquet(train_miss_in)
    
    # Load ONLY the target label y to save ~900MB memory
    logger.info("Loading target labels...")
    df_target = pd.read_parquet(interim_train, columns=["isFraud"])
    y_train = df_target["isFraud"]

    # Re-align TransactionID index if needed or drop it from active analysis cols
    features_to_select = [c for c in df_train.columns if c != "TransactionID"]

    logger.info("Building Feature Selection Pipeline stages...")
    # 1. Information Completeness Filter (max 90% null rate)
    null_sel = NullSelector(threshold=0.90)
    # 2. Zero Variance Filter
    var_sel = VarianceSelector(threshold=0.0)
    # 3. Collinearity Filter (max 0.95 absolute correlation)
    corr_sel = CorrelationSelector(threshold=0.95)
    # 4. Mutual Information Filter (max normalized MI threshold=0.05)
    mi_sel = MutualInformationSelector(threshold=0.05, random_state=42)
    # 5. SHAP attribution Filter (max normalized SHAP score threshold=0.05)
    shap_sel = SHAPSelector(threshold=0.05, random_state=42)
    # 6. Permutation Importance Filter (max normalized Permutation score threshold=0.05)
    perm_sel = PermutationImportanceSelector(threshold=0.05, random_state=42)
    # 7. Recursive Feature Elimination Filter (top recursive features threshold=0.05)
    rfe_sel = RFESelector(threshold=0.05, random_state=42)
    # 8. Importance Filter (RandomForest baseline max normalized score threshold=0.05)
    imp_sel = ImportanceSelector(threshold=0.05, random_state=42)

    pipeline = FeatureSelectionPipeline([null_sel, var_sel, corr_sel, mi_sel, shap_sel, perm_sel, rfe_sel, imp_sel])

    logger.info("Fitting feature selectors sequentially on training data...")
    df_train_features = df_train[features_to_select]
    pipeline.fit(df_train_features, y_train)

    selected_cols = pipeline.selected_features_
    summary_report = pipeline.get_summary_report()

    # Free memory before transforming test/train files to avoid OOM
    logger.info("Freeing memory by dropping train features and target labels...")
    del df_train
    del df_target
    del df_train_features
    gc.collect()

    output_dir = Path("data/feature_store_engineered/v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = output_dir / "train_selected_features.parquet"
    test_out = output_dir / "test_selected_features.parquet"
    report_out = output_dir / "feature_selection_report.json"

    # Save summary report
    with open(report_out, "w") as f:
        json.dump(summary_report, f, indent=4)
    logger.info("Feature drop report saved to: %s", report_out)

    # Transform training dataset by reading ONLY selected columns
    logger.info("Loading and saving selected training features...")
    df_train_selected = pd.read_parquet(train_miss_in, columns=["TransactionID"] + selected_cols)
    df_train_selected.to_parquet(train_out, index=False)
    del df_train_selected
    gc.collect()

    # Transform test dataset by reading ONLY selected columns
    logger.info("Loading and saving selected test features...")
    df_test_selected = pd.read_parquet(test_miss_in, columns=["TransactionID"] + selected_cols)
    df_test_selected.to_parquet(test_out, index=False)
    del df_test_selected
    gc.collect()

    # MLflow Tracking
    logger.info("Logging feature selection parameters & summaries to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="feature_selection_pipeline")
        started = True

    try:
        mlflow.log_params({
            "pipeline_stage": "feature_selection",
            "null_complete_threshold": 0.90,
            "variance_threshold": 0.0,
            "collinearity_threshold": 0.95,
            "mutual_info_threshold": 0.05,
            "shap_threshold": 0.05,
            "permutation_importance_threshold": 0.05,
            "rfe_threshold": 0.05,
            "importance_threshold": 0.05,
            "initial_features_count": summary_report["total_initial_features"],
            "selected_features_count": summary_report["total_final_features"],
        })
        
        # Log feature counts dropped by selector
        for step in summary_report["steps"]:
            mlflow.log_param(f"dropped_by_{step['selector_name']}", step["dropped_count"])
            
        # Log artifacts
        mlflow.log_artifact(str(report_out), artifact_path="feature_selection")
    except Exception as e:
        logger.warning("MLflow tracking logging encountered warning: %s", e)
    finally:
        if started:
            mlflow.end_run()

    logger.info("Feature selection pipeline completed successfully.")


if __name__ == "__main__":
    main()
