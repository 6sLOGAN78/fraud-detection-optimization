"""Pipeline orchestration script executing automated feature selection stages, pruning redundancy, and logging selection results in MLflow. Optimized to prevent OOM."""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np
import mlflow

from src.feature_selection.selectors import NullSelector, VarianceSelector, CorrelationSelector, ImportanceSelector, MutualInformationSelector, SHAPSelector, PermutationImportanceSelector, RFESelector, SequentialSelector, BorutaSelector, SimulatedAnnealingSelector, FeatureStabilitySelector, FeatureSelectionValidator
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
    from src.config.config import ConfigurationManager
    config = ConfigurationManager().get_config()

    # 1. Information Completeness Filter (max 90% null rate)
    null_sel = NullSelector(threshold=config.data.missing_threshold)
    # 2. Zero Variance Filter
    var_sel = VarianceSelector(threshold=0.0)
    # 3. Collinearity Filter (max 0.95 absolute correlation)
    corr_sel = CorrelationSelector(
        threshold=config.data.correlation_threshold, 
        random_state=config.seed,
        max_samples=config.training.max_samples
    )
    # 4. Mutual Information Filter (max normalized MI threshold=0.05)
    mi_sel = MutualInformationSelector(threshold=config.data.selection_threshold, random_state=config.seed)
    # 5. SHAP attribution Filter (max normalized SHAP score threshold=0.05)
    shap_sel = SHAPSelector(threshold=config.data.selection_threshold, random_state=config.seed)
    # 6. Permutation Importance Filter (max normalized Permutation score threshold=0.05)
    perm_sel = PermutationImportanceSelector(threshold=config.data.selection_threshold, random_state=config.seed)
    # 7. Recursive Feature Elimination Filter (top recursive features threshold=0.05)
    rfe_sel = RFESelector(threshold=config.data.selection_threshold, random_state=config.seed)
    # 8. Sequential Feature Selection Filter (greedy forward selection, select 12 features)
    sfs_sel = SequentialSelector(n_features_to_select=12, random_state=config.seed)
    # 9. Boruta Filter (shadow feature threshold=0.05)
    boruta_sel = BorutaSelector(threshold=config.data.selection_threshold, random_state=config.seed)
    # 10. Simulated Annealing Filter (search for best subset)
    sa_sel = SimulatedAnnealingSelector(threshold=config.data.selection_threshold, random_state=config.seed)
    # 11. Feature Stability Filter (drop highly unstable features across bootstraps, threshold=0.05)
    stability_sel = FeatureStabilitySelector(threshold=config.data.selection_threshold, random_state=config.seed)
    # 12. Importance Filter (RandomForest baseline max normalized score threshold=0.05)
    imp_sel = ImportanceSelector(threshold=config.data.selection_threshold, random_state=config.seed)
    # 13. Quality Validation Gate
    validation_gate = FeatureSelectionValidator(random_state=config.seed)

    pipeline = FeatureSelectionPipeline([null_sel, var_sel, corr_sel, mi_sel, shap_sel, perm_sel, rfe_sel, sfs_sel, boruta_sel, sa_sel, stability_sel, imp_sel, validation_gate])

    logger.info("Fitting feature selectors sequentially on training data...")
    df_train_features = df_train[features_to_select]
    
    max_fit_samples = config.training.max_samples * 2  # Feature selection fits on up to 2x train target size
    if len(df_train_features) > max_fit_samples:
        logger.info("Downsampling training data to %d samples for pipeline fit...", max_fit_samples)
        rng = np.random.RandomState(config.seed)
        fit_indices = rng.choice(df_train_features.index, size=max_fit_samples, replace=False)
        df_fit_features = df_train_features.loc[fit_indices]
        y_fit = y_train.loc[fit_indices]
    else:
        df_fit_features = df_train_features
        y_fit = y_train

    pipeline.fit(df_fit_features, y_fit)

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

    # Generate Feature Registry JSON Standard
    from datetime import datetime
    import subprocess
    import pyarrow.parquet as pq

    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        commit_hash = "unknown"

    registry = {
        "version": "v1.0",
        "timestamp": datetime.now().isoformat(),
        "access_control": {
            "read_roles": ["data_scientists", "ml_engineers", "training_pipelines"],
            "write_roles": ["feature_engineering_pipeline"],
            "usage_guidelines": "Pruned, validated feature subset approved for model training and real-time inference."
        },
        "lifecycle_management": {
            "stage": "production",
            "updated_by": "run_feature_selection",
            "git_commit": commit_hash
        },
        "features": {}
    }

    schema = pq.read_schema(train_out)
    for col in selected_cols:
        col_type = str(schema.field(col).type)
        stability_score = float(stability_sel.stability_scores_.get(col, 1.0))
        registry["features"][col] = {
            "data_type": col_type,
            "stability_score": stability_score,
            "selection_status": "selected",
            "lifecycle_stage": "production"
        }

    registry_out = output_dir / "feature_registry.json"
    with open(registry_out, "w") as f:
        json.dump(registry, f, indent=4)
    logger.info("Final Feature Registry JSON saved to: %s", registry_out)

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
            "sfs_features_to_select": 12,
            "boruta_threshold": 0.05,
            "simulated_annealing_threshold": 0.05,
            "feature_stability_threshold": 0.05,
            "importance_threshold": 0.05,
            "initial_features_count": summary_report["total_initial_features"],
            "selected_features_count": summary_report["total_final_features"],
        })
        
        # Log feature counts dropped by selector
        for step in summary_report["steps"]:
            mlflow.log_param(f"dropped_by_{step['selector_name']}", step["dropped_count"])
            
        # Log artifacts
        mlflow.log_artifact(str(report_out), artifact_path="feature_selection")
        mlflow.log_artifact(str(registry_out), artifact_path="feature_selection")
    except Exception as e:
        logger.warning("MLflow tracking logging encountered warning: %s", e)
    finally:
        if started:
            mlflow.end_run()

    logger.info("Feature selection pipeline completed successfully.")


if __name__ == "__main__":
    main()
