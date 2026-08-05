"""Pipeline script to execute Part 10 — MLOps Experiment Tracking & Model Registry."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.monitoring import (
    ArtifactManagementEngine,
    DatasetVersionTracker,
    ExperimentManagementArchitecture,
    ExperimentNamingStandards,
    ExperimentPreExecutionGate,
    MetricTrackingEngine,
    ParameterTrackingEngine,
    ReproducibilityFramework,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 10 Experiment Tracking Pipeline")
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of samples for run tracking")
    args = parser.parse_args()

    train_path = Path("data/interim/train_cleaned.parquet")
    if not train_path.exists():
        train_path = Path("data/interim/train_merged.parquet")

    gate = ExperimentPreExecutionGate()
    gate.verify()

    # Lock seed & snapshot environment
    repro = ReproducibilityFramework(seed=42)
    repro.set_global_seed()
    env_snapshot = repro.capture_environment_snapshot()

    # Track dataset version
    df = pd.read_parquet(train_path)
    if len(df) > args.n_samples:
        df = df.sample(n=args.n_samples, random_state=42).reset_index(drop=True)

    ds_tracker = DatasetVersionTracker()
    dataset_meta = ds_tracker.get_dataset_metadata(df, file_path=str(train_path))

    exp_name = ExperimentNamingStandards.generate_experiment_name("fraud", "tracking")
    arch = ExperimentManagementArchitecture(experiment_name=exp_name)
    arch.initialize_session()

    run_name = ExperimentNamingStandards.generate_run_name("rf_baseline", "experiment")

    with mlflow.start_run(run_name=run_name):
        logger.info(f"Started MLflow run '{run_name}'")

        param_tracker = ParameterTrackingEngine()
        params = {
            "n_estimators": 25,
            "max_depth": 5,
            "random_state": 42,
            "n_samples": args.n_samples,
        }
        param_tracker.log_params(params)

        # Log dataset & environment metadata
        param_tracker.log_params(dataset_meta, prefix="dataset")

        # Train baseline
        y = df["isFraud"].values
        X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")
        X_num = X.select_dtypes(include=["number"]).fillna(0)

        clf = RandomForestClassifier(n_estimators=25, max_depth=5, random_state=42)
        clf.fit(X_num, y)
        acc = float(clf.score(X_num, y))

        metric_tracker = MetricTrackingEngine()
        metric_tracker.log_metrics({"training_accuracy": acc, "dataset_rows": len(df)})

        artifact_engine = ArtifactManagementEngine()
        artifact_engine.log_dict_as_artifact(env_snapshot, "environment_snapshot.json")

        logger.info(f"Part 10 Experiment Tracking run completed successfully. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
