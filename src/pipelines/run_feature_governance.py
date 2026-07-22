"""Pipeline orchestration script executing feature store governance audits, validating metrics, and committing registry tags to MLflow."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
import mlflow

from src.data.governance import FeatureGovernanceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    logger.info("Executing Pre-Execution Pipeline Verification Gate...")

    # Define paths
    registry_path = Path("data/feature_store_foundation/registry.json")
    train_path = Path("data/feature_store_engineered/v1/train_missing_features.parquet")
    test_path = Path("data/feature_store_engineered/v1/test_missing_features.parquet")

    missing_deps = []
    for p in [registry_path, train_path, test_path]:
        if not p.exists():
            missing_deps.append(str(p))

    if missing_deps:
        msg = f"Dependency verification failed! Missing prior feature store registry or inputs: {', '.join(missing_deps)}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Dependency verification passed. Loading datasets...")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    logger.info("Initializing Feature Governance Engine...")
    engine = FeatureGovernanceEngine(registry_path)

    logger.info("Auditing features for null-ratios, variance checks, and drift (PSI)...")
    # Define thresholds
    null_threshold = 0.05
    drift_threshold = 0.25
    admin_token = "ADMIN_TOKEN_999"

    report = engine.audit_features(
        df_train=df_train,
        df_test=df_test,
        null_threshold=null_threshold,
        drift_threshold=drift_threshold,
        token=admin_token
    )

    report_path = Path("data/feature_store_foundation/feature_governance_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info("Feature Governance Audit report saved to: %s", report_path)

    # MLflow instrumentation
    logger.info("Logging feature governance status & metrics to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="feature_governance_audit")
        started = True

    try:
        mlflow.log_params({
            "pipeline_stage": "feature_governance",
            "governance_null_threshold": null_threshold,
            "governance_drift_threshold": drift_threshold,
            "governance_status": report["status"],
            "audited_views_count": report["audited_views_count"],
        })
        
        # Log report file
        mlflow.log_artifact(str(report_path), artifact_path="feature_store")
    except Exception as e:
        logger.warning("MLflow governance logging encountered warning: %s", e)
    finally:
        if started:
            mlflow.end_run()

    logger.info("Feature Store Governance pipeline completed successfully.")


if __name__ == "__main__":
    main()
