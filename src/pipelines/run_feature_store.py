"""Pipeline orchestration script to ingest engineered features into Offline and Online feature stores, and log metadata to MLflow."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import mlflow

from src.data.store import FeatureStoreClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    logger.info("Executing Pre-Execution Pipeline Verification Gate...")

    # Define paths
    train_miss_in = Path("data/feature_store_engineered/v1/train_missing_features.parquet")
    test_miss_in = Path("data/feature_store_engineered/v1/test_missing_features.parquet")

    missing_deps = []
    for path in [train_miss_in, test_miss_in]:
        if not path.exists():
            missing_deps.append(str(path))

    if missing_deps:
        msg = f"Dependency verification failed! Missing prior artifacts: {', '.join(missing_deps)}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Dependency verification passed. Initializing Feature Store Client...")

    # Paths for registry, offline base, and online db
    store_dir = Path("data/feature_store_foundation")
    store_dir.mkdir(parents=True, exist_ok=True)
    
    registry_path = store_dir / "registry.json"
    offline_dir = store_dir / "offline"
    online_db = store_dir / "online.db"

    client = FeatureStoreClient(registry_path, offline_dir, online_db)

    # Load datasets
    df_train = pd.read_parquet(train_miss_in)
    df_test = pd.read_parquet(test_miss_in)

    # 1. Define Features to Register & Ingest
    # We will register "missing_ratio" and "completeness_score"
    features_to_register = ["missing_ratio", "completeness_score"]

    logger.info("Registering train and test feature views...")
    client.register_feature_view(
        name="missingness_train_features",
        entity_id="TransactionID",
        features=features_to_register,
        source_path=str(train_miss_in),
        version="v1",
    )
    
    client.register_feature_view(
        name="missingness_test_features",
        entity_id="TransactionID",
        features=features_to_register,
        source_path=str(test_miss_in),
        version="v1",
    )

    # Ingest train and test features into offline snappy and online SQLite
    logger.info("Ingesting train features into Unified Feature Store...")
    client.ingest("missingness_train_features", df_train)

    logger.info("Ingesting test features into Unified Feature Store...")
    client.ingest("missingness_test_features", df_test)

    # MLflow tracking instrumentation
    logger.info("Logging feature store ingestion metadata to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="feature_store_foundation")
        started = True

    try:
        mlflow.log_params({
            "pipeline_stage": "feature_store_foundation",
            "offline_store_dir": str(offline_dir),
            "online_store_database": str(online_db),
            "registered_views_count": len(client.catalog.list_views()),
            "ingested_train_rows": len(df_train),
            "ingested_test_rows": len(df_test),
        })
    except Exception as e:
        logger.warning("MLflow tracking logging encountered warning: %s", e)
    finally:
        if started:
            mlflow.end_run()

    logger.info("Feature Store Foundation Ingestion Pipeline completed successfully.")


if __name__ == "__main__":
    main()
