"""Pipeline orchestration script to ingest engineered features into Offline and Online feature stores with governance, lineage metadata, and SECURITY RBAC tokens."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import mlflow

from src.data.store import FeatureStoreClient, FeatureLineage

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

    logger.info("Dependency verification passed. Initializing Feature Store Client with Security tokens...")

    # Paths for registry, offline base, and online db
    store_dir = Path("data/feature_store_foundation")
    store_dir.mkdir(parents=True, exist_ok=True)
    
    registry_path = store_dir / "registry.json"
    offline_dir = store_dir / "offline"
    online_db = store_dir / "online.db"

    # Client uses default config tokens
    client = FeatureStoreClient(registry_path, offline_dir, online_db)
    
    # We will use ADMIN_TOKEN_999 for ingestion/registration
    admin_token = "ADMIN_TOKEN_999"

    # Load datasets
    df_train = pd.read_parquet(train_miss_in)
    df_test = pd.read_parquet(test_miss_in)

    # 1. Define Features to Register & Ingest
    features_to_register = ["missing_ratio", "completeness_score"]

    logger.info("Constructing lineage structures...")
    train_lineage = FeatureLineage(
        source_dataset=str(train_miss_in),
        pipeline_stage="feature_missing",
        transformation_type="vectorized-missingness",
        description="Missingness ratio and completeness markers on transaction fields",
    )
    
    test_lineage = FeatureLineage(
        source_dataset=str(test_miss_in),
        pipeline_stage="feature_missing",
        transformation_type="vectorized-missingness",
        description="Missingness ratio and completeness markers on transaction fields",
    )

    logger.info("Registering train and test feature views with lineage and tags...")
    client.register_feature_view(
        name="missingness_train_features",
        entity_id="TransactionID",
        features=features_to_register,
        source_path=str(train_miss_in),
        version="v1",
        owner="Fraud Core Team",
        tags=["missingness", "train", "quality_metrics"],
        description="Train indicators for missing column ratios",
        lineage=train_lineage,
        token=admin_token,
    )
    
    client.register_feature_view(
        name="missingness_test_features",
        entity_id="TransactionID",
        features=features_to_register,
        source_path=str(test_miss_in),
        version="v1",
        owner="Fraud Core Team",
        tags=["missingness", "test", "quality_metrics"],
        description="Test indicators for missing column ratios Verification",
        lineage=test_lineage,
        token=admin_token,
    )

    # Ingest train and test features into offline snappy and online SQLite
    logger.info("Ingesting train features with RBAC validation gate...")
    client.ingest("missingness_train_features", df_train, token=admin_token)

    logger.info("Ingesting test features with RBAC validation gate...")
    client.ingest("missingness_test_features", df_test, token=admin_token)

    # MLflow tracking instrumentation
    logger.info("Logging feature store ingestion and lineage metadata to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="feature_store_operations")
        started = True

    try:
        catalog_views = client.catalog.list_views()
        mlflow.log_params({
            "pipeline_stage": "feature_store_operations",
            "offline_store_dir": str(offline_dir),
            "online_store_database": str(online_db),
            "registered_views_count": len(catalog_views),
            "owner": "Fraud Core Team",
            "tags": ", ".join(catalog_views[0].get("tags", [])),
        })
    except Exception as e:
        logger.warning("MLflow tracking logging encountered warning: %s", e)
    finally:
        if started:
            mlflow.end_run()

    logger.info("Feature Store Operations pipeline completed successfully.")


if __name__ == "__main__":
    main()
