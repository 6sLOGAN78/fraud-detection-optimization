"""Pipeline script to execute feature families extraction and merging."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import mlflow

from src.feature_engineering.families import (
    TransactionFamilyBuilder,
    IdentityFamilyBuilder,
    TimeFamilyBuilder,
    AmountFamilyBuilder,
    EmailFamilyBuilder,
    DeviceFamilyBuilder,
    AddressFamilyBuilder,
    CardFamilyBuilder,
    FeatureFamilyTracker,
    FeatureFamilyIntegrator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    # Pre-Execution Pipeline Verification Gate
    logger.info("Executing Feature Families Pre-Execution Pipeline Verification Gate...")
    
    train_in = Path("data/feature_store_engineered/v1/train_features.parquet")
    test_in = Path("data/feature_store_engineered/v1/test_features.parquet")
    manifest_in = Path("data/feature_store_engineered/v1/manifest.json")

    missing = []
    for path in [train_in, test_in, manifest_in]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        msg = f"Dependency verification failed! Missing prior artifacts: {', '.join(missing)}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Prior stage verification checks passed.")

    # Load parquets
    logger.info("Loading inputs...")
    df_train = pd.read_parquet(train_in)
    df_test = pd.read_parquet(test_in)

    version = "v1"
    store_dir = Path("data/feature_store_engineered")
    
    tracker = FeatureFamilyTracker()
    integrator = FeatureFamilyIntegrator(store_dir)

    builders = {
        "transaction": (TransactionFamilyBuilder(), ["TransactionID", "ProductCD", "TransactionDT"]),
        "identity": (IdentityFamilyBuilder(), ["TransactionID", "DeviceType"]),
        "time": (TimeFamilyBuilder(), ["TransactionID", "TransactionDT"]),
        "amount": (AmountFamilyBuilder(), ["TransactionID", "TransactionAmt"]),
        "email": (EmailFamilyBuilder(), ["TransactionID", "P_emaildomain", "R_emaildomain"]),
        "device": (DeviceFamilyBuilder(), ["TransactionID", "DeviceInfo"]),
        "address": (AddressFamilyBuilder(), ["TransactionID", "addr1", "addr2"]),
        "card": (CardFamilyBuilder(), ["TransactionID", "card1", "card4"]),
    }

    # Process partitions
    for partition, df_input in [("train", df_train), ("test", df_test)]:
        logger.info("Processing execution on partition: %s", partition)
        families_dfs = {}
        
        for name, (builder, deps) in builders.items():
            df_family = builder.transform(df_input)
            families_dfs[name] = df_family

            # Record family metadata registry during train processing
            if partition == "train":
                features_added = [col for col in df_family.columns if col != "TransactionID"]
                tracker.record_family(
                    family_name=name,
                    features_added=features_added,
                    source_dependencies=deps,
                    version=version,
                )

        # Merge families and output complete feature matrix
        integrator.integrate(families_dfs, partition, version)

    # Save central metadata registry files
    tracker.save_metadata(store_dir / version)

    # Log to MLflow
    logger.info("Logging feature families metadata registry parameters to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="feature_families_pipeline")
        started = True

    try:
        mlflow.log_params({
            "pipeline_stage": "feature_families",
            "version": version,
            "feature_families_registered": list(builders.keys()),
        })
    except Exception as e:
        logger.warning("MLflow tracking logging encountered warning: %s", e)
    finally:
        if started:
            mlflow.end_run()

    logger.info("Feature families integration pipeline completed successfully.")


if __name__ == "__main__":
    main()
