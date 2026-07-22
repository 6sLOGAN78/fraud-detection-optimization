"""Pipeline script to execute feature encoding strategy selection, fitting, transformations, and metadata registries."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

from src.feature_engineering.encoding import (
    EncodingStrategySelector,
    VectorizedLabelEncoder,
    VectorizedFrequencyEncoder,
    VectorizedCountEncoder,
    LeakageSafeTargetEncoder,
    VectorizedOneHotEncoder,
    EncodingValidationGate,
    EncoderRegistry,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    logger.info("Executing Pre-Execution Pipeline Verification Gate...")

    train_in = Path("data/feature_store_engineered/v1/train_complete_feature_matrix.parquet")
    test_in = Path("data/feature_store_engineered/v1/test_complete_feature_matrix.parquet")
    interim_train = Path("data/interim/train_merged.parquet")

    missing = []
    for path in [train_in, test_in, interim_train]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        msg = f"Dependency verification failed! Missing prior artifacts: {', '.join(missing)}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Prior stage verification checks passed.")

    # Load complete matrices
    logger.info("Loading inputs...")
    df_train = pd.read_parquet(train_in)
    df_test = pd.read_parquet(test_in)

    # Join targets to train partition
    df_interim = pd.read_parquet(interim_train, columns=["TransactionID", "isFraud"])
    df_train = pd.merge(df_train, df_interim, on="TransactionID", how="left")
    y_train = df_train["isFraud"]

    # We also need categorical variables from raw or interim data to encode them:
    # Let's load the relevant source categorical columns to encode
    cat_columns_source = [
        "ProductCD", "DeviceType", "DeviceInfo", 
        "P_emaildomain", "R_emaildomain", 
        "addr1", "addr2", "card1", "card4", "card6"
    ]
    
    # Load source categoricals and merge them
    df_source_train = pd.read_parquet(interim_train, columns=["TransactionID"] + [c for c in cat_columns_source if c in pd.read_parquet(interim_train).columns])
    df_train = pd.merge(df_train, df_source_train, on="TransactionID", how="left")
    
    interim_test = Path("data/interim/test_merged.parquet")
    df_source_test = pd.read_parquet(interim_test, columns=["TransactionID"] + [c for c in cat_columns_source if c in pd.read_parquet(interim_test).columns])
    df_test = pd.merge(df_test, df_source_test, on="TransactionID", how="left")

    # Automate strategy selections
    selector = EncodingStrategySelector()
    strategies = selector.select_strategy(df_train, [c for c in cat_columns_source if c in df_train.columns])

    # Registry and Gate validators
    registry = EncoderRegistry()
    validator = EncodingValidationGate()

    store_dir = Path("data/feature_store_engineered")
    version = "v1"
    
    df_encoded_train = pd.DataFrame(index=df_train.index)
    df_encoded_train["TransactionID"] = df_train["TransactionID"]
    df_encoded_train["isFraud"] = df_train["isFraud"]

    df_encoded_test = pd.DataFrame(index=df_test.index)
    df_encoded_test["TransactionID"] = df_test["TransactionID"]

    # Iterate and perform encodings based on strategies
    for feature_name, strategy in strategies.items():
        logger.info("Encoding feature %s using strategy %s", feature_name, strategy)
        s_train = df_train[feature_name]
        s_test = df_test[feature_name]
        
        cardinality = s_train.nunique()

        if strategy == "Label":
            enc = VectorizedLabelEncoder()
            enc.fit(s_train)
            df_encoded_train[f"{feature_name}_encoded"] = enc.transform(s_train)
            df_encoded_test[f"{feature_name}_encoded"] = enc.transform(s_test)
            registry.register(feature_name, "Label", enc, cardinality)

        elif strategy == "Frequency":
            enc = VectorizedFrequencyEncoder()
            enc.fit(s_train)
            df_encoded_train[f"{feature_name}_freq"] = enc.transform(s_train)
            df_encoded_test[f"{feature_name}_freq"] = enc.transform(s_test)
            registry.register(feature_name, "Frequency", enc, cardinality)

        elif strategy == "Count":
            enc = VectorizedCountEncoder()
            enc.fit(s_train)
            # Count, log count, percentile features
            df_encoded_train[f"{feature_name}_count"] = enc.transform(s_train, mode="count")
            df_encoded_train[f"{feature_name}_log_count"] = enc.transform(s_train, mode="log_count")
            
            df_encoded_test[f"{feature_name}_count"] = enc.transform(s_test, mode="count")
            df_encoded_test[f"{feature_name}_log_count"] = enc.transform(s_test, mode="log_count")
            
            registry.register(feature_name, "Count", enc, cardinality)

        elif strategy == "Target":
            enc = LeakageSafeTargetEncoder()
            # Out-of-fold fitted transform to prevent train leakage
            df_encoded_train[f"{feature_name}_target"] = enc.fit_transform(s_train, y_train)
            df_encoded_test[f"{feature_name}_target"] = enc.transform(s_test)
            registry.register(feature_name, "Target", enc, cardinality)

        elif strategy == "OneHot":
            enc = VectorizedOneHotEncoder()
            enc.fit(s_train)
            
            ohe_train = enc.transform(s_train)
            ohe_test = enc.transform(s_test)
            
            for col in ohe_train.columns:
                df_encoded_train[col] = ohe_train[col]
                df_encoded_test[col] = ohe_test[col]
                
            registry.register(feature_name, "OneHot", enc, cardinality)

    # Validate output datasets
    val_train_report = validator.validate(df_encoded_train)
    val_test_report = validator.validate(df_encoded_test)

    # Save outputs
    registry.save_bundle(store_dir / version)
    
    # Save parquet matrices
    train_out = store_dir / version / "train_encoded_features.parquet"
    test_out = store_dir / version / "test_encoded_features.parquet"
    
    df_encoded_train.to_parquet(train_out, index=False)
    df_encoded_test.to_parquet(test_out, index=False)
    
    logger.info("Saved encoded matrices to %s and %s", train_out, test_out)

    # Log to MLflow
    logger.info("Logging encoding analytics to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="feature_encoding_pipeline")
        started = True

    try:
        mlflow.log_params({
            "pipeline_stage": "feature_encoding",
            "version": version,
            "encoded_strategies_count": len(strategies),
            "train_validation_gate_status": val_train_report["status"],
            "test_validation_gate_status": val_test_report["status"],
        })
    except Exception as e:
        logger.warning("MLflow tracking logging encountered warning: %s", e)
    finally:
        if started:
            mlflow.end_run()

    logger.info("Feature encoding pipeline completed successfully.")


if __name__ == "__main__":
    main()
