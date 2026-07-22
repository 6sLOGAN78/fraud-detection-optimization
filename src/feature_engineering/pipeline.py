"""Core Feature Engineering Architecture — Part 4.1."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureInventoryManager:
    """Manages the catalog and registers features engineered in the pipeline."""
    def __init__(self) -> None:
        self.registry: dict[str, dict[str, Any]] = {}

    def register_feature(self, name: str, feature_type: str, source_cols: list[str], description: str) -> None:
        self.registry[name] = {
            "name": name,
            "feature_type": feature_type,
            "source_columns": source_cols,
            "description": description,
            "registered_at": pd.Timestamp.now().isoformat(),
        }

    def get_catalog(self) -> list[dict[str, Any]]:
        return list(self.registry.values())

    def save_inventory(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.get_catalog(), f, indent=4)
        logger.info("Feature inventory saved to %s", path)


class ValidationGate:
    """Performs pre-run input and post-run output checks for schema or value warnings."""
    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold

    def validate_inputs(self, df: pd.DataFrame) -> None:
        """Asserts correct columns exist in input."""
        logger.info("Running input validation gate checks...")
        if df.empty:
            raise ValueError("Input dataframe is empty!")
            
        # Target check (e.g. TransactionID must exist)
        if "TransactionID" not in df.columns:
            raise KeyError("TransactionID column is required but missing.")

    def validate_outputs(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validates outputs for extreme values, NaN rates, columns consistency."""
        logger.info("Running output feature validation gate checks...")
        
        checks = {}
        for col in df.columns:
            null_pct = df[col].isnull().mean()
            if null_pct > self.threshold:
                checks[col] = f"WARNING: High NaN rate ({null_pct:.4f})"
                logger.warning("Feature %s has high NaN rate of %.4f", col, null_pct)
            else:
                checks[col] = "PASSED"
                
        return checks


class FeatureStore:
    """Manages serializing features to parquet store and tracking manifest metadata."""
    def __init__(self, store_dir: Path) -> None:
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save_features(self, df: pd.DataFrame, partition_name: str, version: str) -> Path:
        """Saves features to partitioned parquet store."""
        version_dir = self.store_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = version_dir / f"{partition_name}_features.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info("Saved %s features to %s", partition_name, parquet_path)
        return parquet_path

    def save_manifest(self, catalog: list[dict[str, Any]], version: str) -> None:
        manifest = {
            "version": version,
            "created_at": pd.Timestamp.now().isoformat(),
            "features_catalog": catalog,
        }
        with open(self.store_dir / version / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)


class FeatureEngineeringPipeline:
    def __init__(
        self,
        threshold: float = 0.05,
        random_state: int = 42,
        log_level: str = "INFO",
    ) -> None:
        self.threshold = threshold
        self.random_state = random_state
        self.inventory = FeatureInventoryManager()
        self.validator = ValidationGate()
        self.store = FeatureStore(Path("data/feature_store_engineered"))
        
        # Ensure MLflow allow file store
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    def fit_transform(self, df_train: pd.DataFrame, df_test: pd.DataFrame, version: str = "v1") -> tuple[pd.DataFrame, pd.DataFrame]:
        """Orchestrates full feature generation vector processing."""
        logger.info("Executing transformation pipeline on train and test datasets...")
        start_time = time.time()

        # 1. Inbound validation
        self.validator.validate_inputs(df_train)
        self.validator.validate_inputs(df_test)

        # Work on copies
        df_trn_out = df_train.copy()
        df_tst_out = df_test.copy()

        # --- Vectorized feature generation ---

        # Case 1: Temporal Time-of-Day Feature Generator
        # TransactionDT is delta in seconds. Map daily cycle:
        # Time-of-Day = (TransactionDT % 86400) / 3600.0 (hour coordinates)
        if "TransactionDT" in df_trn_out.columns:
            logger.info("Generating vectorized temporal features...")
            df_trn_out["hour_of_day"] = (df_trn_out["TransactionDT"] % 86400) / 3600.0
            df_tst_out["hour_of_day"] = (df_tst_out["TransactionDT"] % 86400) / 3600.0
            
            self.inventory.register_feature(
                "hour_of_day",
                "Temporal / Numeric",
                ["TransactionDT"],
                "Vectorized hour coordinate of transaction within 24h cycle",
            )

        # Case 2: Transaction Amount Log Transformation
        if "TransactionAmt" in df_trn_out.columns:
            logger.info("Generating transaction AMT log features...")
            df_trn_out["log_TransactionAmt"] = np.log1p(df_trn_out["TransactionAmt"])
            df_tst_out["log_TransactionAmt"] = np.log1p(df_tst_out["TransactionAmt"])
            
            self.inventory.register_feature(
                "log_TransactionAmt",
                "Transform / Numeric",
                ["TransactionAmt"],
                "Logarithm transform of TransactionAmt for variance compression",
            )

        # Case 3: Interactive Ratio Feature Generator
        # If both TransactionAmt and C1 exist, compute interaction feature transaction_amt_per_c1
        if "TransactionAmt" in df_trn_out.columns and "C1" in df_trn_out.columns:
            logger.info("Generating interactive ratio features...")
            # Prevent division by zero
            c1_trn = df_trn_out["C1"].replace(0, 1)
            c1_tst = df_tst_out["C1"].replace(0, 1)
            
            df_trn_out["amt_ratio_C1"] = df_trn_out["TransactionAmt"] / c1_trn
            df_tst_out["amt_ratio_C1"] = df_tst_out["TransactionAmt"] / c1_tst
            
            self.inventory.register_feature(
                "amt_ratio_C1",
                "Interaction / Numeric",
                ["TransactionAmt", "C1"],
                "Ratio interaction metric between transaction amount and count C1",
            )

        # Case 4: Category Target-frequency Encoder
        # Frequency encode ProductCD
        if "ProductCD" in df_trn_out.columns:
            logger.info("Generating frequency encoded category features...")
            freq_map = df_trn_out["ProductCD"].value_counts(normalize=True).to_dict()
            df_trn_out["ProductCD_freq"] = df_trn_out["ProductCD"].astype(str).map(freq_map).fillna(0.0)
            df_tst_out["ProductCD_freq"] = df_tst_out["ProductCD"].astype(str).map(freq_map).fillna(0.0)
            
            self.inventory.register_feature(
                "ProductCD_freq",
                "Encoding / Numeric",
                ["ProductCD"],
                "Frequency encoder value representing rarity density of ProductCD categories",
            )

        # 3. Post-run out validation check
        checks_trn = self.validator.validate_outputs(df_trn_out)
        checks_tst = self.validator.validate_outputs(df_tst_out)

        # 4. Save to Feature Store partitions
        self.store.save_features(df_trn_out, "train", version)
        self.store.save_features(df_tst_out, "test", version)
        self.store.save_manifest(self.inventory.get_catalog(), version)

        # 5. Log environment metrics to MLflow
        processing_duration = time.time() - start_time
        logger.info("Feature engineering finished in %.4f seconds.", processing_duration)

        self._log_mlflow(df_trn_out, df_tst_out, processing_duration, checks_trn)

        return df_trn_out, df_tst_out

    def _log_mlflow(self, df_trn: pd.DataFrame, df_tst: pd.DataFrame, duration: float, checks: dict) -> None:
        """Registers metrics inside MLflow runs."""
        active = mlflow.active_run()
        started = False
        if active is None:
            mlflow.start_run(run_name="feature_engineering_pipeline")
            started = True

        try:
            mlflow.log_params({
                "pipeline_stage": "feature_engineering",
                "random_state": self.random_state,
                "nan_threshold": self.threshold,
            })
            
            mlflow.log_metrics({
                "duration_seconds": duration,
                "train_feature_count": float(df_trn.shape[1]),
                "test_feature_count": float(df_tst.shape[1]),
                "validation_failed_count": float(sum(1 for v in checks.values() if "WARNING" in v)),
            })
        except Exception as e:
            logger.warning("MLflow logs registration encountered warnings: %s", e)
        finally:
            if started:
                mlflow.end_run()
