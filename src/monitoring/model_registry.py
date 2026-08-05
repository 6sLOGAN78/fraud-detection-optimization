"""10.7 - 10.9 Model Registry, Run Comparison, and Dataset Versioning Module.

Provides lifecycle management, run comparison, and dataset hash tracking:
- 10.7 Model Registry & Lifecycle Manager
- 10.8 Run Comparison Engine
- 10.9 Dataset Version Tracking Engine
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowModelRegistryManager:
    """10.7 Model Registry lifecycle manager handling model registration and stage transitions (Staging/Production)."""

    def register_model(
        self, model: Any, model_name: str, artifact_path: str = "model"
    ) -> Optional[Any]:
        """Registers a fitted model into the MLflow Model Registry."""
        try:
            res = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=model_name,
            )
            logger.info(f"Successfully registered model '{model_name}' to MLflow Model Registry.")
            return res
        except Exception as e:
            logger.warning(f"Failed to register model '{model_name}' in MLflow: {e}")
            return None

    def transition_stage(self, model_name: str, version: int, stage: str = "Staging") -> bool:
        """Transitions a registered model version to Staging or Production."""
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name, version=str(version), stage=stage
            )
            logger.info(f"Successfully transitioned model '{model_name}' v{version} to stage '{stage}'.")
            return True
        except Exception as e:
            logger.warning(f"Failed to transition model stage: {e}")
            return False


class RunComparisonEngine:
    """10.8 Compares metric performance across multiple MLflow runs to identify the optimal model."""

    def compare_runs(
        self, experiment_name: str = "ieee-cis-fraud-detection", metric_name: str = "val_score"
    ) -> Dict[str, Any]:
        """Queries MLflow tracking runs and returns best performing run metadata."""
        try:
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(experiment_name)
            if exp is None:
                return {"error": f"Experiment '{experiment_name}' not found."}

            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=[f"metrics.{metric_name} DESC"],
            )

            if not runs:
                return {"total_runs": 0, "best_run": None}

            best_run = runs[0]
            return {
                "total_runs": len(runs),
                "best_run_id": best_run.info.run_id,
                "best_run_name": best_run.data.tags.get("mlflow.runName", ""),
                "best_metric_value": best_run.data.metrics.get(metric_name, None),
                "best_params": best_run.data.params,
            }
        except Exception as e:
            logger.warning(f"Failed comparing MLflow runs: {e}")
            return {"error": str(e)}


class DatasetVersionTracker:
    """10.9 Tracks dataset file hashes, shapes, and schema versions."""

    def compute_file_hash(self, file_path: Union[str, Path]) -> str:
        """Computes SHA256 hash of dataset file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_dataset_metadata(self, df: pd.DataFrame, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Extracts dataset shape, columns, missing value counts, and file hash."""
        metadata = {
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "columns": list(df.columns),
            "total_missing_values": int(df.isnull().sum().sum()),
        }
        if file_path:
            metadata["file_path"] = file_path
            metadata["sha256_hash"] = self.compute_file_hash(file_path)

        return metadata
