"""10.4 - 10.6 Parameter, Metric, and Artifact Logging Engines.

Provides logging utilities for hyperparameter tracking, metric time series, and artifact management:
- 10.4 Parameter Tracking Engine
- 10.5 Metric Tracking Engine
- 10.6 Artifact Management Engine
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParameterTrackingEngine:
    """10.4 Hyperparameter and configuration parameter tracker."""

    def log_params(self, params: Dict[str, Any], prefix: str = "") -> None:
        """Logs dictionary of parameters to active MLflow run."""
        clean_params = {}
        for k, v in params.items():
            key_name = f"{prefix}_{k}" if prefix else str(k)
            clean_params[key_name] = str(v)

        try:
            mlflow.log_params(clean_params)
            logger.info(f"Successfully logged {len(clean_params)} parameters to MLflow.")
        except Exception as e:
            logger.warning(f"Failed to log params to MLflow: {e}")


class MetricTrackingEngine:
    """10.5 Metric tracking engine logging epoch metrics and summary stats."""

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Logs metrics dictionary to active MLflow run."""
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, float)) and not (v is None):
                clean_metrics[str(k)] = float(v)

        try:
            mlflow.log_metrics(clean_metrics, step=step)
            logger.info(f"Successfully logged {len(clean_metrics)} metrics to MLflow.")
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")


class ArtifactManagementEngine:
    """10.6 Artifact management engine saving plots, json reports, and model files to MLflow."""

    def log_dict_as_artifact(
        self, data: Dict[str, Any], filename: str, artifact_path: Optional[str] = None
    ) -> Path:
        """Saves a Python dictionary as a JSON artifact in MLflow."""
        tmp_dir = Path("artifacts/tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        local_path = tmp_dir / filename

        with open(local_path, "w") as f:
            json.dump(data, f, indent=2)

        try:
            mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
            logger.info(f"Successfully logged artifact '{filename}' to MLflow.")
        except Exception as e:
            logger.warning(f"Failed to log artifact '{filename}' to MLflow: {e}")

        return local_path

    def log_file(self, local_file_path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
        """Logs an existing local file to MLflow artifacts."""
        path = Path(local_file_path)
        if not path.exists():
            raise FileNotFoundError(f"File to log does not exist: {local_file_path}")

        try:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
            logger.info(f"Successfully logged artifact file '{path.name}' to MLflow.")
        except Exception as e:
            logger.warning(f"Failed to log artifact file '{path.name}' to MLflow: {e}")
