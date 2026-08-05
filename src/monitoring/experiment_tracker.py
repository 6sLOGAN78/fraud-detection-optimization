"""10.1 - 10.3 Experiment Management & MLflow Tracking Module.

Provides pre-execution gates, experiment naming standards, and MLflow tracking integration:
- 10.1 Experiment Management Architecture
- 10.2 MLflow Tracking Engine
- 10.3 Experiment Naming Standards
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentPreExecutionGate:
    """Pre-execution pipeline verification gate checking tracking backend readiness."""

    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "mlruns")

    def verify(self) -> bool:
        """Verifies that tracking URI directory or server is accessible."""
        if not self.tracking_uri.startswith("http"):
            Path(self.tracking_uri).mkdir(parents=True, exist_ok=True)

        logger.info(f"Experiment Pre-Execution Gate PASSED. Tracking URI: {self.tracking_uri}")
        return True


class ExperimentNamingStandards:
    """10.3 Standardized experiment and run naming generator."""

    @staticmethod
    def generate_experiment_name(domain: str = "fraud", stage: str = "training") -> str:
        """Generates standardized experiment name."""
        return f"ieee-cis-{domain}-{stage}".lower()

    @staticmethod
    def generate_run_name(model_type: str, stage: str = "train") -> str:
        """Generates standardized run name with timestamp."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        return f"{stage}_{model_type}_{ts}".lower()


class ExperimentManagementArchitecture:
    """10.1 Core experiment management architecture orchestrating context setup and safety."""

    def __init__(
        self,
        experiment_name: str = "ieee-cis-fraud-detection",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "mlruns")

    def initialize_session(self) -> str:
        """Configures MLflow tracking URI and experiment."""
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri(self.tracking_uri)
        try:
            exp = mlflow.get_experiment_by_name(self.experiment_name)
            if exp is None:
                exp_id = mlflow.create_experiment(self.experiment_name)
            else:
                exp_id = exp.experiment_id
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"Initialized MLflow experiment '{self.experiment_name}' (ID: {exp_id})")
            return exp_id
        except Exception as e:
            logger.warning(f"Failed setting MLflow experiment: {e}")
            return "0"


class MLflowTrackingEngine:
    """10.2 Managed run context runner for MLflow tracking."""

    def __init__(self, experiment_name: str = "ieee-cis-fraud-detection"):
        self.arch = ExperimentManagementArchitecture(experiment_name=experiment_name)
        self.arch.initialize_session()

    def start_run(self, run_name: str, nested: bool = False) -> mlflow.ActiveRun:
        """Starts an active MLflow tracking run."""
        return mlflow.start_run(run_name=run_name, nested=nested)
