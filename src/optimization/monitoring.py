"""7.8 Optimization Monitoring Module.

Provides trial tracking, MLflow logging, convergence metrics calculation,
and hyperparameter importance analysis.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import optuna
from optuna.importance import get_param_importances

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationMonitor:
    """Monitors trial progression, calculates convergence stats, computes parameter importances,

    and handles MLflow integration.
    """

    def __init__(self, experiment_name: str = "fraud_detection_hpo"):
        self.experiment_name = experiment_name
        self._init_mlflow()

    def _init_mlflow(self) -> None:
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment '{self.experiment_name}': {e}")

    def log_trial_to_mlflow(
        self, trial: optuna.FrozenTrial, model_type: str = "model"
    ) -> None:
        """Logs a single completed trial to MLflow as a child run."""
        try:
            with mlflow.start_run(run_name=f"{model_type}_trial_{trial.number}", nested=True):
                mlflow.log_params(trial.params)
                if trial.value is not None:
                    mlflow.log_metric("val_score", trial.value)
                mlflow.log_metric("trial_number", trial.number)
                mlflow.set_tag("state", trial.state.name)
        except Exception as e:
            logger.warning(f"Failed to log trial {trial.number} to MLflow: {e}")

    def log_study_summary(self, study: optuna.Study, model_type: str = "model") -> Dict[str, Any]:
        """Calculates study convergence metrics and logs overall best results to MLflow."""
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        values = [t.value for t in completed_trials if t.value is not None]

        summary = {
            "model_type": model_type,
            "total_trials": len(study.trials),
            "completed_trials": len(completed_trials),
            "pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "best_value": float(study.best_value) if values else None,
            "best_params": study.best_params,
            "mean_value": float(np.mean(values)) if values else None,
            "std_value": float(np.std(values)) if values else None,
        }

        # Calculate hyperparameter importance if trials > 1
        param_importances = {}
        if len(completed_trials) > 1:
            try:
                param_importances = get_param_importances(study)
                summary["param_importances"] = param_importances
            except Exception as e:
                logger.warning(f"Could not calculate param importances: {e}")

        try:
            with mlflow.start_run(run_name=f"{model_type}_hpo_summary"):
                mlflow.log_params(study.best_params)
                if values:
                    mlflow.log_metric("best_value", float(study.best_value))
                    mlflow.log_metric("mean_value", float(np.mean(values)))
                    mlflow.log_metric("total_trials", len(study.trials))
                for param, importance in param_importances.items():
                    mlflow.log_metric(f"importance_{param}", float(importance))
        except Exception as e:
            logger.warning(f"Failed to log study summary to MLflow: {e}")

        return summary
