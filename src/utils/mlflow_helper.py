"""MLflow tracking helper for logging experiments, metrics, parameters, and artifacts.

Provides functions to log parameters, metrics, plots, and models.
"""

import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException

from src.utils.logging import setup_logger

# Enable local file store database for MLflow offline fallbacks
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

logger = setup_logger("mlflow_helper")


class MLflowTracker:
    """Wrapper class for MLflow experiment tracking.

    Handles remote tracking availability and local offline fallbacks.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        config_path: str | None = None,
    ):
        """Initializes the MLflow tracker.

        Args:
            experiment_name: Name of the experiment to log to.
            tracking_uri: URI of the tracking server (defaults to local).
            config_path: Optional path to the configuration file logged.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.config_path = config_path
        self.active_run = None

        server_alive = False
        if tracking_uri and (tracking_uri.startswith("http://") or tracking_uri.startswith("https://")):
            import urllib.request
            from urllib.error import URLError
            try:
                # Quick 1s connection check
                with urllib.request.urlopen(tracking_uri, timeout=1.0) as conn:
                    # MLflow server will return HTTP 200 (or other positive codes) on root
                    if conn.getcode() is not None:
                        server_alive = True
            except (URLError, Exception):
                pass

        if tracking_uri and server_alive:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info("Setting MLflow tracking URI to %s", tracking_uri)
        else:
            if tracking_uri:
                logger.warning(
                    "MLflow tracking server at %s is offline. Falling back to local offline 'file:./mlruns'.",
                    tracking_uri,
                )
            mlflow.set_tracking_uri("file:./mlruns")

        try:
            mlflow.set_experiment(experiment_name)
            logger.info("MLflow experiment set to '%s'", experiment_name)
        except MlflowException as e:
            logger.warning(
                "Could not set MLflow experiment. Falling back to local mlruns: %s",
                e,
            )
            # Revert to local mlflow tracking
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str | None = None) -> mlflow.ActiveRun:
        """Starts a new MLflow run.

        Args:
            run_name: Optional name for this specific execution run.

        Returns:
            The active MLflow run context.
        """
        self.active_run = mlflow.start_run(run_name=run_name)
        logger.info("Started MLflow run: %s", run_name or self.active_run.info.run_id)

        # Log configuration file as artifact if provided
        if self.config_path and Path(self.config_path).exists():
            mlflow.log_artifact(self.config_path, artifact_path="configs")

        return self.active_run

    def log_params(self, params: dict[str, Any]) -> None:
        """Logs a dictionary of parameters.

        Args:
            params: Parameters key-value dictionary.
        """
        if not self.active_run:
            logger.warning("No active run found. Parameter logging skipped.")
            return

        try:
            # MLflow requires string values or standard values, we sanitize them
            sanitized = {k: str(v) for k, v in params.items()}
            mlflow.log_params(sanitized)
            logger.debug("Logged parameters to MLflow: %s", sanitized)
        except MlflowException as e:
            logger.error("Failed to log parameters: %s", e)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Logs a dictionary of metrics.

        Args:
            metrics: Metrics key-value dictionary.
            step: Optional epoch/iteration step number.
        """
        if not self.active_run:
            logger.warning("No active run found. Metric logging skipped.")
            return

        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug("Logged metrics to MLflow: %s", metrics)
        except MlflowException as e:
            logger.error("Failed to log metrics: %s", e)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Logs a local file or directory as an MLflow artifact.

        Args:
            local_path: Absolute or relative path to the local artifact.
            artifact_path: Optional destination directory prefix in mlflow store.
        """
        if not self.active_run:
            logger.warning("No active run found. Artifact logging skipped.")
            return

        path = Path(local_path)
        if not path.exists():
            logger.warning("Artifact path %s does not exist.", local_path)
            return

        try:
            if path.is_dir():
                mlflow.log_artifacts(local_path, artifact_path=artifact_path)
            else:
                mlflow.log_artifact(local_path, artifact_path=artifact_path)
            logger.info("Logged artifact %s to MLflow.", local_path)
        except MlflowException as e:
            logger.error("Failed to log artifact %s: %s", local_path, e)

    def log_artifacts(self, files: list[str], artifact_path: str | None = None) -> None:
        """Logs multiple files as artifacts.

        Args:
            files: List of file paths to log.
            artifact_path: Log path inside MLflow.
        """
        for f in files:
            self.log_path(f, artifact_path=artifact_path)

    def log_path(self, path: str, artifact_path: str | None = None) -> None:
        """Helper to log file or directory path.

        Args:
            path: Target local file/directory.
            artifact_path: Optional MLflow namespace destination.
        """
        self.log_artifact(path, artifact_path=artifact_path)

    def end_run(self) -> None:
        """Ends the active MLflow run."""
        if self.active_run:
            mlflow.end_run()
            logger.info("Ended active MLflow run.")
            self.active_run = None
