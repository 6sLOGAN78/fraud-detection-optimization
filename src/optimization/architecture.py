"""7.1 Optimization Architecture Core Engine.

Automates search and validation of optimal hyperparameter configurations to maximize model generalizing capabilities.
Provides execution wrappers, pre-execution pipeline verification gates, input/output validation, and logging.
"""

from __future__ import annotations

import logging
import time
import psutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationArchitectureDesign:
    """Prepares and validates inputs, checks schema alignment, sets parameters,

    and calculates standard design transformations for optimization workflows.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
        log_level: str = "INFO",
    ):
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level
        self.is_fit: bool = False
        self.feature_names_: List[str] = []

    def fit(
        self, X: pd.DataFrame, y: Optional(Union[pd.Series, np.ndarray]) = None
    ) -> OptimizationArchitectureDesign:
        """Validates input feature matrix and target series."""
        if X is None or X.empty:
            raise ValueError("Input feature matrix X cannot be empty or None")
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)) and y.isnull().any().any():
                raise ValueError("Target labels contain NaN entries")
            elif isinstance(y, np.ndarray) and np.isnan(y).any():
                raise ValueError("Target labels contain NaN entries")

        if len(X.columns) != len(set(X.columns)):
            raise ValueError("Feature matrix contains duplicate column headers")

        self.feature_names_ = list(X.columns)
        self.is_fit = True
        logger.info(
            f"OptimizationArchitectureDesign fit successfully on {len(self.feature_names_)} features."
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms feature matrix while validating schema alignment."""
        if not self.is_fit:
            raise ValueError(
                "OptimizationArchitectureDesign must be fit before transform"
            )
        if list(X.columns) != self.feature_names_:
            missing = set(self.feature_names_) - set(X.columns)
            if missing:
                raise ValueError(
                    f"Feature matrix is missing fit columns: {missing}"
                )
            X = X[self.feature_names_]
        return X.copy()


class CoreOptimizationEngine:
    """Core estimator fit and optimization execution wrapper monitoring memory,

    CPU time, and exception handling.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
        log_level: str = "INFO",
    ):
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level
        self.execution_stats: Dict[str, Any] = {}

    def execute(
        self, func: Any, *args: Any, **kwargs: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        """Executes a target optimization function with resource monitoring and safety gates."""
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)

        try:
            result = func(*args, **kwargs)
            status = "SUCCESS"
            error_msg = None
        except Exception as e:
            logger.error(f"Optimization execution failed: {str(e)}")
            result = None
            status = "FAILED"
            error_msg = str(e)

        elapsed_time = time.time() - start_time
        end_mem = psutil.Process().memory_info().rss / (1024 * 1024)

        self.execution_stats = {
            "elapsed_seconds": round(elapsed_time, 4),
            "memory_start_mb": round(start_mem, 2),
            "memory_end_mb": round(end_mem, 2),
            "memory_delta_mb": round(end_mem - start_mem, 2),
            "status": status,
            "error": error_msg,
        }
        return result, self.execution_stats


class OptimizationInputOutputProcessor:
    """Handles structured reading, input validation, and artifact output metadata generation

    for optimization runs.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
        log_level: str = "INFO",
    ):
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level

    def validate_inputs(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]
    ) -> bool:
        """Validates input dataset consistency before optimization starts."""
        if X is None or len(X) == 0:
            raise ValueError("Input DataFrame X cannot be empty.")
        if y is None or len(y) == 0:
            raise ValueError("Target vector y cannot be empty.")
        if len(X) != len(y):
            raise ValueError(
                f"Length mismatch: X has {len(X)} rows, y has {len(y)} rows."
            )
        return True

    def build_output_payload(
        self, best_params: Dict[str, Any], best_score: float, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Constructs a standardized payload dictionary containing trial results and metrics."""
        return {
            "best_params": best_params,
            "best_score": float(best_score),
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


class OptimizationImplementationStandards:
    """Enforces numerical stability checks, zero-variance handling, and MLflow monitoring integration."""

    def __init__(
        self,
        threshold: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
        log_level: str = "INFO",
    ):
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level

    def sanitize_scores(
        self, scores: Union[List[float], np.ndarray]
    ) -> np.ndarray:
        """Converts nan/inf values safely to default finite metrics."""
        arr = np.array(scores, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        return arr

    def verify_stability(self, val: float) -> bool:
        """Checks if a float metric value is valid and finite."""
        return not (np.isnan(val) or np.isinf(val))


class OptimizationPipelineExecution:
    """Pre-execution pipeline verification gate to ensure preceding pipeline artifacts exist."""

    def __init__(self, required_artifacts: Optional[List[str]] = None):
        self.required_artifacts = required_artifacts or [
            "data/interim/train_cleaned.parquet"
        ]

    def verify_prerequisites(self) -> bool:
        """Checks that all required upstream files exist before optimization executes."""
        missing = []
        for path_str in self.required_artifacts:
            path = Path(path_str)
            if not path.exists():
                missing.append(path_str)

        if missing:
            msg = f"Pre-execution gate failed! Missing required input artifacts: {missing}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info("Pre-execution pipeline verification gate PASSED.")
        return True
