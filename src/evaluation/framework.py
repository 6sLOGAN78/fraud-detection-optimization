"""8.1 Evaluation Framework Architecture Module.

Provides pre-execution verification gates, standardized input/output processing,
and computational stability checks for the model evaluation framework.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationPreExecutionGate:
    """Pre-execution pipeline verification gate checking required model artifacts and datasets."""

    def __init__(self, required_files: Optional[List[str]] = None):
        self.required_files = required_files or [
            "data/interim/train_merged.parquet"
        ]

    def verify(self) -> bool:
        """Verifies that mandatory upstream dependencies exist."""
        missing = []
        for path_str in self.required_files:
            if not Path(path_str).exists():
                missing.append(path_str)

        if missing:
            msg = f"Evaluation Pre-Execution Gate FAILED! Missing dependencies: {missing}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info("Evaluation Pre-Execution Gate PASSED.")
        return True


class EvaluationFrameworkDesign:
    """Core evaluation architecture design validating inputs, label consistency, and numerical boundaries."""

    def __init__(self, random_state: int = 42, log_level: str = "INFO"):
        self.random_state = random_state
        self.log_level = log_level

    def validate_inputs(
        self, y_true: Union[pd.Series, np.ndarray], y_prob: Union[pd.Series, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validates array dimensions, nulls, and probability range constraints."""
        y_true_arr = np.asarray(y_true, dtype=int).ravel()
        y_prob_arr = np.asarray(y_prob, dtype=float).ravel()

        if len(y_true_arr) == 0 or len(y_prob_arr) == 0:
            raise ValueError("Input arrays cannot be empty.")
        if len(y_true_arr) != len(y_prob_arr):
            raise ValueError(
                f"Length mismatch: y_true ({len(y_true_arr)}) vs y_prob ({len(y_prob_arr)})"
            )
        if np.isnan(y_true_arr).any() or np.isnan(y_prob_arr).any():
            raise ValueError("Evaluation inputs contain NaN values.")

        y_prob_arr = np.clip(y_prob_arr, 0.0, 1.0)
        return y_true_arr, y_prob_arr


class EvaluationInputOutputProcessor:
    """Structures metric evaluation inputs and serializes report metadata."""

    def build_report_metadata(
        self, model_name: str, metrics: Dict[str, Any], execution_time: float
    ) -> Dict[str, Any]:
        """Formats evaluation metrics into a standardized report structure."""
        return {
            "model_name": model_name,
            "metrics": metrics,
            "execution_time_seconds": round(execution_time, 4),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


class EvaluationImplementationStandards:
    """Enforces zero variance handling and numerical stability checks."""

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Performs division with zero-denominator fallback."""
        if denominator == 0.0 or np.isnan(denominator):
            return default
        return float(numerator / denominator)
