"""9.1 Explainability Architecture Module.

Provides pre-execution gates, architecture design, and input/output processors for model explainability.
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


class ExplainabilityPreExecutionGate:
    """Pre-execution pipeline verification gate checking model artifacts and datasets."""

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
            msg = f"Explainability Pre-Execution Gate FAILED! Missing dependencies: {missing}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info("Explainability Pre-Execution Gate PASSED.")
        return True


class ExplainabilityArchitectureDesign:
    """Core architecture design for model explainability validation."""

    def validate_inputs(
        self, model: Any, X: pd.DataFrame
    ) -> Tuple[Any, pd.DataFrame]:
        """Validates estimator methods and feature matrix."""
        if X is None or X.empty:
            raise ValueError("Input DataFrame X cannot be empty or None.")

        if not hasattr(model, "predict") and not hasattr(model, "predict_proba"):
            raise TypeError("Model must implement predict or predict_proba interface.")

        num_cols = X.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("DataFrame X must contain at least one numeric feature column.")

        X_clean = X[num_cols].fillna(0)
        return model, X_clean


class ExplainabilityInputOutputProcessor:
    """Builds metadata and payload dictionaries for explainability artifacts."""

    def build_payload(
        self, model_name: str, global_importance: Dict[str, float], metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Structures explainability results payload."""
        return {
            "model_name": model_name,
            "global_feature_importance": global_importance,
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


class ExplainabilityImplementationStandards:
    """Enforces computational stability and zero variance handling."""

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Performs division with zero-denominator fallback."""
        if denominator == 0.0 or np.isnan(denominator):
            return default
        return float(numerator / denominator)

    @staticmethod
    def normalize_importance(importances: Dict[str, float]) -> Dict[str, float]:
        """Normalizes importance values to sum to 1.0."""
        total = sum(importances.values())
        if total == 0.0 or np.isnan(total):
            return {k: 0.0 for k in importances}
        return {k: round(v / total, 6) for k, v in importances.items()}
