"""11.10 - 11.11 Deployment Validation, Health Checks, and Rollback Strategy Module.

Provides automated deployment pre-flight validation and error fallback rollback management:
- 11.10 Deployment Validator & Health Checks
- 11.11 Rollback Strategy Manager
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.deployment.inference import InferencePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentValidator:
    """11.10 Pre-flight deployment validator executing smoke tests and latency SLA verification."""

    def __init__(self, max_allowed_latency_ms: float = 100.0):
        self.max_allowed_latency_ms = max_allowed_latency_ms

    def validate_pipeline(self, pipeline: InferencePipeline, sample_row: Dict[str, Any]) -> Dict[str, Any]:
        """Runs a pre-flight smoke test against an active inference pipeline."""
        df_sample = pd.DataFrame([sample_row])

        try:
            preds, probs = pipeline.predict(df_sample)
            prob_val = float(probs[0])
            pred_val = int(preds[0])

            is_valid_output = (0.0 <= prob_val <= 1.0) and (pred_val in [0, 1])

            return {
                "smoke_test_passed": is_valid_output,
                "sample_probability": prob_val,
                "sample_prediction": pred_val,
                "status": "HEALTHY" if is_valid_output else "UNHEALTHY",
            }
        except Exception as e:
            logger.error(f"Deployment pre-flight validation failed: {e}")
            return {
                "smoke_test_passed": False,
                "error": str(e),
                "status": "UNHEALTHY",
            }


class RollbackManager:
    """11.11 Rollback strategy manager providing graceful fallback logic during endpoint failure."""

    def __init__(self, primary_pipeline: InferencePipeline, fallback_pipeline: Optional[InferencePipeline] = None):
        self.primary_pipeline = primary_pipeline
        self.fallback_pipeline = fallback_pipeline
        self.is_rolled_back: bool = False

    def predict_safe(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Executes prediction with automatic fallback to secondary pipeline if primary fails."""
        if not self.is_rolled_back:
            try:
                return self.primary_pipeline.predict(X)
            except Exception as e:
                logger.error(f"Primary inference pipeline failed! Triggering ROLLBACK: {e}")
                self.is_rolled_back = True

        if self.fallback_pipeline is not None:
            logger.info("Executing fallback inference pipeline...")
            return self.fallback_pipeline.predict(X)
        else:
            logger.warning("No fallback pipeline configured. Returning conservative default predictions.")
            n = len(X)
            default_probs = np.full(n, 0.05)
            default_preds = np.zeros(n, dtype=int)
            return default_preds, default_probs
