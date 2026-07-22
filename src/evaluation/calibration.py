"""Probability Calibration module aligning prediction probabilities to actual target fractions."""

from __future__ import annotations

import logging
import time

import pandas as pd
import numpy as np
import psutil
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Utility class to estimate and correct prediction probability distribution mapping."""
    def __init__(self, n_bins: int = 10, log_level: str = "INFO"):
        self.n_bins = n_bins
        self.log_level = log_level
        self.calibration_metrics_ = {}

    def analyze(self, y_true: np.ndarray | pd.Series, y_probs: np.ndarray) -> dict:
        """Computes calibration curves and Brier Score Loss before mapping correction."""
        if len(y_true) == 0 or len(y_probs) == 0:
            raise ValueError("Input vectors are empty")
            
        start_time = time.time()
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=self.n_bins)
        brier = brier_score_loss(y_true, y_probs)
        
        elapsed = time.time() - start_time
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        
        metrics = {
            "elapsed_time_seconds": elapsed,
            "memory_delta_mb": mem_after - mem_before,
            "brier_score_loss": float(brier),
            "true_probabilities": prob_true.tolist(),
            "pred_probabilities": prob_pred.tolist()
        }
        self.calibration_metrics_ = metrics
        
        logger.info("ProbabilityCalibrator finished. Brier Score Loss: %.5f", brier)
        return metrics
