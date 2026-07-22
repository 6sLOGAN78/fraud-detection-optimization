"""Threshold Optimization module calculating metrics and determining optimal decision thresholds."""

from __future__ import annotations

import logging
import time

import pandas as pd
import numpy as np
import psutil
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, fbeta_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """Threshold optimizer to identify decision cutoffs according to specific optimization targets."""
    def __init__(self, target_metric: str = "f1", beta: float = 2.0, log_level: str = "INFO"):
        self.target_metric = target_metric
        self.beta = beta
        self.log_level = log_level
        self.best_threshold_ = 0.5
        self.optimization_metrics_ = {}

    def optimize(self, y_true: np.ndarray | pd.Series, y_probs: np.ndarray) -> float:
        """Finds the optimal threshold on a validation dataset to maximize the target metric."""
        if len(y_true) == 0 or len(y_probs) == 0:
            raise ValueError("Input true labels or predicted probabilities are empty")
            
        start_time = time.time()
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Test thresholds dynamically
        thresholds = np.linspace(0.01, 0.99, 99)
        best_score = -1.0
        best_thresh = 0.5
        
        scores = []
        precision_vals = []
        recall_vals = []
        
        y_true_arr = np.asarray(y_true)
        
        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            prec = precision_score(y_true_arr, y_pred, zero_division=0)
            rec = recall_score(y_true_arr, y_pred, zero_division=0)
            
            if self.target_metric == "f1":
                score = f1_score(y_true_arr, y_pred, zero_division=0)
            elif self.target_metric == "f2":
                score = fbeta_score(y_true_arr, y_pred, beta=2.0, zero_division=0)
            else:
                score = score = fbeta_score(y_true_arr, y_pred, beta=self.beta, zero_division=0)
                
            scores.append(score)
            precision_vals.append(prec)
            recall_vals.append(rec)
            
            if score > best_score:
                best_score = score
                best_thresh = t
                
        elapsed = time.time() - start_time
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        
        self.best_threshold_ = float(best_thresh)
        self.optimization_metrics_ = {
            "optimize_time_seconds": elapsed,
            "memory_delta_mb": mem_after - mem_before,
            "best_threshold": float(best_thresh),
            "best_score": float(best_score),
            "thresholds": thresholds.tolist(),
            "scores": scores,
            "precisions": precision_vals,
            "recalls": recall_vals
        }
        
        logger.info("ThresholdOptimizer complete. Best %s threshold: %.2f (Score: %.5f)", 
                    self.target_metric, best_thresh, best_score)
        return float(best_thresh)
