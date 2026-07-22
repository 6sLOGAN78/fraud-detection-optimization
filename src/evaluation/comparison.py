"""Model Comparison module to evaluate multiple models side-by-side."""

from __future__ import annotations

import logging
import time

import pandas as pd
import numpy as np
import psutil
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, fbeta_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Utility class to compare multiple trained models quantitatively."""
    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level
        self.comparison_results_ = {}

    def compare(self, models: dict, X_test: pd.DataFrame, y_test: np.ndarray | pd.Series) -> dict:
        """Evaluates multiple models on a common testing dataset."""
        if not models:
            raise ValueError("Models dictionary is empty")
        if len(X_test) == 0 or len(y_test) == 0:
            raise ValueError("Input test features or test labels are empty")
            
        start_time = time.time()
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        
        results = {}
        
        y_test_arr = np.asarray(y_test)
        
        for name, clf in models.items():
            logger.info("Computing metrics for model: %s", name)
            y_probs = clf.predict_proba(X_test)[:, 1]
            
            # Determine threshold
            threshold = getattr(clf, "threshold", 0.5)
            y_pred = (y_probs >= threshold).astype(int)
            
            auc = roc_auc_score(y_test_arr, y_probs)
            acc = accuracy_score(y_test_arr, y_pred)
            prec = precision_score(y_test_arr, y_pred, zero_division=0)
            rec = recall_score(y_test_arr, y_pred, zero_division=0)
            f1 = f1_score(y_test_arr, y_pred, zero_division=0)
            f2 = fbeta_score(y_test_arr, y_pred, beta=2.0, zero_division=0)
            
            results[name] = {
                "auc": float(auc),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "f2_score": float(f2),
                "threshold": float(threshold)
            }
            
        elapsed = time.time() - start_time
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        
        self.comparison_results_ = {
            "elapsed_time_seconds": elapsed,
            "memory_delta_mb": mem_after - mem_before,
            "comparisons": results
        }
        
        logger.info("ModelComparator run complete. Models compared: %s", list(models.keys()))
        return self.comparison_results_
