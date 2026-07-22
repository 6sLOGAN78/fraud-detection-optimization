"""Time Series Cross Validation module executing temporal folds split and profiling execution."""

from __future__ import annotations

import logging
import time

import pandas as pd
import numpy as np
import psutil
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesCrossValidator:
    """Time-series cross-validator wrapper executing temporal split folds."""
    def __init__(self, n_splits: int = 5, log_level: str = "INFO"):
        self.n_splits = n_splits
        self.log_level = log_level
        self.cv = TimeSeriesSplit(n_splits=self.n_splits)
        self.split_metrics_ = []

    def split(self, X: pd.DataFrame, y: pd.Series | None = None):
        """Generates temporal splits indices."""
        if X is None or X.empty:
            raise ValueError("Input feature matrix X is empty or None")
            
        start_time = time.time()
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        
        folds = list(self.cv.split(X, y))
        
        elapsed = time.time() - start_time
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        mem_delta = mem_after - mem_before
        
        self.split_metrics_.append({
            "split_time_seconds": elapsed,
            "memory_delta_mb": mem_delta,
            "n_samples": len(X)
        })
        
        logger.info("TimeSeriesCrossValidator generated %d folds.", len(folds))
        return folds
