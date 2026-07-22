"""LightGBM classifier component with built-in null checking, model formulation, and metrics monitoring."""

from __future__ import annotations

import logging
import time

import pandas as pd
import numpy as np
import psutil
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightGBMClassifier:
    """LightGBM classifier wrapped estimator with hyperparameter profile, safety hooks, and performance telemetry."""
    def __init__(self, threshold: float = 0.05, random_state: int = 42, n_jobs: int = -1, log_level: str = "INFO"):
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level
        # Default hyperparameter profiles
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            importance_type="gain",
            verbose=-1
        )
        self.fit_metrics_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMClassifier:
        if X is None or X.empty:
            raise ValueError("Input feature matrix X is empty or None")
            
        X_clean = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        start_time = time.time()
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        
        self.model.fit(X_clean, y)
        
        elapsed = time.time() - start_time
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        mem_delta = mem_after - mem_before
        
        self.fit_metrics_ = {
            "fit_time_seconds": elapsed,
            "memory_delta_mb": mem_delta,
            "samples_count": len(X_clean)
        }
        
        logger.info("LightGBMClassifier trained in %.2fs. Memory Delta: %.2fMB", elapsed, mem_delta)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        try:
            X_clean = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            return self.model.predict_proba(X_clean)
        except Exception as e:
            logger.warning("LightGBM prediction failed: %s. Returning default zero probabilities.", e)
            return np.zeros((len(X), 2))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            X_clean = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            return self.model.predict(X_clean)
        except Exception as e:
            logger.warning("LightGBM prediction failed: %s. Returning default class 0.", e)
            return np.zeros(len(X), dtype=int)
