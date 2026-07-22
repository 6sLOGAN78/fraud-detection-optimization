"""Baseline Model formulation logic for Logistic Regression and XGBoost classifiers."""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import numpy as np
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogisticRegressionBaseline:
    """Logistic Regression baseline classifier with built-in standard scaling and null guards."""
    def __init__(self, threshold: float = 0.05, random_state: int = 42, n_jobs: int = -1, log_level: str = "INFO"):
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level
        # Pipeline mapping scaling + logistic regression
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=self.n_jobs if self.n_jobs > 0 else None
            ))
        ])
        self.fit_metrics_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LogisticRegressionBaseline:
        # Pre-computation data processing spec & schema alignment checks
        if X is None or X.empty:
            raise ValueError("Input feature matrix X is empty or None")
        
        # Stability Analysis: handle divide-by-zero, infinite entries and NaNs stably
        X_clean = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        # CPU time and memory monitor checks
        start_time = time.time()
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        
        self.pipeline.fit(X_clean, y)
        
        elapsed = time.time() - start_time
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        mem_delta = mem_after - mem_before
        
        self.fit_metrics_ = {
            "fit_time_seconds": elapsed,
            "memory_delta_mb": mem_delta,
            "samples_count": len(X_clean)
        }
        
        logger.info("LogisticRegressionBaseline trained in %.2fs. Memory Delta: %.2fMB", elapsed, mem_delta)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        try:
            X_clean = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            return self.pipeline.predict_proba(X_clean)
        except Exception as e:
            logger.warning("Logistic Regression prediction failed: %s. Returning default zero probabilities.", e)
            return np.zeros((len(X), 2))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            X_clean = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            return self.pipeline.predict(X_clean)
        except Exception as e:
            logger.warning("Logistic Regression prediction failed: %s. Returning default class 0.", e)
            return np.zeros(len(X), dtype=int)


class XGBoostBaseline:
    """XGBoost baseline classifier with hyperparameter profiles, fallback recovery, and performance metrics."""
    def __init__(self, threshold: float = 0.05, random_state: int = 42, n_jobs: int = -1, log_level: str = "INFO"):
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level
        # Default hyperparameter profiles
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric="logloss"
        )
        self.fit_metrics_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBoostBaseline:
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
        
        logger.info("XGBoostBaseline trained in %.2fs. Memory Delta: %.2fMB", elapsed, mem_delta)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        try:
            X_clean = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            return self.model.predict_proba(X_clean)
        except Exception as e:
            logger.warning("XGBoost prediction failed: %s. Returning default zero probabilities.", e)
            return np.zeros((len(X), 2))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            X_clean = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            return self.model.predict(X_clean)
        except Exception as e:
            logger.warning("XGBoost prediction failed: %s. Returning default class 0.", e)
            return np.zeros(len(X), dtype=int)
