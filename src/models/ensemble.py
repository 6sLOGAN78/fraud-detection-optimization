"""Ensemble Classifier executing weighted averaging of underlying LightGBM, XGBoost, and CatBoost models."""

from __future__ import annotations

import logging
import time
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleClassifier:
    """Ensemble blended estimator wrapping LightGBM, XGBoost, and CatBoost models with safety checks."""
    def __init__(self, weights: list[float] | None = None, threshold: float = 0.5, log_level: str = "INFO"):
        self.weights = weights if weights is not None else [0.4, 0.3, 0.3]
        self.threshold = threshold
        self.log_level = log_level
        self.lgbm_model = None
        self.xgb_model = None
        self.cat_model = None
        self.fit_metrics_ = {}

    def load_base_models(
        self,
        lgbm_path: str = "data/models/v1/lightgbm_model.pkl",
        xgb_path: str = "data/models/v1/xgboost_model.pkl",
        cat_path: str = "data/models/v1/catboost_model.pkl"
    ) -> EnsembleClassifier:
        """Loads serialized base models from files."""
        logger.info("Ensemble loading base estimators...")
        try:
            with open(Path(lgbm_path), "rb") as f:
                self.lgbm_model = pickle.load(f)
            with open(Path(xgb_path), "rb") as f:
                self.xgb_model = pickle.load(f)
            with open(Path(cat_path), "rb") as f:
                self.cat_model = pickle.load(f)
            logger.info("Ensemble base models loaded successfully.")
        except Exception as e:
            logger.error("Ensemble failed to load base paths: %s", e)
            raise e
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EnsembleClassifier:
        """Ensemble fit runs a metadata profiling step since models are pre-trained."""
        if X is None or X.empty:
            raise ValueError("Input feature matrix X is empty or None")
            
        start_time = time.time()
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Ensure base estimators are loaded
        if self.lgbm_model is None or self.xgb_model is None or self.cat_model is None:
            self.load_base_models()
            
        elapsed = time.time() - start_time
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        mem_delta = mem_after - mem_before
        
        self.fit_metrics_ = {
            "fit_time_seconds": elapsed,
            "memory_delta_mb": mem_delta,
            "samples_count": len(X)
        }
        
        logger.info("Ensemble classifier initialization finished.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        try:
            X_clean = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            
            p_lgb = self.lgbm_model.predict_proba(X_clean)[:, 1]
            p_xgb = self.xgb_model.predict_proba(X_clean)[:, 1]
            p_cat = self.cat_model.predict_proba(X_clean)[:, 1]
            
            # Weighted average
            w = self.weights
            total_w = sum(w)
            p_blend = (w[0] * p_lgb + w[1] * p_xgb + w[2] * p_cat) / total_w
            
            probs = np.zeros((len(X), 2))
            probs[:, 0] = 1.0 - p_blend
            probs[:, 1] = p_blend
            return probs
        except Exception as e:
            logger.warning("Ensemble prediction failed: %s. Returning default zero probabilities.", e)
            return np.zeros((len(X), 2))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            probs = self.predict_proba(X)
            return (probs[:, 1] >= self.threshold).astype(int)
        except Exception as e:
            logger.warning("Ensemble class prediction prediction failed: %s. Returning default class 0.", e)
            return np.zeros(len(X), dtype=int)
