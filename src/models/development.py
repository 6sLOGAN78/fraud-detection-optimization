"""Model Development Architecture classes for splitters, estimators, serialized models, and validation fallback pipelines."""

from __future__ import annotations

import logging
import time
import pickle
from typing import Any
from pathlib import Path

import pandas as pd
import numpy as np
import psutil
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDevelopmentArchitectureDesign:
    """Prepares and validates inputs, checks schema alignment, sets parameters, and calculates standard design transformations."""
    def __init__(self, threshold: float = 0.05, random_state: int = 42, n_jobs: int = -1, log_level: str = "INFO"):
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level
        self.scaler_mean_ = None
        self.scaler_std_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ModelDevelopmentArchitectureDesign:
        # Pre-computation validation gate
        if X is None or X.empty:
            raise ValueError("Input feature matrix cannot be empty or None")
        if y is not None and y.isnull().any():
            raise ValueError("Target labels contain NaN entries")
            
        # Ensure schema alignment and check duplicate columns
        if len(X.columns) != len(set(X.columns)):
            raise ValueError("Feature matrix contains duplicate column headers")

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.scaler_mean_ = X[numeric_cols].mean()
            self.scaler_std_ = X[numeric_cols].std().replace(0.0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_mean_ is None:
            raise ValueError("ModelDevelopmentArchitectureDesign must be fit before transform")
            
        X_out = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_out[numeric_cols] = (X[numeric_cols] - self.scaler_mean_) / self.scaler_std_
        return X_out


class CoreModelEngine:
    """Core estimator fit and predict execution wrapper monitoring memory, CPU time, and row drops."""
    def __init__(self, threshold: float = 0.05, random_state: int = 42, n_jobs: int = -1, log_level: str = "INFO"):
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level
        # Baseline model
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.fit_metrics_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> CoreModelEngine:
        # Handle zero-variance, singulars, and NaNs safely
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
        
        logger.info("CoreModelEngine fit complete in %.2fs. Memory Delta: %.2fMB", elapsed, mem_delta)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        return self.model.predict_proba(X_clean)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        return self.model.predict(X_clean)


class InputOutputProcessor:
    """Splits target vector and input arrays validation scopes memory-efficiently."""
    def __init__(self, threshold: float = 0.05, random_state: int = 42, n_jobs: int = -1, log_level: str = "INFO"):
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level

    def split_train_val(self, X: pd.DataFrame, y: pd.Series, val_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        # Segment splits index-aligned without overlapping to avoid data leakage
        split_idx = int(len(X) * (1 - val_ratio))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        return X_train, X_val, y_train, y_val


class ModelImplementationStandards:
    """Handles standard serialization, assertions, and safe fallback predictions on calculation failures."""
    def __init__(self, threshold: float = 0.05, random_state: int = 42, n_jobs: int = -1, log_level: str = "INFO"):
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level

    def serialize(self, bundle: dict[str, Any], filepath: str | Path) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(bundle, f)

    def deserialize(self, filepath: str | Path) -> dict[str, Any]:
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def predict_safe(self, model: Any, X: pd.DataFrame, fallback_val: float = 0.0) -> np.ndarray:
        try:
            if X is None or X.empty:
                raise ValueError("Input feature matrix is empty")
            X_clean = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            return model.predict_proba(X_clean)[:, 1]
        except Exception as e:
            logger.warning("Predict failed: %s. Reverting to fallback prediction.", e)
            return np.full(len(X), fallback_val)


class ModelDevelopmentPipeline:
    """Orchestrator pipeline wrapping design, processing, training, and standard checkpoints."""
    def __init__(self, threshold: float = 0.05, random_state: int = 42, n_jobs: int = -1, log_level: str = "INFO"):
        self.design = ModelDevelopmentArchitectureDesign(threshold, random_state, n_jobs, log_level)
        self.engine = CoreModelEngine(threshold, random_state, n_jobs, log_level)
        self.processor = InputOutputProcessor(threshold, random_state, n_jobs, log_level)
        self.standards = ModelImplementationStandards(threshold, random_state, n_jobs, log_level)

    def fit_and_validate(self, X: pd.DataFrame, y: pd.Series, val_ratio: float = 0.2) -> dict[str, Any]:
        logger.info("Executing Model Development Pipeline fitting...")
        
        # 1. Split training and validation sets
        X_train, X_val, y_train, y_val = self.processor.split_train_val(X, y, val_ratio)
        
        # 2. Fit design scaler transformations on train features
        self.design.fit(X_train, y_train)
        X_train_scaled = self.design.transform(X_train)
        X_val_scaled = self.design.transform(X_val)
        
        # 3. Fit Core Model Engine
        self.engine.fit(X_train_scaled, y_train)
        
        # 4. Score probabilities
        y_train_pred = self.standards.predict_safe(self.engine, X_train_scaled, fallback_val=0.0)
        y_val_pred = self.standards.predict_safe(self.engine, X_val_scaled, fallback_val=0.0)
        
        # Calculate scores
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        train_auc = roc_auc_score(y_train, y_train_pred) if len(np.unique(y_train)) > 1 else 0.5
        val_auc = roc_auc_score(y_val, y_val_pred) if len(np.unique(y_val)) > 1 else 0.5
        
        train_acc = accuracy_score(y_train, (y_train_pred >= 0.5).astype(int))
        val_acc = accuracy_score(y_val, (y_val_pred >= 0.5).astype(int))
        
        summary = {
            "train_auc": train_auc,
            "val_auc": val_auc,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "fit_metrics": self.engine.fit_metrics_
        }
        
        logger.info("Pipeline fit complete. Val AUC: %.4f | Val Accuracy: %.4f", val_auc, val_acc)
        return summary
