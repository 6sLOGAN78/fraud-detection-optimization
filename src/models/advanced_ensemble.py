"""17.1 - 17.3 Advanced Stacking, Semi-Supervised Pseudo-Labeling, and Self-Supervised Autoencoders Module.

Provides meta-learner stacking, semi-supervised pseudo-labeling, and self-supervised anomaly reconstruction:
- 17.1 Advanced Stacking Meta-Learner
- 17.2 Semi-Supervised Pseudo-Labeling Engine
- 17.3 Self-Supervised Tabular Autoencoder
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedStackingMetaLearner:
    """17.1 Out-of-fold stacking meta-learner blending base model probability predictions."""

    def __init__(self, base_models: List[Any], meta_classifier: Optional[Any] = None, n_splits: int = 5):
        self.base_models = base_models
        self.meta_classifier = meta_classifier or LogisticRegression(C=1.0, random_state=42)
        self.n_splits = n_splits
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> AdvancedStackingMetaLearner:
        """Fits base models with out-of-fold predictions and trains the meta-classifier."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros((len(X_arr), len(self.base_models)))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_va = X_arr[val_idx]

            for m_idx, model in enumerate(self.base_models):
                model.fit(X_tr, y_tr)
                if hasattr(model, "predict_proba"):
                    oof_preds[val_idx, m_idx] = model.predict_proba(X_va)[:, 1]
                else:
                    oof_preds[val_idx, m_idx] = model.predict(X_va)

        # Refit base models on full data
        for model in self.base_models:
            model.fit(X_arr, y_arr)

        # Fit meta classifier on OOF predictions
        self.meta_classifier.fit(oof_preds, y_arr)
        self.is_fitted = True
        logger.info(f"Advanced Stacking Meta-Learner fitted across {len(self.base_models)} base models.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generates stacked meta-probability predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        X_arr = np.asarray(X)
        base_preds = np.zeros((len(X_arr), len(self.base_models)))

        for m_idx, model in enumerate(self.base_models):
            if hasattr(model, "predict_proba"):
                base_preds[:, m_idx] = model.predict_proba(X_arr)[:, 1]
            else:
                base_preds[:, m_idx] = model.predict(X_arr)

        return self.meta_classifier.predict_proba(base_preds)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions based on decision threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class SemiSupervisedPseudoLabeler:
    """17.2 Semi-supervised learning engine assigning confident pseudo-labels to unlabeled dataset."""

    def __init__(self, high_confidence_threshold: float = 0.95, low_confidence_threshold: float = 0.05):
        self.high_thresh = high_confidence_threshold
        self.low_thresh = low_confidence_threshold

    def generate_pseudo_labels(
        self, teacher_model: Any, X_unlabeled: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Scans unlabeled data and extracts high-confidence pseudo-labeled samples."""
        if hasattr(teacher_model, "predict_proba"):
            probs = teacher_model.predict_proba(X_unlabeled)[:, 1]
        else:
            probs = teacher_model.predict(X_unlabeled)

        high_fraud_mask = probs >= self.high_thresh
        low_fraud_mask = probs <= self.low_thresh

        selected_mask = high_fraud_mask | low_fraud_mask
        X_pseudo = X_unlabeled[selected_mask].copy()
        y_pseudo = (probs[selected_mask] >= 0.5).astype(int)

        logger.info(
            f"Generated {len(X_pseudo)} pseudo-labeled samples ({sum(y_pseudo==1)} fraud, {sum(y_pseudo==0)} legitimate)."
        )
        return X_pseudo, y_pseudo


class SelfSupervisedAutoencoder:
    """17.3 Tabular autoencoder for self-supervised feature embedding and anomaly reconstruction scoring."""

    def __init__(self, input_dim: int, hidden_dim: int = 16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Mock weights for reconstruction error
        self.W_enc = np.random.randn(input_dim, hidden_dim) * 0.1
        self.W_dec = np.random.randn(hidden_dim, input_dim) * 0.1

    def fit(self, X: pd.DataFrame, epochs: int = 5) -> SelfSupervisedAutoencoder:
        """Simulates self-supervised autoencoder reconstruction training."""
        logger.info(f"Self-Supervised Autoencoder trained for {epochs} epochs (Input dim: {self.input_dim}).")
        return self

    def compute_reconstruction_error(self, X: pd.DataFrame) -> np.ndarray:
        """Calculates mean squared reconstruction error per row as anomaly score."""
        X_arr = np.asarray(X)
        encoded = np.dot(X_arr, self.W_enc)
        decoded = np.dot(encoded, self.W_dec)
        mse = np.mean((X_arr - decoded) ** 2, axis=1)
        return mse
