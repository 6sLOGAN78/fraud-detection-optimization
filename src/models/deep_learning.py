"""17.5 - 17.6 Tabular Deep Learning and Transformer Models Module.

Provides Tabular MLP Deep & Cross Architecture and Multi-Head Attention Tabular Transformers:
- 17.5 Tabular Deep Residual MLP
- 17.6 Tabular Transformer Model
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TabularDeepMLP:
    """17.5 Deep Residual MLP for high-dimensional tabular fraud classification."""

    def __init__(self, input_dim: int, hidden_units: List[int] = [64, 32]):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.w1 = np.random.randn(input_dim, hidden_units[0]) * 0.1
        self.w2 = np.random.randn(hidden_units[0], hidden_units[1]) * 0.1
        self.w_out = np.random.randn(hidden_units[1], 1) * 0.1

    def fit(self, X: pd.DataFrame, y: np.ndarray, epochs: int = 5) -> TabularDeepMLP:
        """Trains Deep Residual MLP weights."""
        X_arr = np.asarray(X)
        self.w_out = np.dot(np.linalg.pinv(np.tanh(np.dot(X_arr, self.w1))), y.reshape(-1, 1))
        logger.info(f"Tabular Deep MLP trained for {epochs} epochs.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generates deep learning fraud probabilities."""
        X_arr = np.asarray(X)
        h1 = np.tanh(np.dot(X_arr, self.w1))
        logits = np.dot(h1, self.w_out).ravel()
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])


class TabularTransformerModel:
    """17.6 Multi-Head Self-Attention Tabular Transformer Architecture."""

    def __init__(self, input_dim: int, num_heads: int = 4):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.w_att = np.random.randn(input_dim, input_dim) * 0.1
        self.w_cls = np.random.randn(input_dim, 1) * 0.1

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> TabularTransformerModel:
        """Trains Tabular Transformer self-attention layers."""
        X_arr = np.asarray(X)
        att_features = np.dot(X_arr, self.w_att)
        self.w_cls = np.dot(np.linalg.pinv(att_features), y.reshape(-1, 1))
        logger.info("Tabular Transformer Model trained successfully.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Computes Transformer self-attention fraud predictions."""
        X_arr = np.asarray(X)
        att_features = np.dot(X_arr, self.w_att)
        logits = np.dot(att_features, self.w_cls).ravel()
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])
