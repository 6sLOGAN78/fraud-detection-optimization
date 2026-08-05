"""8.2 - 8.8 Classification Metrics Evaluation Modules.

Provides standard evaluation metric calculators:
- 8.2 ROC-AUC
- 8.3 PR-AUC (Average Precision)
- 8.4 Precision
- 8.5 Recall
- 8.6 F1-Score
- 8.7 Matthews Correlation Coefficient (MCC)
- 8.8 Kolmogorov-Smirnov (KS) Statistic
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from src.evaluation.framework import EvaluationFrameworkDesign, EvaluationImplementationStandards

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ROCAUCEngine:
    """8.2 ROC-AUC metric calculation and ROC curve curve coordinate generation."""

    def calculate(
        self, y_true: Union[pd.Series, np.ndarray], y_prob: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        val = EvaluationFrameworkDesign()
        yt, yp = val.validate_inputs(y_true, y_prob)

        score = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.5
        fpr, tpr, thresholds = roc_curve(yt, yp)

        return {
            "roc_auc": round(score, 5),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }


class PRAUCEngine:
    """8.3 PR-AUC (Precision-Recall Area Under Curve / Average Precision) calculation."""

    def calculate(
        self, y_true: Union[pd.Series, np.ndarray], y_prob: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        val = EvaluationFrameworkDesign()
        yt, yp = val.validate_inputs(y_true, y_prob)

        score = float(average_precision_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.0
        precision, recall, thresholds = precision_recall_curve(yt, yp)

        return {
            "pr_auc": round(score, 5),
            "precision_curve": precision.tolist(),
            "recall_curve": recall.tolist(),
            "thresholds": thresholds.tolist(),
        }


class PrecisionEngine:
    """8.4 Precision metric calculation at decision threshold."""

    def calculate(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        threshold: float = 0.5,
    ) -> float:
        val = EvaluationFrameworkDesign()
        yt, yp = val.validate_inputs(y_true, y_prob)
        y_pred = (yp >= threshold).astype(int)
        score = precision_score(yt, y_pred, zero_division=0)
        return float(score)


class RecallEngine:
    """8.5 Recall metric calculation at decision threshold."""

    def calculate(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        threshold: float = 0.5,
    ) -> float:
        val = EvaluationFrameworkDesign()
        yt, yp = val.validate_inputs(y_true, y_prob)
        y_pred = (yp >= threshold).astype(int)
        score = recall_score(yt, y_pred, zero_division=0)
        return float(score)


class F1ScoreEngine:
    """8.6 F1-Score metric calculation at decision threshold."""

    def calculate(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        threshold: float = 0.5,
    ) -> float:
        val = EvaluationFrameworkDesign()
        yt, yp = val.validate_inputs(y_true, y_prob)
        y_pred = (yp >= threshold).astype(int)
        score = f1_score(yt, y_pred, zero_division=0)
        return float(score)


class MCCEngine:
    """8.7 Matthews Correlation Coefficient (MCC) calculation."""

    def calculate(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        threshold: float = 0.5,
    ) -> float:
        val = EvaluationFrameworkDesign()
        yt, yp = val.validate_inputs(y_true, y_prob)
        y_pred = (yp >= threshold).astype(int)
        score = matthews_corrcoef(yt, y_pred)
        return float(score)


class KSStatisticEngine:
    """8.8 Kolmogorov-Smirnov (KS) statistic engine evaluating positive vs negative distribution separation."""

    def calculate(
        self, y_true: Union[pd.Series, np.ndarray], y_prob: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        val = EvaluationFrameworkDesign()
        yt, yp = val.validate_inputs(y_true, y_prob)

        pos_probs = yp[yt == 1]
        neg_probs = yp[yt == 0]

        if len(pos_probs) == 0 or len(neg_probs) == 0:
            return {"ks_statistic": 0.0, "p_value": 1.0, "max_ks_threshold": 0.5}

        res = ks_2samp(pos_probs, neg_probs)
        ks_stat = float(res.statistic)
        p_val = float(res.pvalue)

        # Calculate optimal threshold for max separation
        fpr, tpr, thresholds = roc_curve(yt, yp)
        ks_diffs = np.abs(tpr - fpr)
        max_idx = np.argmax(ks_diffs)
        max_ks_threshold = float(thresholds[max_idx]) if max_idx < len(thresholds) else 0.5

        return {
            "ks_statistic": round(ks_stat, 5),
            "p_value": float(p_val),
            "max_ks_threshold": round(max_ks_threshold, 4),
        }
