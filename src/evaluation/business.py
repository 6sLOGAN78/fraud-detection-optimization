"""8.13 - 8.15 Error Analysis, Robustness & Business KPI Evaluation Modules.

Provides specialized evaluation engines:
- 8.13 Error Analysis Engine (hard false positives / false negatives profiling)
- 8.14 Robustness & Stress Evaluation Engine (feature noise/perturbation resilience)
- 8.15 Business KPI & Financial Cost-Loss Evaluation Engine (chargebacks, manual review costs, friction loss)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.evaluation.framework import EvaluationFrameworkDesign, EvaluationImplementationStandards

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorAnalysisEngine:
    """8.13 Error analysis engine identifying hardest false positives and false negatives."""

    def analyze(
        self,
        X: pd.DataFrame,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        threshold: float = 0.5,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        val = EvaluationFrameworkDesign()
        yt, yp = val.validate_inputs(y_true, y_prob)

        df = X.copy()
        df["y_true"] = yt
        df["y_prob"] = yp
        df["error"] = np.abs(df["y_true"] - df["y_prob"])

        # False Positives: y_true=0, high y_prob
        fps = df[(df["y_true"] == 0) & (df["y_prob"] >= threshold)].sort_values(
            by="y_prob", ascending=False
        )

        # False Negatives: y_true=1, low y_prob
        fns = df[(df["y_true"] == 1) & (df["y_prob"] < threshold)].sort_values(
            by="y_prob", ascending=True
        )

        return {
            "total_false_positives": len(fps),
            "total_false_negatives": len(fns),
            "top_false_positive_indices": fps.index[:top_k].tolist(),
            "top_false_negative_indices": fns.index[:top_k].tolist(),
            "mean_fp_confidence": round(float(fps["y_prob"].mean()), 4) if len(fps) > 0 else 0.0,
            "mean_fn_confidence": round(float(fns["y_prob"].mean()), 4) if len(fns) > 0 else 0.0,
        }


class RobustnessEvaluationEngine:
    """8.14 Evaluates model stability under feature noise injection and missing value perturbation."""

    def evaluate_perturbation(
        self,
        model: Any,
        X: pd.DataFrame,
        y_true: Union[pd.Series, np.ndarray],
        noise_level: float = 0.05,
    ) -> Dict[str, Any]:
        val = EvaluationFrameworkDesign()

        if hasattr(model, "predict_proba"):
            base_probs = model.predict_proba(X)[:, 1]
        else:
            base_probs = model.predict(X)

        yt, base_yp = val.validate_inputs(y_true, base_probs)

        # Inject Gaussian noise into numerical features
        X_noisy = X.copy()
        num_cols = X_noisy.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            stds = X_noisy[num_cols].std().values
            stds = np.where(stds == 0, 1.0, stds)
            noise = np.random.normal(0, noise_level * stds, size=X_noisy[num_cols].shape)
            X_noisy[num_cols] = X_noisy[num_cols] + noise

        if hasattr(model, "predict_proba"):
            noisy_probs = model.predict_proba(X_noisy)[:, 1]
        else:
            noisy_probs = model.predict(X_noisy)

        _, noisy_yp = val.validate_inputs(y_true, noisy_probs)

        prob_delta = np.abs(base_yp - noisy_yp)
        mean_abs_error = float(np.mean(prob_delta))
        max_abs_error = float(np.max(prob_delta))

        return {
            "noise_level": noise_level,
            "mean_probability_shift": round(mean_abs_error, 5),
            "max_probability_shift": round(max_abs_error, 5),
            "robustness_score": round(float(1.0 - mean_abs_error), 5),
        }


class BusinessKPIEvaluator:
    """8.15 Financial Cost-Loss & Business KPI evaluator calculating net savings, chargeback costs, and review friction."""

    def calculate(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        amounts: Union[pd.Series, np.ndarray],
        threshold: float = 0.5,
        review_cost_per_tx: float = 2.0,
        chargeback_fee: float = 15.0,
    ) -> Dict[str, Any]:
        val = EvaluationFrameworkDesign()
        yt, yp = val.validate_inputs(y_true, y_prob)
        tx_amounts = np.asarray(amounts, dtype=float).ravel()

        if len(tx_amounts) != len(yt):
            raise ValueError("Length mismatch between target vector and transaction amounts.")

        y_pred = (yp >= threshold).astype(int)

        # TP: Caught Fraud (Savings = Transaction Amount - Review Cost)
        tp_mask = (yt == 1) & (y_pred == 1)
        # FN: Missed Fraud (Loss = Transaction Amount + Chargeback Fee)
        fn_mask = (yt == 1) & (y_pred == 0)
        # FP: False Alarm (Loss = Review Cost + Customer Friction)
        fp_mask = (yt == 0) & (y_pred == 1)
        # TN: Legitimate Transaction (Zero Loss)
        tn_mask = (yt == 0) & (y_pred == 0)

        fraud_prevented_value = float(np.sum(tx_amounts[tp_mask]))
        fraud_missed_loss = float(np.sum(tx_amounts[fn_mask]) + (np.sum(fn_mask) * chargeback_fee))
        false_positive_cost = float(np.sum(fp_mask) * review_cost_per_tx)
        total_review_cost = float((np.sum(tp_mask) + np.sum(fp_mask)) * review_cost_per_tx)

        net_savings = fraud_prevented_value - (false_positive_cost + total_review_cost)

        return {
            "threshold": threshold,
            "fraud_prevented_value": round(fraud_prevented_value, 2),
            "fraud_missed_loss": round(fraud_missed_loss, 2),
            "false_positive_cost": round(false_positive_cost, 2),
            "total_review_cost": round(total_review_cost, 2),
            "net_financial_savings": round(net_savings, 2),
        }
