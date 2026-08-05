"""8.9 - 8.12 Diagnostic Evaluation Modules.

Provides diagnostic evaluation engines:
- 8.9 Calibration Analysis Engine (ECE, Brier score, reliability curve)
- 8.10 Lift & Gain Charts Engine (Cumulative gain & lift curves)
- 8.11 Threshold Analysis Engine (Optimal threshold sweep across F1, F-beta, Cost)
- 8.12 Confusion Matrix Engine (TP, FP, TN, FN, FPR, FNR, FDR calculations)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix

from src.evaluation.framework import EvaluationFrameworkDesign, EvaluationImplementationStandards

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibrationAnalysisEngine:
    """8.9 Calibration analysis computing Expected Calibration Error (ECE) and reliability curve data."""

    def calculate(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        val = EvaluationFrameworkDesign()
        yt, yp = val.validate_inputs(y_true, y_prob)

        brier = float(brier_score_loss(yt, yp))
        prob_true, prob_pred = calibration_curve(yt, yp, n_bins=n_bins, strategy="uniform")

        # Expected Calibration Error (ECE)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(yp, bins) - 1
        ece = 0.0
        n_samples = len(yp)

        for i in range(n_bins):
            mask = bin_ids == i
            if np.any(mask):
                bin_size = np.sum(mask)
                bin_acc = np.mean(yt[mask])
                bin_conf = np.mean(yp[mask])
                ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

        return {
            "brier_score": round(brier, 5),
            "expected_calibration_error": round(float(ece), 5),
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }


class LiftGainEngine:
    """8.10 Cumulative Gain and Lift chart generator across deciles."""

    def calculate(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        n_deciles: int = 10,
    ) -> Dict[str, Any]:
        val = EvaluationFrameworkDesign()
        yt, yp = val.validate_inputs(y_true, y_prob)

        df = pd.DataFrame({"y_true": yt, "y_prob": yp})
        df = df.sort_values(by="y_prob", ascending=False).reset_index(drop=True)

        df["decile"] = pd.qcut(df.index, q=n_deciles, labels=False, duplicates="drop")
        total_positives = df["y_true"].sum()

        decile_stats = []
        cumulative_positives = 0
        cumulative_count = 0

        for d, group in df.groupby("decile"):
            count = len(group)
            positives = group["y_true"].sum()
            cumulative_positives += positives
            cumulative_count += count

            cum_gain = EvaluationImplementationStandards.safe_divide(
                cumulative_positives, total_positives
            )
            base_rate = EvaluationImplementationStandards.safe_divide(total_positives, len(df))
            decile_rate = EvaluationImplementationStandards.safe_divide(positives, count)
            lift = EvaluationImplementationStandards.safe_divide(decile_rate, base_rate)

            decile_stats.append({
                "decile": int(d + 1),
                "count": int(count),
                "positives": int(positives),
                "cumulative_gain": round(cum_gain, 4),
                "lift": round(lift, 4),
            })

        return {
            "total_samples": len(df),
            "total_positives": int(total_positives),
            "deciles": decile_stats,
        }


class ThresholdAnalysisEngine:
    """8.11 Sweeps threshold range to find optimal cutoff for F1, F-beta, or custom cost criteria."""

    def sweep(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        beta: float = 1.0,
        cost_fp: float = 10.0,
        cost_fn: float = 100.0,
    ) -> Dict[str, Any]:
        val = EvaluationFrameworkDesign()
        yt, yp = val.validate_inputs(y_true, y_prob)

        thresholds = np.linspace(0.01, 0.99, 99)
        best_fbeta = -1.0
        best_fbeta_thresh = 0.5
        min_cost = float("inf")
        min_cost_thresh = 0.5

        history = []

        for t in thresholds:
            y_pred = (yp >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(yt, y_pred, labels=[0, 1]).ravel()

            prec = EvaluationImplementationStandards.safe_divide(tp, tp + fp)
            rec = EvaluationImplementationStandards.safe_divide(tp, tp + fn)
            fbeta = EvaluationImplementationStandards.safe_divide(
                (1 + beta**2) * prec * rec, (beta**2 * prec) + rec
            )

            total_cost = (fp * cost_fp) + (fn * cost_fn)

            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_fbeta_thresh = t

            if total_cost < min_cost:
                min_cost = total_cost
                min_cost_thresh = t

            history.append({
                "threshold": round(float(t), 2),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "fbeta": round(fbeta, 4),
                "total_cost": round(float(total_cost), 2),
            })

        return {
            "best_fbeta_threshold": round(float(best_fbeta_thresh), 2),
            "best_fbeta_score": round(float(best_fbeta), 4),
            "min_cost_threshold": round(float(min_cost_thresh), 2),
            "min_cost_value": round(float(min_cost), 2),
            "sweep_history": history,
        }


class ConfusionMatrixEngine:
    """8.12 Detailed confusion matrix breakdown metrics (TP, FP, TN, FN, FPR, FNR, FDR, NPV)."""

    def calculate(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        val = EvaluationFrameworkDesign()
        yt, yp = val.validate_inputs(y_true, y_prob)
        y_pred = (yp >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(yt, y_pred, labels=[0, 1]).ravel()

        fpr = EvaluationImplementationStandards.safe_divide(fp, fp + tn)
        fnr = EvaluationImplementationStandards.safe_divide(fn, fn + tp)
        fdr = EvaluationImplementationStandards.safe_divide(fp, fp + tp)
        npv = EvaluationImplementationStandards.safe_divide(tn, tn + fn)

        return {
            "threshold": threshold,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "false_positive_rate": round(fpr, 5),
            "false_negative_rate": round(fnr, 5),
            "false_discovery_rate": round(fdr, 5),
            "negative_predictive_value": round(npv, 5),
        }
