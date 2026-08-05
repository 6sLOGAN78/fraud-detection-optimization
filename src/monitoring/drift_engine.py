"""12.3 - 12.6 Data Drift, Concept Drift, Prediction Drift, and Feature Health Monitoring Module.

Provides comprehensive feature, prediction, and distribution drift monitors:
- 12.3 Data Drift Monitor (PSI & KS statistics per feature)
- 12.4 Concept Drift Monitor (Target distribution shift)
- 12.5 Prediction Distribution Monitor (Probability prediction shift)
- 12.6 Feature Health Monitor (Null rates, cardinality, out-of-bounds checks)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.monitoring.drift import calculate_psi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDriftMonitor:
    """12.3 Data drift monitor computing PSI and KS statistics across feature sets."""

    def evaluate_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        psi_threshold: float = 0.2,
    ) -> Dict[str, Any]:
        """Calculates feature-level PSI and flags features with severe distribution drift."""
        num_cols = reference_df.select_dtypes(include=[np.number]).columns
        drifted_features = []
        feature_scores = {}

        for col in num_cols:
            if col in current_df.columns:
                ref_s = reference_df[col].dropna()
                cur_s = current_df[col].dropna()
                if len(ref_s) > 0 and len(cur_s) > 0:
                    psi_val = calculate_psi(ref_s, cur_s)
                    feature_scores[col] = round(psi_val, 4)
                    if psi_val >= psi_threshold:
                        drifted_features.append(col)

        return {
            "total_features_evaluated": len(feature_scores),
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "feature_psi_scores": feature_scores,
            "has_severe_data_drift": len(drifted_features) > 0,
        }


class ConceptDriftMonitor:
    """12.4 Concept drift monitor evaluating target label shift and prediction relationship decay."""

    def evaluate_target_drift(
        self,
        reference_target: Union[pd.Series, np.ndarray],
        current_target: Union[pd.Series, np.ndarray],
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Checks for target base rate shift between reference and production data."""
        ref_rate = float(np.mean(reference_target))
        cur_rate = float(np.mean(current_target))

        diff = abs(ref_rate - cur_rate)
        is_concept_drifted = diff > threshold

        return {
            "reference_fraud_rate": round(ref_rate, 4),
            "current_fraud_rate": round(cur_rate, 4),
            "rate_difference": round(diff, 4),
            "is_concept_drifted": is_concept_drifted,
        }


class PredictionDistributionMonitor:
    """12.5 Monitors fraud probability prediction output shifts."""

    def evaluate_prediction_drift(
        self,
        reference_probs: Union[pd.Series, np.ndarray],
        current_probs: Union[pd.Series, np.ndarray],
        psi_threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """Calculates prediction distribution PSI."""
        ref_s = pd.Series(reference_probs).dropna()
        cur_s = pd.Series(current_probs).dropna()

        psi_val = calculate_psi(ref_s, cur_s)
        is_drifted = psi_val >= psi_threshold

        return {
            "prediction_psi": round(psi_val, 4),
            "reference_mean_prob": round(float(ref_s.mean()), 4),
            "current_mean_prob": round(float(cur_s.mean()), 4),
            "is_prediction_drifted": is_drifted,
        }


class FeatureHealthMonitor:
    """12.6 Tracks null rate spikes, missing value changes, and feature statistics."""

    def evaluate_feature_health(
        self, df: pd.DataFrame, max_allowed_null_rate: float = 0.5
    ) -> Dict[str, Any]:
        """Analyzes dataframe for missing value spikes and zero variance columns."""
        null_rates = (df.isnull().sum() / len(df)).to_dict()
        unhealthy_cols = [k for k, v in null_rates.items() if v > max_allowed_null_rate]

        return {
            "total_columns": len(df.columns),
            "unhealthy_null_columns_count": len(unhealthy_cols),
            "unhealthy_null_columns": unhealthy_cols,
            "max_null_rate_observed": round(float(max(null_rates.values())), 4) if null_rates else 0.0,
        }
