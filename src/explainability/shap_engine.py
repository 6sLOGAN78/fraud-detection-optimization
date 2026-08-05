"""9.2 - 9.6 SHAP Analysis & Feature Importance Engine Module.

Provides SHAP explanations, global & local feature importance, waterfall data, and force plot visualizations:
- 9.2 Global Feature Importance Engine
- 9.3 SHAP Analysis Engine
- 9.4 Local Explanations Engine
- 9.5 Waterfall Plots Engine
- 9.6 Force Plots Engine
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap

from src.explainability.architecture import ExplainabilityArchitectureDesign, ExplainabilityImplementationStandards

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlobalFeatureImportanceEngine:
    """9.2 Global Feature Importance engine calculating mean absolute SHAP values across dataset."""

    def calculate(self, model: Any, X: pd.DataFrame) -> Dict[str, float]:
        design = ExplainabilityArchitectureDesign()
        model, X_clean = design.validate_inputs(model, X)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_clean)
        except Exception:
            explainer = shap.Explainer(model, X_clean)
            shap_values = explainer(X_clean).values

        if isinstance(shap_values, list):
            shap_vals = np.abs(shap_values[1]).mean(axis=0)
        elif len(shap_values.shape) == 3:
            shap_vals = np.abs(shap_values[:, :, 1]).mean(axis=0)
        else:
            shap_vals = np.abs(shap_values).mean(axis=0)

        importance_dict = dict(zip(X_clean.columns, shap_vals.tolist()))
        sorted_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return ExplainabilityImplementationStandards.normalize_importance(sorted_dict)


class SHAPAnalysisEngine:
    """9.3 Full SHAP analysis engine computing SHAP matrix and summary metrics."""

    def calculate_shap_matrix(self, model: Any, X: pd.DataFrame) -> Dict[str, Any]:
        design = ExplainabilityArchitectureDesign()
        model, X_clean = design.validate_inputs(model, X)

        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_clean)
            expected_val = float(explainer.expected_value) if isinstance(explainer.expected_value, (int, float, np.number)) else float(explainer.expected_value[1])
        except Exception:
            explainer = shap.Explainer(model, X_clean)
            sv = explainer(X_clean)
            shap_vals = sv.values
            expected_val = float(np.mean(sv.base_values))

        if isinstance(shap_vals, list):
            vals = shap_vals[1]
        elif len(shap_vals.shape) == 3:
            vals = shap_vals[:, :, 1]
        else:
            vals = shap_vals

        return {
            "expected_value": expected_val,
            "shap_values": vals.tolist(),
            "feature_names": list(X_clean.columns),
        }


class LocalExplanationsEngine:
    """9.4 Local explanations engine providing feature contribution breakdown for individual transactions."""

    def explain_sample(
        self, model: Any, X: pd.DataFrame, sample_idx: int = 0
    ) -> Dict[str, Any]:
        shap_engine = SHAPAnalysisEngine()
        res = shap_engine.calculate_shap_matrix(model, X.iloc[[sample_idx]])

        shap_vec = res["shap_values"][0]
        feat_names = res["feature_names"]
        sample_vals = X.iloc[sample_idx][feat_names].to_dict()

        contributions = []
        for name, shap_val in zip(feat_names, shap_vec):
            contributions.append({
                "feature": name,
                "feature_value": float(sample_vals[name]),
                "shap_contribution": round(float(shap_val), 5),
            })

        contributions.sort(key=lambda x: abs(x["shap_contribution"]), reverse=True)

        return {
            "sample_index": sample_idx,
            "base_value": res["expected_value"],
            "contributions": contributions,
        }


class WaterfallPlotsEngine:
    """9.5 Waterfall plot data structure generator for step-by-step feature impact visualization."""

    def generate_plot_data(self, model: Any, X: pd.DataFrame, sample_idx: int = 0) -> Dict[str, Any]:
        local_engine = LocalExplanationsEngine()
        explanation = local_engine.explain_sample(model, X, sample_idx=sample_idx)

        base_val = explanation["base_value"]
        running_val = base_val
        steps = []

        for item in explanation["contributions"][:10]:  # Top 10 factors
            prev_val = running_val
            running_val += item["shap_contribution"]
            steps.append({
                "feature": item["feature"],
                "value": item["feature_value"],
                "delta": item["shap_contribution"],
                "start_value": round(prev_val, 4),
                "end_value": round(running_val, 4),
            })

        return {
            "sample_index": sample_idx,
            "base_value": round(base_val, 4),
            "final_prediction": round(running_val, 4),
            "waterfall_steps": steps,
        }


class ForcePlotsEngine:
    """9.6 Force plot data generator balancing positive and negative pushing forces."""

    def generate_force_data(self, model: Any, X: pd.DataFrame, sample_idx: int = 0) -> Dict[str, Any]:
        local_engine = LocalExplanationsEngine()
        explanation = local_engine.explain_sample(model, X, sample_idx=sample_idx)

        pos_forces = [c for c in explanation["contributions"] if c["shap_contribution"] > 0]
        neg_forces = [c for c in explanation["contributions"] if c["shap_contribution"] < 0]

        return {
            "sample_index": sample_idx,
            "base_value": explanation["base_value"],
            "positive_forces": pos_forces,
            "negative_forces": neg_forces,
        }
