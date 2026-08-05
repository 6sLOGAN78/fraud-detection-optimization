"""9.10 - 9.12 Model Transparency, Fairness, and Explainability Reports Module.

Provides model transparency, fairness & bias metrics, and automated report generation:
- 9.10 Model Transparency Engine (Surrogate decision rules extraction)
- 9.11 Fairness & Bias Assessment Engine (Disparate Impact, Demographic Parity)
- 9.12 Explainability Reports Generator
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from src.explainability.architecture import ExplainabilityArchitectureDesign, ExplainabilityImplementationStandards

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTransparencyEngine:
    """9.10 Model Transparency engine extracting human-interpretable surrogate decision rules."""

    def extract_surrogate_rules(
        self, model: Any, X: pd.DataFrame, max_depth: int = 3
    ) -> Dict[str, Any]:
        design = ExplainabilityArchitectureDesign()
        model, X_clean = design.validate_inputs(model, X)

        if hasattr(model, "predict_proba"):
            y_pseudo = (model.predict_proba(X_clean)[:, 1] >= 0.5).astype(int)
        else:
            y_pseudo = model.predict(X_clean)

        surrogate = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        surrogate.fit(X_clean, y_pseudo)

        tree_rules = export_text(surrogate, feature_names=list(X_clean.columns))
        surrogate_accuracy = float(surrogate.score(X_clean, y_pseudo))

        return {
            "surrogate_max_depth": max_depth,
            "surrogate_fidelity_score": round(surrogate_accuracy, 4),
            "extracted_rules_text": tree_rules,
        }


class FairnessBiasAssessmentEngine:
    """9.11 Fairness & Bias Assessment engine evaluating Disparate Impact Ratio and Equalized Odds across demographic groups or transaction slices."""

    def evaluate_fairness(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        protected_attribute: Union[pd.Series, np.ndarray],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        yt = np.asarray(y_true, dtype=int).ravel()
        yp = np.asarray(y_prob, dtype=float).ravel()
        pa = np.asarray(protected_attribute).ravel()

        y_pred = (yp >= threshold).astype(int)

        groups = np.unique(pa)
        if len(groups) < 2:
            return {"error": "Protected attribute must have at least 2 distinct groups."}

        group_stats = {}
        selection_rates = {}

        for g in groups:
            mask = pa == g
            g_yt = yt[mask]
            g_ypred = y_pred[mask]

            pos_rate = float(np.mean(g_ypred))
            selection_rates[str(g)] = pos_rate

            tpr = float(np.mean(g_ypred[g_yt == 1])) if np.sum(g_yt == 1) > 0 else 0.0
            fpr = float(np.mean(g_ypred[g_yt == 0])) if np.sum(g_yt == 0) > 0 else 0.0

            group_stats[str(g)] = {
                "sample_count": int(np.sum(mask)),
                "positive_rate": round(pos_rate, 4),
                "true_positive_rate": round(tpr, 4),
                "false_positive_rate": round(fpr, 4),
            }

        rates = list(selection_rates.values())
        min_rate, max_rate = min(rates), max(rates)
        disparate_impact_ratio = round(
            ExplainabilityImplementationStandards.safe_divide(min_rate, max_rate, default=1.0), 4
        )

        return {
            "disparate_impact_ratio": disparate_impact_ratio,
            "group_metrics": group_stats,
            "fair_by_80_percent_rule": disparate_impact_ratio >= 0.8,
        }


class ExplainabilityReporter:
    """9.12 Explainability Reports engine generating consolidated JSON/HTML interpretability reports."""

    def generate_report(
        self,
        model_name: str,
        global_importance: Dict[str, float],
        transparency_rules: Dict[str, Any],
        fairness_results: Optional[Dict[str, Any]] = None,
        output_dir: Union[str, Path] = "reports/explainability",
    ) -> Path:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        report_data = {
            "model_name": model_name,
            "global_feature_importance": global_importance,
            "transparency": transparency_rules,
            "fairness_bias": fairness_results or {},
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        json_file = out_path / f"{model_name}_explainability_report.json"
        with open(json_file, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Successfully generated Explainability Report at {json_file}")
        return json_file
