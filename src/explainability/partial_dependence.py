"""9.7 - 9.9 Partial Dependence & Feature Interaction Explanations Engine Module.

Provides Partial Dependence Plots (PDP), Individual Conditional Expectation (ICE),
and 2D Feature Interaction analysis:
- 9.7 Partial Dependence Plots (PDP) Engine
- 9.8 Individual Conditional Expectation (ICE) Engine
- 9.9 Feature Interaction Explanations Engine
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

from src.explainability.architecture import ExplainabilityArchitectureDesign

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PartialDependenceEngine:
    """9.7 Partial Dependence Plot (PDP) engine calculating marginal feature effects."""

    def calculate_pdp(
        self, model: Any, X: pd.DataFrame, feature_name: str, grid_resolution: int = 20
    ) -> Dict[str, Any]:
        design = ExplainabilityArchitectureDesign()
        model, X_clean = design.validate_inputs(model, X)

        if feature_name not in X_clean.columns:
            raise ValueError(f"Feature '{feature_name}' not found in DataFrame.")

        res = partial_dependence(
            model, X_clean, features=[feature_name], grid_resolution=grid_resolution, kind="average"
        )

        grid_values = res["grid_values"][0].tolist()
        pdp_values = res["average"][0].tolist()

        return {
            "feature": feature_name,
            "grid_values": grid_values,
            "pdp_values": pdp_values,
        }


class IndividualConditionalExpectationEngine:
    """9.8 Individual Conditional Expectation (ICE) engine evaluating individual instance trajectories."""

    def calculate_ice(
        self, model: Any, X: pd.DataFrame, feature_name: str, grid_resolution: int = 15
    ) -> Dict[str, Any]:
        design = ExplainabilityArchitectureDesign()
        model, X_clean = design.validate_inputs(model, X)

        if feature_name not in X_clean.columns:
            raise ValueError(f"Feature '{feature_name}' not found in DataFrame.")

        res = partial_dependence(
            model, X_clean, features=[feature_name], grid_resolution=grid_resolution, kind="individual"
        )

        grid_values = res["grid_values"][0].tolist()
        ice_curves = res["individual"][0].tolist()

        return {
            "feature": feature_name,
            "grid_values": grid_values,
            "ice_curves": ice_curves,
        }


class FeatureInteractionExplanationsEngine:
    """9.9 2D Partial Dependence & Feature Interaction Explanations Engine."""

    def calculate_2d_pdp(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_tuple: Tuple[str, str],
        grid_resolution: int = 10,
    ) -> Dict[str, Any]:
        design = ExplainabilityArchitectureDesign()
        model, X_clean = design.validate_inputs(model, X)

        f1, f2 = feature_tuple
        if f1 not in X_clean.columns or f2 not in X_clean.columns:
            raise ValueError(f"Features {feature_tuple} not found in DataFrame.")

        res = partial_dependence(
            model, X_clean, features=[(f1, f2)], grid_resolution=grid_resolution, kind="average"
        )

        grid_f1 = res["grid_values"][0].tolist()
        grid_f2 = res["grid_values"][1].tolist()
        pdp_2d_matrix = res["average"][0].tolist()

        return {
            "feature_1": f1,
            "feature_2": f2,
            "grid_1": grid_f1,
            "grid_2": grid_f2,
            "pdp_2d": pdp_2d_matrix,
        }
