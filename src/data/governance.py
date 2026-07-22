"""Feature Governance engine verifying feature data quality metrics, thresholds, and updating registry lifecycle states accordingly."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.data.store import FeatureRegistry, FeatureViewMetadata


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureGovernanceEngine:
    """Manages feature catalog health by running validations on nulls, drift (PSI), and variance, setting lifecycle states."""
    def __init__(self, registry_path: Path) -> None:
        self.registry_path = Path(registry_path)
        self.registry = FeatureRegistry(self.registry_path)

    @staticmethod
    def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """Vectorized Population Stability Index calculation comparing expected and actual feature vectors."""
        # Clean nulls
        exp = expected.dropna().values
        act = actual.dropna().values

        if len(exp) == 0 or len(act) == 0:
            return 0.0

        try:
            # Create quantiles/bins based on expected values
            percentiles = np.linspace(0, 100, bins + 1)
            bin_edges = np.percentile(exp, percentiles)
            # Ensure unique edges to avoid bin errors
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                return 0.0

            # Calculate bucket distributions
            exp_counts, _ = np.histogram(exp, bins=bin_edges)
            act_counts, _ = np.histogram(act, bins=bin_edges)

            # Convert counts to fractions
            exp_probs = exp_counts / len(exp)
            act_probs = act_counts / len(act)

            # Smooth probability distributions to avoid division by zero
            exp_probs = np.where(exp_probs == 0, 1e-4, exp_probs)
            act_probs = np.where(act_probs == 0, 1e-4, act_probs)

            # Re-normalize to sum to 1
            exp_probs /= exp_probs.sum()
            act_probs /= act_probs.sum()

            # Calculate PSI
            psi = np.sum((act_probs - exp_probs) * np.log(act_probs / exp_probs))
            return float(psi)
        except Exception as e:
            logger.warning("Error calculating PSI in governance model: %s", e)
            return 0.0

    def audit_features(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        null_threshold: float = 0.05,
        drift_threshold: float = 0.25,
        token: str | None = None
    ) -> dict[str, Any]:
        """Runs governance check suite over registered features, updating their status metadata configs."""
        report: dict[str, Any] = {
            "audited_views_count": len(self.registry.views),
            "status": "PASS",
            "view_logs": []
        }
        
        has_critical_failures = False
        
        for view_name, metadata in list(self.registry.views.items()):
            view_log = {
                "view_name": view_name,
                "features_audited": [],
                "actions_taken": [],
                "status": "PASS"
            }
            
            entity_id = metadata.entity_id
            features = metadata.features
            
            # Check presence of entity and features in dataframes
            for df_name, df in [("train", df_train), ("test", df_test)]:
                if entity_id not in df.columns:
                    raise KeyError(f"Entity index '{entity_id}' not found in {df_name} dataframe")
                    
            for feature in features:
                feature_log = {
                    "feature_name": feature,
                    "null_ratio_train": 0.0,
                    "null_ratio_test": 0.0,
                    "variance_train": 0.0,
                    "psi_drift": 0.0,
                    "checks": {
                        "missingness": "PASS",
                        "variance": "PASS",
                        "drift": "PASS"
                    }
                }
                
                # Verify features exist in input columns
                for df_name, df in [("train", df_train), ("test", df_test)]:
                    if feature not in df.columns:
                        raise KeyError(f"Feature '{feature}' not found in {df_name} dataframe")
                        
                # 1. Check missingness ratio
                trn_null = float(df_train[feature].isnull().mean())
                tst_null = float(df_test[feature].isnull().mean())
                feature_log["null_ratio_train"] = trn_null
                feature_log["null_ratio_test"] = tst_null
                
                if trn_null > null_threshold or tst_null > null_threshold:
                    feature_log["checks"]["missingness"] = "FAIL"
                    logger.warning("Feature Governance Alert: Feature '%s' in view '%s' exceeded null threshold (%s)", feature, view_name, null_threshold)
                    
                # 2. Check variance
                try:
                    trn_var = float(df_train[feature].var(ddof=0))
                except Exception:
                    trn_var = 0.0
                feature_log["variance_train"] = trn_var
                
                if np.isnan(trn_var) or trn_var == 0.0:
                    feature_log["checks"]["variance"] = "FAIL"
                    logger.warning("Feature Governance Alert: Feature '%s' in view '%s' has zero or undefined variance", feature, view_name)
                    
                # 3. Check PSI drift
                psi_val = self.calculate_psi(df_train[feature], df_test[feature])
                feature_log["psi_drift"] = psi_val
                
                if psi_val > drift_threshold:
                    feature_log["checks"]["drift"] = "FAIL"
                    logger.warning("Feature Governance Alert: Feature '%s' in view '%s' has high drift PSI=%s (limit=%s)", feature, view_name, psi_val, drift_threshold)
                
                # Determine final status for this feature
                fails = [check for check, stat in feature_log["checks"].items() if stat == "FAIL"]
                if fails:
                    has_critical_failures = True
                    view_log["status"] = "FAIL"
                    view_log["actions_taken"].append(f"Flagged feature '{feature}' as CRITICAL due to: {', '.join(fails)}")
                
                view_log["features_audited"].append(feature_log)
                
            # If a view has failures, we adjust its governance tag to mark it
            if view_log["status"] == "FAIL":
                # Add lifecycle tag / metadata tags
                if not metadata.tags:
                    metadata.tags = []
                if "CRITICAL" not in metadata.tags:
                    metadata.tags.append("CRITICAL")
                    self.registry.views[view_name] = metadata
            
            report["view_logs"].append(view_log)
            
        if has_critical_failures:
            report["status"] = "WARNING"
            
        # Write registry updates if dirty
        self.registry.save()
        return report
