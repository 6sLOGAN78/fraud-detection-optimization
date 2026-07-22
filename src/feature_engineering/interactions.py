"""Interaction feature classes for vectorized operations, automatic discovery, explosion control, validation, and registry."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorizedInteractionEngine:
    """Computes stable pairwise interactions (multiplication, safety division, addition, subtraction) on numerical series."""
    def __init__(self, default_val: float = 0.0) -> None:
        self.default_val = default_val

    def compute_interaction(self, series_a: pd.Series, series_b: pd.Series, operation: str) -> pd.Series:
        # Align series index
        a_clean = series_a.fillna(0.0)
        b_clean = series_b.fillna(0.0)
        
        if operation == "multiplication":
            res = a_clean * b_clean
        elif operation == "division":
            # standard division, treating 0.0 as NaN to avoid division by zero
            denom_safe = b_clean.replace(0.0, np.nan)
            res = a_clean / denom_safe
            res = res.mask(a_clean == 0.0, 0.0)
        elif operation == "addition":
            res = a_clean + b_clean
        elif operation == "subtraction":
            res = a_clean - b_clean
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
        # Repair any NaNs or infinite values
        res = res.replace([np.inf, -np.inf], np.nan)
        res = res.fillna(self.default_val)
        return res.astype(float)


class AutomaticInteractionDiscoveryEngine:
    """Automatically pair-matches and discovers logical numerical interactions from combinations of base cols and baseline metrics."""
    def __init__(self, target_cols: list[str] | None = None) -> None:
        self.target_cols = target_cols or ["TransactionAmt", "dist1"]

    def discover_pairings(self, df_cols: list[str]) -> list[tuple[str, str, str, str]]:
        """Pairs base target numerical features with group aggregations with operations (multiplication and safety division)."""
        pairings = []
        
        for col_a in self.target_cols:
            if col_a not in df_cols:
                continue
            
            # Find group baseline metrics
            for col_b in df_cols:
                if col_b == col_a:
                    continue
                # Match baseline suffix columns
                if col_a in col_b and col_b.endswith(("_mean", "_median", "_std", "_roll_mean")):
                    # Multiplication
                    mult_name = f"{col_a}_x_{col_b}"
                    pairings.append((mult_name, col_a, col_b, "multiplication"))
                    # Safety division
                    div_name = f"{col_a}_div_{col_b}"
                    pairings.append((div_name, col_a, col_b, "division"))
                    
        return pairings


class FeatureExplosionController:
    """Screens interaction variables by variance or standard deviation thresholds to prune non-informative combinations."""
    def __init__(self, variance_threshold: float = 0.01) -> None:
        self.variance_threshold = variance_threshold

    def filter_features(self, df_interactions: pd.DataFrame) -> list[str]:
        """Returns the subset of features that pass the minimum variance threshold."""
        valid_cols = []
        for col in df_interactions.columns:
            if col == "TransactionID":
                valid_cols.append(col)
                continue
            
            # Calculate variance
            var_val = df_interactions[col].var()
            if pd.isna(var_val) or var_val < self.variance_threshold:
                logger.info("Dropping low variance feature: %s (var: %s)", col, var_val)
                continue
                
            valid_cols.append(col)
        return valid_cols


class InteractionValidationGate:
    """Validates computed interaction features checking for NaNs, Infs, duplicate indices or constant outputs."""
    def validate(self, df_inter: pd.DataFrame) -> dict[str, Any]:
        logger.info("Executing interaction validation checks...")
        
        nan_cols = [col for col in df_inter.columns if df_inter[col].isnull().any()]
        inf_cols = [col for col in df_inter.columns if np.isinf(df_inter[col]).any()]
        const_cols = [col for col in df_inter.columns if df_inter[col].nunique() <= 1]
        dup_cols = df_inter.columns[df_inter.columns.duplicated()].tolist()

        report = {
            "nan_columns_count": len(nan_cols),
            "nan_columns": nan_cols,
            "inf_columns_count": len(inf_cols),
            "inf_columns": inf_cols,
            "constant_columns_count": len(const_cols),
            "constant_columns": const_cols,
            "duplicate_columns_count": len(dup_cols),
            "duplicate_columns": dup_cols,
            "status": "PASS" if not (nan_cols or inf_cols or dup_cols) else "WARN",
        }
        
        logger.info("Validation Gate checks finished. Status: %s", report["status"])
        return report


class InteractionRegistry:
    """Manages metadata registry mapping interaction feature lineage, cataloging combinations and operation strategies."""
    def __init__(self) -> None:
        self.metadata: list[dict[str, Any]] = []

    def register(self, feature_name: str, col_a: str, col_b: str, operation: str) -> None:
        self.metadata.append({
            "feature_name": feature_name,
            "column_a": col_a,
            "column_b": col_b,
            "operation": operation,
            "created_at": pd.Timestamp.now().isoformat(),
        })

    def save_catalog(self, dest_dir: Path) -> tuple[Path, Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = dest_dir / "interaction_pipeline_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "registry": self.metadata,
                "version": "v1.0",
                "owner": "ML-Engineering-Team",
            }, f, indent=4)
            
        csv_path = dest_dir / "interaction_catalog.csv"
        pd.DataFrame(self.metadata).to_csv(csv_path, index=False)
        
        logger.info("Saved interaction manifest to %s and catalog to %s", manifest_path, csv_path)
        return manifest_path, csv_path
