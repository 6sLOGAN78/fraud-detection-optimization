"""Difference feature classes for vectorized subtractions, automated discovery, stability gates, and registries."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorizedDifferenceEngine:
    """Computes stable absolute difference/deviation features handling null elements gracefully."""
    def __init__(self, default_val: float = 0.0) -> None:
        self.default_val = default_val

    def compute_difference(self, numerators: pd.Series, denominators: pd.Series) -> pd.Series:
        # Align series index
        num_clean = numerators.fillna(0.0)
        den_clean = denominators.fillna(0.0)
        
        diff = num_clean - den_clean
        
        # Repair any NaNs or infinite values
        diff = diff.replace([np.inf, -np.inf], np.nan)
        diff = diff.fillna(self.default_val)
        return diff.astype(float)


class AutomaticDifferenceDiscoveryEngine:
    """Automatically pair-matches and discovers logical difference combinations from numerical and aggregated columns."""
    def __init__(self, target_numerators: list[str] | None = None) -> None:
        self.target_numerators = target_numerators or ["TransactionAmt", "dist1", "dist2"]

    def discover_pairings(self, df_cols: list[str]) -> list[tuple[str, str, str]]:
        """Finds denoms ending in typical stats suffix that match numerator (e.g. card1_TransactionAmt_mean)."""
        pairings = []
        
        for numer in self.target_numerators:
            # Look for cols containing the numerator name and ending with stats suffixes
            for col in df_cols:
                if col == numer:
                    continue
                # E.g. card1_TransactionAmt_mean contains 'TransactionAmt' and ends with '_mean', '_median', etc.
                if numer in col:
                    if col.endswith(("_mean", "_median", "_std", "_min", "_max", "_roll_mean", "_exp_mean")):
                        # Form feature name
                        feat_name = f"{col}_diff"
                        pairings.append((feat_name, numer, col))
                        
        return pairings


class DifferenceValidationGate:
    """Validates computed difference features checking for NaNs, Infs, duplicate indices or constant outputs."""
    def validate(self, df_diff: pd.DataFrame) -> dict[str, Any]:
        logger.info("Executing difference validation checks...")
        
        nan_cols = [col for col in df_diff.columns if df_diff[col].isnull().any()]
        inf_cols = [col for col in df_diff.columns if np.isinf(df_diff[col]).any()]
        const_cols = [col for col in df_diff.columns if df_diff[col].nunique() <= 1]
        dup_cols = df_diff.columns[df_diff.columns.duplicated()].tolist()

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


class DifferenceRegistry:
    """Manages metadata registry mapping difference feature lineage, cataloging inputs and source fields."""
    def __init__(self) -> None:
        self.metadata: list[dict[str, Any]] = []

    def register(self, feature_name: str, numerator_col: str, denominator_col: str) -> None:
        self.metadata.append({
            "feature_name": feature_name,
            "numerator_column": numerator_col,
            "denominator_column": denominator_col,
            "created_at": pd.Timestamp.now().isoformat(),
        })

    def save_catalog(self, dest_dir: Path) -> tuple[Path, Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = dest_dir / "difference_pipeline_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "registry": self.metadata,
                "version": "v1.0",
                "owner": "ML-Engineering-Team",
            }, f, indent=4)
            
        csv_path = dest_dir / "difference_catalog.csv"
        pd.DataFrame(self.metadata).to_csv(csv_path, index=False)
        
        logger.info("Saved difference manifest to %s and catalog to %s", manifest_path, csv_path)
        return manifest_path, csv_path
