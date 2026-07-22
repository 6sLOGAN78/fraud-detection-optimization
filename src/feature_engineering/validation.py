"""Feature validation engines including schema alignment, datatype verification, missingness tests, statistical range gates, outlier checks, and pipeline validation report builder."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchemaValidator:
    """Checks columns for existence, matches expected data types, and flags deviations."""
    def validate(self, df: pd.DataFrame, expected_types: dict[str, str]) -> dict[str, Any]:
        missing_cols = []
        dtype_mismatches = {}
        for col, expected_type in expected_types.items():
            if col not in df.columns:
                missing_cols.append(col)
                continue
                
            actual_type = str(df[col].dtype)
            # Map simple classifications
            is_numeric = "float" in actual_type or "int" in actual_type
            is_cat = "object" in actual_type or "category" in actual_type or "bool" in actual_type
            
            if expected_type == "numeric" and not is_numeric:
                dtype_mismatches[col] = {"expected": expected_type, "actual": actual_type}
            elif expected_type == "categorical" and not is_cat:
                dtype_mismatches[col] = {"expected": expected_type, "actual": actual_type}

        return {
            "missing_columns_count": len(missing_cols),
            "missing_columns": missing_cols,
            "dtype_mismatches_count": len(dtype_mismatches),
            "dtype_mismatches": dtype_mismatches,
            "status": "PASS" if not (missing_cols or dtype_mismatches) else "FAIL",
        }


class MissingValueValidator:
    """Verifies missingness percentages against thresholds and detects infinite values."""
    def validate(self, df: pd.DataFrame, thresholds: dict[str, float] = None, default_max_missing: float = 0.95) -> dict[str, Any]:
        thresholds = thresholds or {}
        high_missingness = {}
        inf_cols = []
        
        for col in df.columns:
            # Check NaN missing percentage
            missing_pct = df[col].isnull().mean()
            limit = thresholds.get(col, default_max_missing)
            if missing_pct > limit:
                high_missingness[col] = {"missing_ratio": missing_pct, "limit": limit}
                
            # Check Infinity values (only applicable to numeric columns)
            if "float" in str(df[col].dtype) or "int" in str(df[col].dtype):
                if np.isinf(df[col]).any():
                    inf_cols.append(col)

        return {
            "high_missingness_count": len(high_missingness),
            "high_missingness": high_missingness,
            "infinite_columns_count": len(inf_cols),
            "infinite_columns": inf_cols,
            "status": "PASS" if not (high_missingness or inf_cols) else "WARN",
        }


class StatisticalValidator:
    """Validates features for numerical ranges, zero variance constants, and outlier frequencies."""
    def validate(self, df: pd.DataFrame, range_bounds: dict[str, tuple[float, float]] = None, outlier_sigma: float = 4.0) -> dict[str, Any]:
        range_bounds = range_bounds or {}
        out_of_bounds = {}
        constant_cols = []
        outliers = {}
        
        for col in df.columns:
            if "float" in str(df[col].dtype) or "int" in str(df[col].dtype):
                series_clean = df[col].dropna()
                if series_clean.empty:
                    continue
                    
                # Constness check
                if series_clean.nunique() <= 1:
                    constant_cols.append(col)
                    continue
                
                # Bounds check
                if col in range_bounds:
                    min_val, max_val = range_bounds[col]
                    val_min = series_clean.min()
                    val_max = series_clean.max()
                    if val_min < min_val or val_max > max_val:
                        out_of_bounds[col] = {"min": float(val_min), "max": float(val_max), "bounds": [min_val, max_val]}
                
                # Outlier check using standard deviation
                mean_val = series_clean.mean()
                std_val = series_clean.std()
                if std_val > 0.0:
                    z_scores = np.abs((series_clean - mean_val) / std_val)
                    outliers_count = int((z_scores > outlier_sigma).sum())
                    if outliers_count > 0:
                        outliers[col] = {"count": outliers_count, "ratio": outliers_count / len(series_clean)}

        return {
            "constant_columns_count": len(constant_cols),
            "constant_columns": constant_cols,
            "out_of_bounds_count": len(out_of_bounds),
            "out_of_bounds": out_of_bounds,
            "outliers_detected_count": len(outliers),
            "outliers": outliers,
            "status": "PASS" if not (out_of_bounds or constant_cols) else "WARN",
        }


class FeatureValidationPipeline:
    """Unified validation pipeline coordinating multi-layered validation checks."""
    def __init__(self) -> None:
        self.schema_validator = SchemaValidator()
        self.missing_validator = MissingValueValidator()
        self.stats_validator = StatisticalValidator()

    def run_validation(
        self,
        df: pd.DataFrame,
        expected_types: dict[str, str],
        range_bounds: dict[str, tuple[float, float]] = None,
        missing_thresholds: dict[str, float] = None,
    ) -> dict[str, Any]:
        logger.info("Executing comprehensive feature validation checks...")
        
        schema_report = self.schema_validator.validate(df, expected_types)
        missing_report = self.missing_validator.validate(df, missing_thresholds)
        stats_report = self.stats_validator.validate(df, range_bounds)
        
        overall_status = "PASS"
        if schema_report["status"] == "FAIL":
            overall_status = "FAIL"
        elif "WARN" in [missing_report["status"], stats_report["status"]]:
            overall_status = "WARN"
            
        report = {
            "overall_status": overall_status,
            "timestamp": pd.Timestamp.now().isoformat(),
            "schema_validation": schema_report,
            "missingness_validation": missing_report,
            "statistical_validation": stats_report,
        }
        
        logger.info("Feature Validation Pipeline complete. Overall Status: %s", overall_status)
        return report

    def save_report(self, report: dict[str, Any], dest_path: Path) -> Path:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info("Validation report saved successfully to %s", dest_path)
        return dest_path
