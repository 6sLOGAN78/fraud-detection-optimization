"""Feature validation engines including schema alignment, datatype verification, missingness tests, statistical range gates, outlier checks, PSI drift detection, leakages, correlation redundancy, and decision-tree surrogates."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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
            missing_pct = df[col].isnull().mean()
            limit = thresholds.get(col, default_max_missing)
            if missing_pct > limit:
                high_missingness[col] = {"missing_ratio": missing_pct, "limit": limit}
                
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
                    
                if series_clean.nunique() <= 1:
                    constant_cols.append(col)
                    continue
                
                if col in range_bounds:
                    min_val, max_val = range_bounds[col]
                    val_min = series_clean.min()
                    val_max = series_clean.max()
                    if val_min < min_val or val_max > max_val:
                        out_of_bounds[col] = {"min": float(val_min), "max": float(val_max), "bounds": [min_val, max_val]}
                
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


class DriftDetector:
    """Computes Population Stability Index (PSI) to identify distribution shifts."""
    def compute_psi(self, baseline: pd.Series, target: pd.Series, bins: int = 10) -> float:
        """Vectorized execution of PSI between baseline (train) and target (test) series."""
        base_clean = baseline.dropna()
        tgt_clean = target.dropna()
        
        if base_clean.empty or tgt_clean.empty:
            return 0.0
            
        # Determine quantile-based boundaries on baseline to ensure balanced bin shapes
        try:
            percentiles = np.linspace(0, 100, bins + 1)
            cuts = np.percentile(base_clean, percentiles)
            cuts = np.unique(cuts)  # deduplicate bounds
            if len(cuts) < 2:
                return 0.0
        except Exception:
            return 0.0
            
        baseline_counts, _ = np.histogram(base_clean, bins=cuts)
        target_counts, _ = np.histogram(tgt_clean, bins=cuts)
        
        # Calculate percentages
        P = baseline_counts / len(base_clean)
        Q = target_counts / len(tgt_clean)
        
        # Use epsilon to prevent division by zero or log(0)
        eps = 1e-4
        P = np.where(P == 0.0, eps, P)
        Q = np.where(Q == 0.0, eps, Q)
        
        # Re-normalize to sum to 1.0
        P /= P.sum()
        Q /= Q.sum()
        
        # PSI sum( (P - Q) * ln(P/Q) )
        # Note: here standard baseline is Q, target is P, or vice versa
        # Traditional PSI: sum((Actual - Expected) * ln(Actual / Expected))
        # Let Target (Actual) be P, Baseline (Expected) be Q
        psi_val = np.sum((P - Q) * np.log(P / Q))
        return float(psi_val)

    def validate_drift(self, df_train: pd.DataFrame, df_test: pd.DataFrame, threshold: float = 0.25) -> dict[str, Any]:
        lst_drifted = {}
        for col in df_train.columns:
            if col == "TransactionID" or col == "isFraud":
                continue
            if "float" in str(df_train[col].dtype) or "int" in str(df_train[col].dtype):
                psi = self.compute_psi(df_train[col], df_test[col])
                if psi > threshold:
                    lst_drifted[col] = {"psi": psi, "threshold": threshold}
                    
        return {
            "drifted_columns_count": len(lst_drifted),
            "drifted_columns": lst_drifted,
            "status": "PASS" if not lst_drifted else "WARN",
        }


class LeakageDetector:
    """Validates features for target leakage based on target correlation thresholds."""
    def validate_leakage(self, df: pd.DataFrame, target: pd.Series, threshold: float = 0.90) -> dict[str, Any]:
        suspicious_leakages = {}
        for col in df.columns:
            if col == "TransactionID":
                continue
            if "float" in str(df[col].dtype) or "int" in str(df[col].dtype):
                corr = df[col].corr(target)
                if abs(corr) > threshold:
                    suspicious_leakages[col] = {"correlation": float(corr), "threshold": threshold}
                    
        return {
            "leakage_columns_count": len(suspicious_leakages),
            "leakage_columns": suspicious_leakages,
            "status": "PASS" if not suspicious_leakages else "WARN",
        }


class CorrelationValidator:
    """Screens features for Pearson & Spearman redundancies."""
    def validate_redundancy(self, df: pd.DataFrame, threshold: float = 0.85) -> dict[str, Any]:
        num_cols = [col for col in df.columns if ("float" in str(df[col].dtype) or "int" in str(df[col].dtype)) and col != "TransactionID"]
        if len(num_cols) < 2:
            return {"redundancies_count": 0, "pairs": {}, "status": "PASS"}
            
        df_num = df[num_cols].dropna()
        if df_num.empty or len(df_num) < 5:
            return {"redundancies_count": 0, "pairs": {}, "status": "PASS"}
            
        # Downsample rows if too large to prevent Memory/OOM limit issues on the host
        if len(df_num) > 20000:
            df_num = df_num.sample(n=20000, random_state=42)

        # Pearson is very fast
        corr_matrix_p = df_num.corr(method="pearson").abs()
        
        # Optimize Spearman using rank-transform first (order of magnitude faster than pandas native spearman loop)
        try:
            df_ranked = df_num.rank()
            corr_matrix_s = df_ranked.corr(method="pearson").abs()
        except Exception as e:
            logger.warning("Spearman optimization failed, using Pearson correlation as fallback: %s", e)
            corr_matrix_s = corr_matrix_p
        
        redundancy_pairs = {}
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col1 = num_cols[i]
                col2 = num_cols[j]
                
                val_p = corr_matrix_p.iloc[i, j]
                val_s = corr_matrix_s.iloc[i, j]
                
                if val_p > threshold or val_s > threshold:
                    redundancy_pairs[f"{col1}__vs__{col2}"] = {
                        "pearson": float(val_p) if not np.isnan(val_p) else 0.0,
                        "spearman": float(val_s) if not np.isnan(val_s) else 0.0,
                        "threshold": threshold,
                    }
                    
        return {
            "redundancies_count": len(redundancy_pairs),
            "pairs": redundancy_pairs,
            "status": "PASS" if not redundancy_pairs else "WARN",
        }


class ImportanceValidator:
    """Trains surrogate decision tree model to assert feature importance bounds."""
    def validate_importance(self, df: pd.DataFrame, target: pd.Series, threshold: float = 0.01) -> dict[str, Any]:
        num_cols = [col for col in df.columns if ("float" in str(df[col].dtype) or "int" in str(df[col].dtype)) and col != "TransactionID"]
        if not num_cols:
            return {"zero_importance_count": 0, "importances": {}, "status": "PASS"}
            
        # Downsample for lightning-fast training of surrogate on massive datasets
        if len(df) > 50000:
            df_sample = df.sample(n=50000, random_state=42)
            target_sample = target.loc[df_sample.index]
        else:
            df_sample = df
            target_sample = target

        X = df_sample[num_cols].fillna(0.0)
        y = target_sample.fillna(0.0)
        
        if len(y.unique()) <= 2:
            model = DecisionTreeClassifier(max_depth=5, random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=5, random_state=42)
            
        try:
            model.fit(X, y)
            importances = dict(zip(num_cols, model.feature_importances_))
        except Exception as e:
            logger.warning("Surrogate model execution failed: %s", e)
            importances = {col: 0.0 for col in num_cols}
            
        zero_imp = {col: float(val) for col, val in importances.items() if val < threshold}
        
        return {
            "zero_importance_count": len(zero_imp),
            "importances": {col: float(val) for col, val in importances.items()},
            "zero_importance_features": zero_imp,
            "status": "PASS",
        }



class FeatureValidationPipeline:
    """Unified validation pipeline coordinating multi-layered validation checks."""
    def __init__(self) -> None:
        self.schema_validator = SchemaValidator()
        self.missing_validator = MissingValueValidator()
        self.stats_validator = StatisticalValidator()
        self.drift_detector = DriftDetector()
        self.leakage_detector = LeakageDetector()
        self.corr_validator = CorrelationValidator()
        self.importance_validator = ImportanceValidator()

    def run_validation(
        self,
        df: pd.DataFrame,
        expected_types: dict[str, str],
        range_bounds: dict[str, tuple[float, float]] = None,
        missing_thresholds: dict[str, float] = None,
        # Advanced optional checks
        df_ref: pd.DataFrame = None,
        target: pd.Series = None,
        drift_threshold: float = 0.25,
        leakage_threshold: float = 0.90,
        redundancy_threshold: float = 0.85,
        importance_threshold: float = 0.01,
    ) -> dict[str, Any]:
        logger.info("Executing comprehensive feature validation checks...")
        
        schema_report = self.schema_validator.validate(df, expected_types)
        missing_report = self.missing_validator.validate(df, missing_thresholds)
        stats_report = self.stats_validator.validate(df, range_bounds)
        
        # Drift checks
        if df_ref is not None:
            drift_report = self.drift_detector.validate_drift(df_ref, df, threshold=drift_threshold)
        else:
            drift_report = {"status": "PASS", "info": "Reference set not provided"}
            
        # Leakage checks
        if target is not None:
            leakage_report = self.leakage_detector.validate_leakage(df, target, threshold=leakage_threshold)
            importance_report = self.importance_validator.validate_importance(df, target, threshold=importance_threshold)
        else:
            leakage_report = {"status": "PASS", "info": "Target column not provided"}
            importance_report = {"status": "PASS", "info": "Target column not provided"}
            
        # Redundancy checks
        corr_report = self.corr_validator.validate_redundancy(df, threshold=redundancy_threshold)
        
        overall_status = "PASS"
        if schema_report["status"] == "FAIL":
            overall_status = "FAIL"
        elif "WARN" in [
            missing_report["status"],
            stats_report["status"],
            drift_report["status"],
            leakage_report["status"],
            corr_report["status"],
            importance_report["status"],
        ]:
            overall_status = "WARN"
            
        report = {
            "overall_status": overall_status,
            "timestamp": pd.Timestamp.now().isoformat(),
            "schema_validation": schema_report,
            "missingness_validation": missing_report,
            "statistical_validation": stats_report,
            "drift_validation": drift_report,
            "leakage_validation": leakage_report,
            "correlation_validation": corr_report,
            "importance_validation": importance_report,
        }
        
        logger.info("Feature Validation Pipeline complete. Overall Status: %s", overall_status)
        return report

    def save_report(self, report: dict[str, Any], dest_path: Path) -> Path:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info("Validation report saved successfully to %s", dest_path)
        return dest_path
