"""Missing feature classes with robust train/test column name mapping and alignment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_actual_column(df: pd.DataFrame, col: str) -> str | None:
    """Helper to locate a column in the DataFrame, resolving train/test differences like underscores or dashes (e.g. id_01 vs id-01)."""
    if col in df.columns:
        return col
    dash_col = col.replace("_", "-")
    if dash_col in df.columns:
        return dash_col
    under_col = col.replace("-", "_")
    if under_col in df.columns:
        return under_col
    return None


class VectorizedMissingEngine:
    """Computes row-level and column-specific missingness metrics (counts, ratios, completeness, binary indicators)."""
    def compute_indicators(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Returns binary indicators (1.0 if missing, 0.0 if present) for specific columns, using robust column mapping."""
        df_indicators = pd.DataFrame(index=df.index)
        for col in cols:
            actual = _get_actual_column(df, col)
            if actual is not None:
                # Keep output name standardized to the requested col parameter
                df_indicators[f"{col}_isna"] = df[actual].isnull().astype(float)
            else:
                # If not present in either shape, fill with 1.0 (indicating completely missing)
                df_indicators[f"{col}_isna"] = 1.0
        return df_indicators

    def compute_row_stats(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Returns row-level missing count, missing ratio, and feature completeness score, resolving any column mismatches."""
        mapped_cols = []
        for col in cols:
            actual = _get_actual_column(df, col)
            if actual is not None:
                mapped_cols.append(actual)
                
        if not mapped_cols:
            res = pd.DataFrame(index=df.index)
            res["missing_count"] = 0.0
            res["missing_ratio"] = 0.0
            res["completeness_score"] = 1.0
            return res
            
        df_sub = df[mapped_cols]
        missing_matrix = df_sub.isnull()
        
        counts = missing_matrix.sum(axis=1).astype(float)
        # Note: we divide by original requested columns count to maintain feature space alignment!
        ratios = counts / len(cols)
        completeness = 1.0 - ratios
        
        res = pd.DataFrame(index=df.index)
        res["missing_count"] = counts
        res["missing_ratio"] = ratios
        res["completeness_score"] = completeness
        return res


class MissingPatternBuilder:
    """Builds numerical unique missingness patterns and hash codes across groups of features."""
    def build_pattern_hashes(self, df: pd.DataFrame, cols: list[str]) -> pd.Series:
        """Vectorized computation of binary indicator integer hash representation, mapping missing codes to unique integers."""
        # Convert each binary indicator to a power of 2 digit place
        # Support underscore vs dash mappings
        hash_series = pd.Series(0.0, index=df.index)
        for i, col in enumerate(cols):
            actual = _get_actual_column(df, col)
            if actual is not None:
                hash_series += df[actual].isnull().astype(int) * (2 ** i)
            else:
                # Treat as completely missing if the column doesn't exist
                hash_series += 1 * (2 ** i)
            
        return hash_series.astype(float).rename("missing_pattern_hash")


class AutomaticMissingDiscoveryEngine:
    """Discovers feature columns with non-trivial missing percentages that carry positive signal."""
    def __init__(self, min_missing_ratio: float = 0.05, max_missing_ratio: float = 0.95) -> None:
        self.min_missing_ratio = min_missing_ratio
        self.max_missing_ratio = max_missing_ratio

    def discover_missing_columns(self, df: pd.DataFrame) -> list[str]:
        """Scans the dataframe and lists numerical or categorical columns that meet missing percentage boundaries."""
        discovered = []
        for col in df.columns:
            if col == "TransactionID" or col == "isFraud":
                continue
            ratio = df[col].isnull().mean()
            if self.min_missing_ratio <= ratio <= self.max_missing_ratio:
                discovered.append(col)
        return discovered


class MissingValidationGate:
    """Validates missing feature indicators and metrics to ensure zero Inf propagation or constant values."""
    def validate(self, df_missing: pd.DataFrame) -> dict[str, Any]:
        logger.info("Executing missing validation checks...")
        
        nan_cols = [col for col in df_missing.columns if df_missing[col].isnull().any()]
        inf_cols = [col for col in df_missing.columns if np.isinf(df_missing[col]).any()]
        const_cols = [col for col in df_missing.columns if df_missing[col].nunique() <= 1]
        dup_cols = df_missing.columns[df_missing.columns.duplicated()].tolist()

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


class MissingRegistry:
    """Manages lineage and cataloging for engineered missing indicator features."""
    def __init__(self) -> None:
        self.metadata: list[dict[str, Any]] = []

    def register(self, feature_name: str, base_column: str, feature_type: str) -> None:
        self.metadata.append({
            "feature_name": feature_name,
            "base_column": base_column,
            "feature_type": feature_type,
            "created_at": pd.Timestamp.now().isoformat(),
        })

    def save_catalog(self, dest_dir: Path) -> tuple[Path, Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = dest_dir / "missing_pipeline_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "registry": self.metadata,
                "version": "v1.0",
                "owner": "ML-Engineering-Team",
            }, f, indent=4)
            
        csv_path = dest_dir / "missing_catalog.csv"
        pd.DataFrame(self.metadata).to_csv(csv_path, index=False)
        
        logger.info("Saved missing manifest to %s and catalog to %s", manifest_path, csv_path)
        return manifest_path, csv_path
