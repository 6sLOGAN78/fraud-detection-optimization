"""Feature aggregation classes for Mean, Median, Std, Min, Max, Count, and Rolling group calculations."""

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


class AggregationGroupBuilder:
    """Configures entity aggregation keys and target columns to aggregate."""
    def __init__(self, config: dict[str, list[str]] | None = None) -> None:
        self.config = config or {
            "card_keys": ["card1", "card2"],
            "device_keys": ["DeviceType", "DeviceInfo"],
            "addr_keys": ["addr1", "addr2"],
            "product_keys": ["ProductCD"],
            "email_keys": ["P_emaildomain"],
        }
        
    def get_group_keys(self) -> list[str]:
        keys = []
        for g_list in self.config.values():
            keys.extend(g_list)
        return sorted(list(set(keys)))


class VectorizedAggregationEngine:
    """Computes summary statistics (mean, median, std, min, max, count) grouped by entities on train, mapped to test to avoid leakage."""
    def __init__(self, group_col: str, agg_col: str) -> None:
        self.group_col = group_col
        self.agg_col = agg_col
        self.stats_: dict[str, dict[str, float]] = {}

    def fit(self, df_train: pd.DataFrame) -> VectorizedAggregationEngine:
        # Avoid computing on NaNs in group column
        df_clean = df_train.dropna(subset=[self.group_col])
        if df_clean.empty:
            logger.warning("Blank dataframe for group column %s", self.group_col)
            return self
            
        grp = df_clean.groupby(self.group_col)[self.agg_col]
        
        # Calculate stats vectorized
        means = grp.mean().to_dict()
        medians = grp.median().to_dict()
        stds = grp.std().fillna(0.0).to_dict()
        mins = grp.min().to_dict()
        maxs = grp.max().to_dict()
        counts = grp.count().to_dict()

        for key in means.keys():
            self.stats_[str(key)] = {
                "mean": float(means[key]),
                "median": float(medians[key]),
                "std": float(stds[key]),
                "min": float(mins[key]),
                "max": float(maxs[key]),
                "count": int(counts[key]),
            }
            
        return self

    def transform(self, df: pd.DataFrame, stat_type: str) -> pd.Series:
        s_grp = df[self.group_col].astype(str)
        # Vectorized map of calculated stat
        mapping = {k: v[stat_type] for k, v in self.stats_.items()}
        out = s_grp.map(mapping)
        
        # Fill missing categories with logical defaults
        if stat_type == "count":
            out = out.fillna(0).astype(int)
        elif stat_type == "std":
            out = out.fillna(0.0).astype(float)
        else:
            # Fill with global stats of train dataset
            global_val = 0.0
            if self.stats_:
                vals = [v[stat_type] for v in self.stats_.values()]
                global_val = float(np.nanmean(vals)) if vals else 0.0
            out = out.fillna(global_val).astype(float)
            
        return out


class RollingAggregationEngine:
    """Calculates time-based expanding and rolling summary statistics chronologically to prevent temporal leakage."""
    def __init__(self, group_col: str, agg_col: str, window_size: int = 5) -> None:
        self.group_col = group_col
        self.agg_col = agg_col
        self.window_size = window_size

    def compute_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sorts chronologically and calculates expanding/rolling mean and counts inside group."""
        # Assume df is chronologically indexed or sorted by TransactionID
        df_sorted = df.copy()
        
        # Sort values to guarantee temporal ordering
        if "TransactionID" in df_sorted.columns:
            df_sorted = df_sorted.sort_values("TransactionID")
        
        # Calculate expanding statistics (cumulative values up to current transaction)
        # Shifted by 1 to exclude current transaction's value and prevent self-lookahead target leakage
        grp = df_sorted.groupby(self.group_col)[self.agg_col]
        
        expanding_sum = grp.apply(lambda x: x.shift(1).expanding().sum()).reset_index(level=0, drop=True)
        expanding_cnt = grp.apply(lambda x: x.shift(1).expanding().count()).reset_index(level=0, drop=True)
        expanding_mean = expanding_sum / expanding_cnt.replace(0, np.nan)
        
        # Calculate rolling statistics (last N transactions)
        rolling_sum = grp.apply(lambda x: x.shift(1).rolling(self.window_size, min_periods=1).sum()).reset_index(level=0, drop=True)
        rolling_cnt = grp.apply(lambda x: x.shift(1).rolling(self.window_size, min_periods=1).count()).reset_index(level=0, drop=True)
        rolling_mean = rolling_sum / rolling_cnt.replace(0, np.nan)
        
        out = pd.DataFrame(index=df_sorted.index)
        out[f"{self.group_col}_{self.agg_col}_exp_mean"] = expanding_mean.fillna(0.0)
        out[f"{self.group_col}_{self.agg_col}_exp_cnt"] = expanding_cnt.fillna(0).astype(int)
        out[f"{self.group_col}_{self.agg_col}_roll_mean"] = rolling_mean.fillna(0.0)
        
        return out


class AggregationValidationGate:
    """Validates aggregated features checking for NaN propagation or infinite values."""
    def validate(self, df_agg: pd.DataFrame) -> dict[str, Any]:
        logger.info("Executing aggregation validation checks...")
        
        nan_cols = [col for col in df_agg.columns if df_agg[col].isnull().any()]
        inf_cols = [col for col in df_agg.columns if np.isinf(df_agg[col]).any()]
        
        const_cols = [col for col in df_agg.columns if df_agg[col].nunique() <= 1]
        dup_cols = df_agg.columns[df_agg.columns.duplicated()].tolist()

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


class AggregationRegistry:
    """Tracks aggregation metadata catalog and writes registry definitions."""
    def __init__(self) -> None:
        self.metadata: list[dict[str, Any]] = []

    def register(self, feature_name: str, agg_type: str, src_col: str, group_key: str, window: str = "global") -> None:
        self.metadata.append({
            "feature_name": feature_name,
            "aggregation_type": agg_type,
            "source_column": src_col,
            "group_key": group_key,
            "window": window,
            "created_at": pd.Timestamp.now().isoformat(),
        })

    def save_catalog(self, dest_dir: Path) -> tuple[Path, Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = dest_dir / "aggregation_pipeline_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "registry": self.metadata,
                "version": "v1.0",
                "owner": "ML-Engineering-Team",
            }, f, indent=4)
            
        csv_path = dest_dir / "aggregation_catalog.csv"
        pd.DataFrame(self.metadata).to_csv(csv_path, index=False)
        
        logger.info("Saved aggregation manifest to %s and catalog to %s", manifest_path, csv_path)
        return manifest_path, csv_path
