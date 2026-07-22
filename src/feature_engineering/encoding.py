"""Feature encoding classes for Label, Frequency, Count, One-Hot, and Target encoding strategies."""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EncodingStrategySelector:
    """Routs categorical features to recommended encoding strategy based on cardinality and characteristics."""
    def __init__(self, target_cols: list[str] | None = None) -> None:
        self.target_cols = target_cols or ["isFraud"]

    def select_strategy(self, df: pd.DataFrame, cat_cols: list[str]) -> dict[str, str]:
        strategies = {}
        for col in cat_cols:
            if col in self.target_cols:
                continue
            
            # Count unique non-null categories
            num_unique = df[col].nunique(dropna=True)
            
            # Binary variables or small ordered categoricals
            if num_unique <= 2:
                strategies[col] = "Label"
            elif num_unique <= 10:
                strategies[col] = "OneHot"
            elif num_unique < 50:
                strategies[col] = "Frequency"
            else:
                # High cardinality features
                strategies[col] = "Target"

        logger.info("Automatically determined encoding strategies: %s", strategies)
        return strategies


class VectorizedLabelEncoder:
    """Label encoding mapped via vectorized pandas dictionary replacement."""
    def __init__(self) -> None:
        self.mapping_: dict[str, int] = {}
        self.default_val: int = -1

    def fit(self, s: pd.Series) -> VectorizedLabelEncoder:
        cats = sorted(s.dropna().unique().tolist())
        self.mapping_ = {str(cat): idx for idx, cat in enumerate(cats)}
        return self

    def transform(self, s: pd.Series) -> pd.Series:
        s_str = s.astype(str)
        # Vectorized mapping replacement
        out = s_str.map(self.mapping_).fillna(self.default_val).astype(int)
        return out


class VectorizedFrequencyEncoder:
    """Frequency encoding replacement using value proportions."""
    def __init__(self) -> None:
        self.mapping_: dict[str, float] = {}
        self.default_val: float = 0.0

    def fit(self, s: pd.Series) -> VectorizedFrequencyEncoder:
        counts = s.astype(str).value_counts(normalize=True).to_dict()
        self.mapping_ = counts
        return self

    def transform(self, s: pd.Series) -> pd.Series:
        s_str = s.astype(str)
        # Vectorized mapping
        out = s_str.map(self.mapping_).fillna(self.default_val).astype(float)
        return out


class VectorizedCountEncoder:
    """Count encoding replacement tracking counts, log-counts, frequencies, and percentiles."""
    def __init__(self) -> None:
        self.counts_: dict[str, int] = {}
        self.percentiles_: dict[str, float] = {}
        self.total_count: int = 0

    def fit(self, s: pd.Series) -> VectorizedCountEncoder:
        counts = s.astype(str).value_counts().to_dict()
        self.counts_ = counts
        self.total_count = len(s)
        
        # Calculate rank percentile mapping
        ranks = pd.Series(counts).rank(pct=True).to_dict()
        self.percentiles_ = ranks
        return self

    def transform(self, s: pd.Series, mode: str = "count") -> pd.Series:
        s_str = s.astype(str)
        if mode == "count":
            return s_str.map(self.counts_).fillna(0).astype(int)
        elif mode == "log_count":
            counts = s_str.map(self.counts_).fillna(0).astype(float)
            return np.log1p(counts)
        elif mode == "percentile":
            return s_str.map(self.percentiles_).fillna(0.0).astype(float)
        else:
            raise ValueError(f"Unknown count encoding mode: {mode}")


class LeakageSafeTargetEncoder:
    """Out-of-fold K-Fold target encoder with smoothing to prevent target leakage."""
    def __init__(
        self,
        min_samples_leaf: float = 20.0,
        smoothing: float = 10.0,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self.n_folds = n_folds
        self.random_state = random_state
        self.global_prior_: float = 0.0
        self.encoding_map_: dict[str, float] = {}

    def fit(self, s: pd.Series, y: pd.Series) -> LeakageSafeTargetEncoder:
        # Check target prior
        self.global_prior_ = float(y.mean())
        
        # Build global smoothed target encoding dictionary
        s_str = s.astype(str)
        stats = pd.DataFrame({"category": s_str, "target": y.astype(float)})
        grouped = stats.groupby("category")["target"].agg(["count", "mean"])
        
        counts = grouped["count"]
        means = grouped["mean"]
        
        # Smoothed weights: lambda = 1 / (1 + e^-( (count - min_samples_leaf) / smoothing ))
        weights = 1.0 / (1.0 + np.exp(-(counts - self.min_samples_leaf) / self.smoothing))
        smoothed = weights * means + (1.0 - weights) * self.global_prior_
        self.encoding_map_ = smoothed.to_dict()
        return self

    def fit_transform(self, s: pd.Series, y: pd.Series) -> pd.Series:
        self.global_prior_ = float(y.mean())
        
        # Perform K-Fold out-of-fold statistics
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        encoded_s = pd.Series(index=s.index, dtype=float)
        s_str = s.astype(str)

        # Run out-of-fold computation loops
        for train_idx, val_idx in kf.split(s):
            s_train = s_str.iloc[train_idx]
            y_train = y.iloc[train_idx].astype(float)
            
            s_val = s_str.iloc[val_idx]
            
            # Prior on train split folds
            fold_prior = float(y_train.mean())
            
            # Smoothed statistics
            grouped = pd.DataFrame({"cat": s_train, "tgt": y_train}).groupby("cat")["tgt"].agg(["count", "mean"])
            weights = 1.0 / (1.0 + np.exp(-(grouped["count"] - self.min_samples_leaf) / self.smoothing))
            smoothed = weights * grouped["mean"] + (1.0 - weights) * fold_prior
            
            # Impute validations
            encoded_s.iloc[val_idx] = s_val.map(smoothed).fillna(fold_prior)

        # Fit final global mapping on complete training partition
        self.fit(s, y)
        return encoded_s

    def transform(self, s: pd.Series) -> pd.Series:
        s_str = s.astype(str)
        out = s_str.map(self.encoding_map_).fillna(self.global_prior_).astype(float)
        return out


class VectorizedOneHotEncoder:
    """One-hot encoder containing unique columns dictionary mappings."""
    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold
        self.categories_: list[str] = []

    def fit(self, s: pd.Series) -> VectorizedOneHotEncoder:
        # Include categories exceeding threshold popularity, sorting them
        counts = s.astype(str).value_counts(normalize=True)
        self.categories_ = sorted([str(cat) for cat in counts[counts >= self.threshold].index])
        return self

    def transform(self, s: pd.Series) -> pd.DataFrame:
        out = pd.DataFrame(index=s.index)
        s_str = s.astype(str)
        for cat in self.categories_:
            out[f"{s.name}_{cat}"] = (s_str == cat).astype(int)
        
        # Collect unknown category column
        unknowns = ~s_str.isin(self.categories_)
        out[f"{s.name}_unknown"] = unknowns.astype(int)
        return out


class EncodingValidationGate:
    """Validates encoded outputs checks for NaNs, infinites, duplicates, and constants."""
    def validate(self, df_encoded: pd.DataFrame) -> dict[str, Any]:
        logger.info("Executing encoding validation checks...")
        
        nan_cols = [col for col in df_encoded.columns if df_encoded[col].isnull().any()]
        inf_cols = [col for col in df_encoded.columns if np.isinf(df_encoded[col]).any()]
        
        # Check standard deviations for const features
        const_cols = [col for col in df_encoded.columns if df_encoded[col].nunique() <= 1]
        
        # Check duplicate headers names
        dup_cols = df_encoded.columns[df_encoded.columns.duplicated()].tolist()

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
        
        logger.info("Validation Gate finished. Status: %s", report["status"])
        return report


class EncoderRegistry:
    """Manages metadata serialization and serializes production encoder bundles."""
    def __init__(self) -> None:
        self.metadata: list[dict[str, Any]] = []
        self.encoders: dict[str, Any] = {}

    def register(self, feature_name: str, encoder_type: str, encoder_instance: Any, cardinality: int) -> None:
        self.encoders[feature_name] = encoder_instance
        self.metadata.append({
            "feature_name": feature_name,
            "encoder_type": encoder_type,
            "cardinality": cardinality,
            "registered_at": pd.Timestamp.now().isoformat(),
        })

    def save_bundle(self, dest_dir: Path) -> tuple[Path, Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        bundle_path = dest_dir / "production_encoder_bundle.pkl"
        with open(bundle_path, "wb") as f:
            pickle.dump(self.encoders, f)
            
        manifest_path = dest_dir / "encoding_pipeline_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "registry": self.metadata,
                "version": "v1.0",
                "owner": "ML-Engineering-Team",
            }, f, indent=4)

        logger.info("Saved production encoder bundle to %s and manifest to %s", bundle_path, manifest_path)
        return bundle_path, manifest_path
