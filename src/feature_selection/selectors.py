"""Concrete implementations of statistical and algorithmic feature selectors: Null, Variance, Correlation, and LightGBM Importance."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import numpy as np

from src.feature_selection.base import BaseFeatureSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NullSelector(BaseFeatureSelector):
    """Filters variables having missing/null ratios higher than the specified threshold."""
    def __init__(
        self,
        threshold: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
        log_level: str = "INFO"
    ) -> None:
        super().__init__("NullSelector")
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> NullSelector:
        logger.info("Executing %s fit verification gate...", self.name)
        null_ratios = X.isnull().mean()
        
        self.selected_features_ = list(null_ratios[null_ratios <= self.threshold].index)
        self.dropped_features_ = list(null_ratios[null_ratios > self.threshold].index)
        
        logger.info("%s: Retained %d, Dropped %d features.", self.name, len(self.selected_features_), len(self.dropped_features_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Keep entity ID if present as an index/column and not analyzed
        cols = [c for c in X.columns if c in self.selected_features_]
        return X[cols]


class VarianceSelector(BaseFeatureSelector):
    """Filters low/zero variance continuous features below or equal to threshold."""
    def __init__(
        self,
        threshold: float = 0.0,
        random_state: int = 42,
        n_jobs: int = -1,
        log_level: str = "INFO"
    ) -> None:
        super().__init__("VarianceSelector")
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> VarianceSelector:
        logger.info("Executing %s fit verification gate...", self.name)
        
        # Only analyze numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        variances = X[numeric_cols].var(ddof=0)
        
        selected_numeric = list(variances[variances > self.threshold].index)
        dropped_numeric = list(variances[variances <= self.threshold].index)
        
        # Non-numeric columns are retained by default
        non_numeric = list(X.select_dtypes(exclude=[np.number]).columns)
        
        self.selected_features_ = selected_numeric + non_numeric
        self.dropped_features_ = dropped_numeric
        
        logger.info("%s: Retained %d, Dropped %d features.", self.name, len(self.selected_features_), len(self.dropped_features_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in X.columns if c in self.selected_features_]
        return X[cols]


class CorrelationSelector(BaseFeatureSelector):
    """Drops collinear feature pairs above correlation threshold, retaining higher variance features."""
    def __init__(
        self,
        threshold: float = 0.95,
        random_state: int = 42,
        n_jobs: int = -1,
        log_level: str = "INFO"
    ) -> None:
        super().__init__("CorrelationSelector")
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CorrelationSelector:
        logger.info("Executing %s fit verification gate...", self.name)
        
        numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        if len(numeric_cols) < 2:
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Calculate absolute correlation matrix
        corr_matrix = X[numeric_cols].corr().abs()
        
        # Find upper triangle to avoid double checking pairs
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = set()
        # Evaluate variance to keep the feature with more information
        variances = X[numeric_cols].var(ddof=0)
        
        for col in upper.columns:
            # Find collinear features
            collinear = upper.index[upper[col] > self.threshold].tolist()
            if collinear:
                for c in collinear:
                    # Keep the one with higher variance, drop the other
                    if variances[c] >= variances[col]:
                        to_drop.add(col)
                    else:
                        to_drop.add(c)
                        
        self.dropped_features_ = list(to_drop)
        self.selected_features_ = [c for c in X.columns if c not in self.dropped_features_]
        
        logger.info("%s: Retained %d, Dropped %d collinear features.", self.name, len(self.selected_features_), len(self.dropped_features_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in X.columns if c in self.selected_features_]
        return X[cols]


class ImportanceSelector(BaseFeatureSelector):
    """Fits a lightweight RandomForest baseline model on training set and filters low importance variables."""
    def __init__(self, threshold: float = 0.05, random_state: int = 42, n_jobs: int = -1) -> None:
        super().__init__("ImportanceSelector")
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.importances_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ImportanceSelector:
        logger.info("Executing %s fit verification gate...", self.name)
        if y is None:
            logger.warning("%s requires target labels y for importance scoring. Retaining all.", self.name)
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self
            
        # Select numeric columns for modeling
        numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        if not numeric_cols:
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Downsample to avoid memory issues (OOM) on large datasets
        max_samples = 50000
        if len(X) > max_samples:
            logger.info("Downsampling %s training data to %d rows to avoid OOM...", self.name, max_samples)
            # Use random sampling to downsample
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(X.index, size=max_samples, replace=False)
            X_subset = X.loc[indices, numeric_cols]
            y_subset = y.loc[indices]
        else:
            X_subset = X[numeric_cols]
            y_subset = y

        # Use scikit-learn RandomForestClassifier to get relative importance
        from sklearn.ensemble import RandomForestClassifier
        
        # Fill missing values for baseline classifier stability
        X_imputed = X_subset.fillna(0.0)
        
        model = RandomForestClassifier(
            n_estimators=30,
            max_depth=6,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        model.fit(X_imputed, y_subset)
        
        importances = model.feature_importances_
        # Normalize relative importances relative to the maximum feature importance value
        max_imp = float(np.max(importances)) if len(importances) > 0 else 1.0
        if max_imp == 0.0:
            max_imp = 1.0
            
        normalized_importances = importances / max_imp
        
        feature_scores = dict(zip(numeric_cols, normalized_importances))
        self.importances_ = feature_scores

        selected_numeric = []
        dropped_numeric = []
        for col, val in feature_scores.items():
            if val >= self.threshold:
                selected_numeric.append(col)
            else:
                dropped_numeric.append(col)
                
        # Non-numeric parameters are retained by default
        non_numeric = list(X.select_dtypes(exclude=[np.number]).columns)
        
        self.selected_features_ = selected_numeric + non_numeric
        self.dropped_features_ = dropped_numeric
        
        logger.info("%s: Retained %d, Dropped %d low importance features.", self.name, len(self.selected_features_), len(self.dropped_features_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in X.columns if c in self.selected_features_]
        return X[cols]


class MutualInformationSelector(BaseFeatureSelector):
    """Filters low Mutual Information features below or equal to threshold using KSG estimation."""
    def __init__(
        self,
        threshold: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
        log_level: str = "INFO"
    ) -> None:
        super().__init__("MutualInformationSelector")
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level
        self.mi_scores_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> MutualInformationSelector:
        logger.info("Executing %s fit verification gate...", self.name)
        if y is None:
            logger.warning("%s requires target labels y for MI scoring. Retaining all.", self.name)
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Select numeric columns for modeling
        numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        if not numeric_cols:
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Downsample to avoid extreme execution time (KNN is O(N^2))
        max_samples = 20000
        if len(X) > max_samples:
            logger.info("Downsampling %s training data to %d rows to avoid OOM and CPU stalls...", self.name, max_samples)
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(X.index, size=max_samples, replace=False)
            X_subset = X.loc[indices, numeric_cols]
            y_subset = y.loc[indices]
        else:
            X_subset = X[numeric_cols]
            y_subset = y

        from sklearn.feature_selection import mutual_info_classif
        
        # Fill missing values for stability
        X_imputed = X_subset.fillna(0.0)
        
        # Compute discrete mutual information scores
        scores = mutual_info_classif(
            X_imputed,
            y_subset,
            random_state=self.random_state
        )
        
        # Max-scale the scores to normalize between 0 and 1
        max_score = float(np.max(scores)) if len(scores) > 0 else 1.0
        if max_score == 0.0:
            max_score = 1.0
            
        normalized_scores = scores / max_score
        
        feature_scores = dict(zip(numeric_cols, normalized_scores))
        self.mi_scores_ = feature_scores

        selected_numeric = []
        dropped_numeric = []
        for col, val in feature_scores.items():
            if val >= self.threshold:
                selected_numeric.append(col)
            else:
                dropped_numeric.append(col)

        non_numeric = list(X.select_dtypes(exclude=[np.number]).columns)
        
        self.selected_features_ = selected_numeric + non_numeric
        self.dropped_features_ = dropped_numeric
        
        logger.info("%s: Retained %d, Dropped %d low MI features.", self.name, len(self.selected_features_), len(self.dropped_features_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in X.columns if c in self.selected_features_]
        return X[cols]

