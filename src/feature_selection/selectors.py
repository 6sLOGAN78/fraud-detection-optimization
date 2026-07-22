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


class SHAPSelector(BaseFeatureSelector):
    """Filters features based on mean absolute SHAP values calculated from a baseline model."""
    def __init__(
        self,
        threshold: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
        log_level: str = "INFO"
    ) -> None:
        super().__init__("SHAPSelector")
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level
        self.shap_importances_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> SHAPSelector:
        logger.info("Executing %s fit verification gate...", self.name)
        if y is None:
            logger.warning("%s requires target labels y for computing SHAP contributions. Retaining all.", self.name)
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Numeric columns selection
        numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        if not numeric_cols:
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Downsample to avoid memory and performance OOM/stalls during tree explaining
        max_samples = 5000
        if len(X) > max_samples:
            logger.info("Downsampling %s training data to %d rows to avoid CPU explaining execution bottlenecks...", self.name, max_samples)
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(X.index, size=max_samples, replace=False)
            X_subset = X.loc[indices, numeric_cols]
            y_subset = y.loc[indices]
        else:
            X_subset = X[numeric_cols]
            y_subset = y

        # Fill NaNs for classifier model
        X_imputed = X_subset.fillna(0.0)

        # Baseline model training
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=20,
            max_depth=6,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        model.fit(X_imputed, y_subset)

        # Compute SHAP values
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_imputed)

        # Handle list and 3D array return structure for classification / multi-class models
        if isinstance(shap_vals, list):
            if len(shap_vals) > 1:
                shap_vals = shap_vals[1]
            else:
                shap_vals = shap_vals[0]
        elif isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 3:
                # Shape could be (n_samples, n_features, n_classes) or (n_samples, n_classes, n_features)
                if shap_vals.shape[1] == X_imputed.shape[1]:
                    if shap_vals.shape[2] > 1:
                        shap_vals = shap_vals[..., 1]
                    else:
                        shap_vals = shap_vals[..., 0]
                elif shap_vals.shape[2] == X_imputed.shape[1]:
                    if shap_vals.shape[1] > 1:
                        shap_vals = shap_vals[:, 1, :]
                    else:
                        shap_vals = shap_vals[:, 0, :]

        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

        # Normalize relative to the maximum mean absolute SHAP value
        max_val = float(np.max(mean_abs_shap)) if len(mean_abs_shap) > 0 else 1.0
        if max_val == 0.0:
            max_val = 1.0

        normalized_shap = mean_abs_shap / max_val
        feature_scores = dict(zip(numeric_cols, normalized_shap))
        self.shap_importances_ = feature_scores

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

        logger.info("%s: Retained %d, Dropped %d low contribution features.", self.name, len(self.selected_features_), len(self.dropped_features_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in X.columns if c in self.selected_features_]
        return X[cols]


class PermutationImportanceSelector(BaseFeatureSelector):
    """Filters features based on permutation feature importance values."""
    def __init__(
        self,
        threshold: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
        log_level: str = "INFO"
    ) -> None:
        super().__init__("PermutationImportanceSelector")
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level
        self.permutation_importances_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> PermutationImportanceSelector:
        logger.info("Executing %s fit verification gate...", self.name)
        if y is None:
            logger.warning("%s requires target labels y for permutation scoring. Retaining all.", self.name)
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Numeric columns selection
        numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        if not numeric_cols:
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Downsample to avoid slow execution paths
        max_samples = 5000
        if len(X) > max_samples:
            logger.info("Downsampling %s training data to %d rows to avoid CPU explanation bottlenecks...", self.name, max_samples)
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(X.index, size=max_samples, replace=False)
            X_subset = X.loc[indices, numeric_cols]
            y_subset = y.loc[indices]
        else:
            X_subset = X[numeric_cols]
            y_subset = y

        # Fill NaNs for classifier model
        X_imputed = X_subset.fillna(0.0)

        # Baseline model training
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=20,
            max_depth=6,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        model.fit(X_imputed, y_subset)

        # Compute Permutation Importance
        from sklearn.inspection import permutation_importance
        result = permutation_importance(
            model,
            X_imputed,
            y_subset,
            scoring="roc_auc",
            n_repeats=3,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        importances_mean = result.importances_mean

        # Max-scale the scores to normalize between 0 and 1
        max_val = float(np.max(importances_mean)) if len(importances_mean) > 0 else 1.0
        if max_val <= 0.0:
            max_val = 1.0

        normalized_importances = importances_mean / max_val
        feature_scores = dict(zip(numeric_cols, normalized_importances))
        self.permutation_importances_ = feature_scores

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

        logger.info("%s: Retained %d, Dropped %d low contribution features.", self.name, len(self.selected_features_), len(self.dropped_features_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in X.columns if c in self.selected_features_]
        return X[cols]


class RFESelector(BaseFeatureSelector):
    """Filters features using Recursive Feature Elimination with baseline model."""
    def __init__(
        self,
        threshold: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
        log_level: str = "INFO"
    ) -> None:
        super().__init__("RFESelector")
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level
        self.rfe_rankings_: dict[str, int] = {}
        self.rfe_scores_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> RFESelector:
        logger.info("Executing %s fit verification gate...", self.name)
        if y is None:
            logger.warning("%s requires target labels y for recursive ranking. Retaining all.", self.name)
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Numeric columns selection
        numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        if not numeric_cols:
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Downsample to avoid extreme execution time blowups (RFE fits model recursively)
        max_samples = 2000
        if len(X) > max_samples:
            logger.info("Downsampling %s training data to %d rows to avoid CPU explanation bottlenecks...", self.name, max_samples)
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(X.index, size=max_samples, replace=False)
            X_subset = X.loc[indices, numeric_cols]
            y_subset = y.loc[indices]
        else:
            X_subset = X[numeric_cols]
            y_subset = y

        # Fill NaNs for classifier model
        X_imputed = X_subset.fillna(0.0)

        # Baseline model training
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        from sklearn.feature_selection import RFE
        # We select 1 feature recursively down to rank the rest
        rfe = RFE(estimator=model, n_features_to_select=1, step=1)
        rfe.fit(X_imputed, y_subset)

        rankings = rfe.ranking_
        self.rfe_rankings_ = dict(zip(numeric_cols, rankings))

        # Convert ranking to normalized score (score = 1.0 / rank)
        selected_numeric = []
        dropped_numeric = []
        for col, rank in self.rfe_rankings_.items():
            score = 1.0 / float(rank)
            self.rfe_scores_[col] = score
            if score >= self.threshold:
                selected_numeric.append(col)
            else:
                dropped_numeric.append(col)

        non_numeric = list(X.select_dtypes(exclude=[np.number]).columns)

        self.selected_features_ = selected_numeric + non_numeric
        self.dropped_features_ = dropped_numeric

        logger.info("%s: Retained %d, Dropped %d low contribution features.", self.name, len(self.selected_features_), len(self.dropped_features_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in X.columns if c in self.selected_features_]
        return X[cols]


class SequentialSelector(BaseFeatureSelector):
    """Filters features using Sequential Feature Selection (greedy forward selection)."""
    def __init__(
        self,
        n_features_to_select: int = 15,
        random_state: int = 42,
        n_jobs: int = -1,
        log_level: str = "INFO"
    ) -> None:
        super().__init__("SequentialSelector")
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level
        self.selected_mask_: list[bool] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> SequentialSelector:
        logger.info("Executing %s fit verification gate...", self.name)
        if y is None:
            logger.warning("%s requires target labels y for sequential selection. Retaining all.", self.name)
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Numeric columns selection
        numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        if not numeric_cols:
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Downsample to avoid extreme execution time blowups (SFS fits model repeatedly)
        max_samples = 2000
        if len(X) > max_samples:
            logger.info("Downsampling %s training data to %d rows to avoid CPU explanation bottlenecks...", self.name, max_samples)
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(X.index, size=max_samples, replace=False)
            X_subset = X.loc[indices, numeric_cols]
            y_subset = y.loc[indices]
        else:
            X_subset = X[numeric_cols]
            y_subset = y

        # Fill NaNs for classifier model
        X_imputed = X_subset.fillna(0.0)

        # Baseline model training
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        from sklearn.feature_selection import SequentialFeatureSelector
        # Determine number of features to select (never exceed number of available numeric cols)
        select_k = min(self.n_features_to_select, len(numeric_cols))
        
        sfs = SequentialFeatureSelector(
            estimator=model,
            n_features_to_select=select_k,
            direction="forward",
            cv=3,
            scoring="roc_auc",
            n_jobs=self.n_jobs
        )
        sfs.fit(X_imputed, y_subset)

        support = sfs.get_support()
        self.selected_mask_ = list(support)

        selected_numeric = []
        dropped_numeric = []
        for col, is_selected in zip(numeric_cols, support):
            if is_selected:
                selected_numeric.append(col)
            else:
                dropped_numeric.append(col)

        non_numeric = list(X.select_dtypes(exclude=[np.number]).columns)

        self.selected_features_ = selected_numeric + non_numeric
        self.dropped_features_ = dropped_numeric

        logger.info("%s: Retained %d, Dropped %d low contribution features.", self.name, len(self.selected_features_), len(self.dropped_features_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in X.columns if c in self.selected_features_]
        return X[cols]


class BorutaSelector(BaseFeatureSelector):
    """Filters features using a custom vectorized Boruta shadow permuted feature importance algorithm."""
    def __init__(
        self,
        threshold: float = 0.05,
        n_iterations: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        log_level: str = "INFO"
    ) -> None:
        super().__init__("BorutaSelector")
        self.threshold = threshold
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level
        self.boruta_scores_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> BorutaSelector:
        logger.info("Executing %s fit verification gate...", self.name)
        if y is None:
            logger.warning("%s requires target labels y for shadow scoring. Retaining all.", self.name)
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Numeric columns selection
        numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        if not numeric_cols:
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Downsample to avoid memory and execution stalling
        max_samples = 5000
        if len(X) > max_samples:
            logger.info("Downsampling %s training data to %d rows to avoid CPU explanation bottlenecks...", self.name, max_samples)
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(X.index, size=max_samples, replace=False)
            X_subset = X.loc[indices, numeric_cols]
            y_subset = y.loc[indices]
        else:
            X_subset = X[numeric_cols]
            y_subset = y

        # Fill NaNs for classifier model
        X_imputed = X_subset.fillna(0.0)

        # Baseline model training setup
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=20,
            max_depth=5,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        n_features = len(numeric_cols)
        hits = np.zeros(n_features)
        
        # Track statistics across iterations
        rng = np.random.RandomState(self.random_state)
        for i in range(self.n_iterations):
            # Create shadow features: shuffle copy of original columns
            X_shadow = X_imputed.copy()
            for col in X_shadow.columns:
                X_shadow[col] = rng.permutation(X_shadow[col].values)
            X_shadow.columns = [f"shadow_{c}" for c in numeric_cols]

            # Concatenate original and shadow features side-by-side
            X_merged = pd.concat([X_imputed, X_shadow], axis=1)

            # Fit estimator
            model.fit(X_merged, y_subset)
            importances = model.feature_importances_

            # Extract original and shadow importances
            orig_imp = importances[:n_features]
            shad_imp = importances[n_features:]

            # Maximum shadow features importance
            max_shadow_imp = float(np.max(shad_imp)) if len(shad_imp) > 0 else 0.0

            # Record hits where original > max shadow
            hits += (orig_imp > max_shadow_imp).astype(int)

        # Scores range between 0.0 and 1.0 (hitrate)
        scores = hits / float(self.n_iterations)
        self.boruta_scores_ = dict(zip(numeric_cols, scores))

        selected_numeric = []
        dropped_numeric = []
        for col, score in self.boruta_scores_.items():
            if score >= self.threshold:
                selected_numeric.append(col)
            else:
                dropped_numeric.append(col)

        non_numeric = list(X.select_dtypes(exclude=[np.number]).columns)

        self.selected_features_ = selected_numeric + non_numeric
        self.dropped_features_ = dropped_numeric

        logger.info("%s: Retained %d, Dropped %d low contribution features.", self.name, len(self.selected_features_), len(self.dropped_features_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in X.columns if c in self.selected_features_]
        return X[cols]


class SimulatedAnnealingSelector(BaseFeatureSelector):
    """Filters features using a Simulated Annealing metadata search optimizer."""
    def __init__(
        self,
        threshold: float = 0.05,
        n_iterations: int = 15,
        T0: float = 1.0,
        alpha: float = 0.85,
        feature_penalty: float = 0.01,
        random_state: int = 42,
        n_jobs: int = -1,
        log_level: str = "INFO"
    ) -> None:
        super().__init__("SimulatedAnnealingSelector")
        self.threshold = threshold
        self.n_iterations = n_iterations
        self.T0 = T0
        self.alpha = alpha
        self.feature_penalty = feature_penalty
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.log_level = log_level
        self.SA_scores_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> SimulatedAnnealingSelector:
        logger.info("Executing %s fit verification gate...", self.name)
        if y is None:
            logger.warning("%s requires target labels y for energetic search. Retaining all.", self.name)
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Numeric columns selection
        numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        if not numeric_cols:
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self

        # Downsample to avoid extreme OOM and processing delays
        max_samples = 2000
        if len(X) > max_samples:
            logger.info("Downsampling %s training data to %d rows to avoid CPU explanation bottlenecks...", self.name, max_samples)
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(X.index, size=max_samples, replace=False)
            X_subset = X.loc[indices, numeric_cols]
            y_subset = y.loc[indices]
        else:
            X_subset = X[numeric_cols]
            y_subset = y

        # Fill NaNs for classifier model
        X_imputed = X_subset.fillna(0.0)

        # Baseline model training setup
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        model = RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        n_features = len(numeric_cols)
        rng = np.random.RandomState(self.random_state)

        # Initial random state mask
        current_mask = rng.rand(n_features) > 0.5
        # Ensure at least one feature is selected
        if not np.any(current_mask):
            current_mask[0] = True

        def evaluate_mask(mask: np.ndarray) -> float:
            selected_indices = np.where(mask)[0]
            if len(selected_indices) == 0:
                return 0.0
            
            selected_features = [numeric_cols[idx] for idx in selected_indices]
            X_sel = X_imputed[selected_features]
            
            # 3-Fold CV ROC-AUC
            scores = cross_val_score(model, X_sel, y_subset, cv=3, scoring="roc_auc", n_jobs=self.n_jobs)
            mean_cv = float(np.mean(scores))
            
            # Apply feature penalty to favor smaller subsets
            penalty = self.feature_penalty * (len(selected_indices) / n_features)
            return mean_cv - penalty

        current_energy = evaluate_mask(current_mask)
        best_mask = current_mask.copy()
        best_energy = current_energy

        T = self.T0
        
        # Annealing Loop
        for step in range(self.n_iterations):
            # Generate neighbor: mutate (flip) 1 or 2 random positions
            neighbor_mask = current_mask.copy()
            flip_k = rng.choice([1, 2])
            flip_indices = rng.choice(n_features, size=min(flip_k, n_features), replace=False)
            for idx in flip_indices:
                neighbor_mask[idx] = not neighbor_mask[idx]
            
            if not np.any(neighbor_mask):
                neighbor_mask[rng.choice(n_features)] = True

            neighbor_energy = evaluate_mask(neighbor_mask)

            # Accept/Reject logic
            if neighbor_energy > current_energy:
                current_mask = neighbor_mask.copy()
                current_energy = neighbor_energy
                if neighbor_energy > best_energy:
                    best_mask = neighbor_mask.copy()
                    best_energy = neighbor_energy
            else:
                dE = current_energy - neighbor_energy
                prob = np.exp(-dE / max(T, 1e-8))
                if rng.rand() < prob:
                    current_mask = neighbor_mask.copy()
                    current_energy = neighbor_energy

            # Cooling step
            T *= self.alpha

        # Score features in best_mask as 1.0, otherwise 0.0
        selected_numeric = []
        dropped_numeric = []
        for i, col in enumerate(numeric_cols):
            score = 1.0 if best_mask[i] else 0.0
            self.SA_scores_[col] = score
            if score >= self.threshold:
                selected_numeric.append(col)
            else:
                dropped_numeric.append(col)

        non_numeric = list(X.select_dtypes(exclude=[np.number]).columns)

        self.selected_features_ = selected_numeric + non_numeric
        self.dropped_features_ = dropped_numeric

        logger.info("%s: Retained %d, Dropped %d low contribution features.", self.name, len(self.selected_features_), len(self.dropped_features_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in X.columns if c in self.selected_features_]
        return X[cols]







