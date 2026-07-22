"""Pipeline class grouping and sequentially executing feature selection selectors."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.feature_selection.base import BaseFeatureSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelectionPipeline:
    """Orchestrates sequential running of feature selection steps maintaining comprehensive metrics."""
    def __init__(self, selectors: list[BaseFeatureSelector] | None = None) -> None:
        self.selectors = selectors or []
        self.selected_features_: list[str] = []
        self.dropped_features_: dict[str, list[str]] = {}
        self.history_: list[dict[str, Any]] = []

    def add_selector(self, selector: BaseFeatureSelector) -> FeatureSelectionPipeline:
        """Appends a new selection filtering step to the pipeline execution registry."""
        self.selectors.append(selector)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureSelectionPipeline:
        """Sequential fit runner executing selectors one after the other on reduced subsets."""
        logger.info("Fitting Feature Selection Pipeline containing %d selectors...", len(self.selectors))
        current_features = list(X.columns)
        current_df = X.copy()
        
        self.selected_features_ = current_features
        self.dropped_features_ = {}
        self.history_ = []

        for selector in self.selectors:
            # We fit the selector on the currently active feature subset
            selector.fit(current_df[current_features], y)
            
            # Record dropped features
            dropped = selector.dropped_features_
            self.dropped_features_[selector.name] = dropped
            
            # Recalculate remaining active features
            current_features = [c for c in current_features if c not in dropped]
            
            self.history_.append({
                "selector_name": selector.name,
                "retained_count": len(current_features),
                "dropped_count": len(dropped),
                "dropped_features": list(dropped)
            })
            
            logger.info("Pipeline step %s filtered features. Remaining: %d.", selector.name, len(current_features))

        self.selected_features_ = current_features
        logger.info("Feature Selection Pipeline fitting complete. Final selected count: %d", len(self.selected_features_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies selection transforms restricting columns to survivors of all pipeline selectors."""
        cols = [c for c in X.columns if c in self.selected_features_]
        return X[cols]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Sequentially fits and transforms target dataset."""
        return self.fit(X, y).transform(X)

    def get_summary_report(self) -> dict[str, Any]:
        """Compiles structural diagnosis detail metrics of drop logs across execution steps."""
        return {
            "total_initial_features": sum(h["dropped_count"] for h in self.history_) + len(self.selected_features_),
            "total_final_features": len(self.selected_features_),
            "steps": self.history_,
            "dropped_by_selector": self.dropped_features_,
            "selected_features": self.selected_features_
        }
