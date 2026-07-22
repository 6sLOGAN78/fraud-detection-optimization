"""Abstract base class establishing input/output contracts for feature selection selectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseFeatureSelector(ABC):
    """Abstract Base Selector interface enforcing layout structure consistency for statistical selectors."""
    def __init__(self, name: str) -> None:
        self.name = name
        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> BaseFeatureSelector:
        """Analyzes training feature relationships and stores selection masks."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms dataframe X by retaining only selected features."""
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fits selector criteria and yields selected subset dataframe."""
        return self.fit(X, y).transform(X)
