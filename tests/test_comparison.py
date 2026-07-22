"""Unit tests verifying ModelComparator quantitative comparison metrics."""

import pytest
import pandas as pd
import numpy as np

from src.evaluation.comparison import ModelComparator


class MockClassifier:
    """Mock classifier to simulate predictions in comparison tests."""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Return probability matrix where second column has 0.1 and 0.8
        probs = np.array([0.1, 0.8, 0.2, 0.9])
        return np.column_stack([1 - probs, probs])


def test_model_comparator() -> None:
    y_test = np.array([0, 1, 0, 1])
    X_test = pd.DataFrame({"feat1": [1, 2, 3, 4]})
    
    models = {
        "Mock1": MockClassifier(threshold=0.5),
        "Mock2": MockClassifier(threshold=0.3)
    }
    
    comparator = ModelComparator()
    results = comparator.compare(models, X_test, y_test)
    
    assert "Mock1" in results["comparisons"]
    assert "Mock2" in results["comparisons"]
    assert results["comparisons"]["Mock1"]["accuracy"] == 1.0
    
    y_empty = np.array([])
    with pytest.raises(ValueError):
        comparator.compare(models, X_empty := pd.DataFrame(), y_empty)
