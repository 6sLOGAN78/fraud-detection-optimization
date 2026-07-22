"""Unit tests verifying CatBoostClassifierWrapper baseline formulation."""

import pytest
import pandas as pd
import numpy as np

from src.models.catboost_model import CatBoostClassifierWrapper


def test_catboost_estimator() -> None:
    X = pd.DataFrame({
        "f1": np.random.randn(30),
        "f2": np.random.randn(30)
    })
    y = pd.Series([1, 0, 1] * 10)

    model = CatBoostClassifierWrapper(random_state=42)
    model.fit(X, y)

    assert "fit_time_seconds" in model.fit_metrics_
    assert "memory_delta_mb" in model.fit_metrics_

    preds = model.predict(X)
    assert len(preds) == 30
    probs = model.predict_proba(X)
    assert probs.shape == (30, 2)

    with pytest.raises(ValueError):
        model.fit(pd.DataFrame(), y)

    # Fallbackpredictability check
    model.model = None  # force error
    fallback_probs = model.predict_proba(X)
    np.testing.assert_array_equal(fallback_probs, np.zeros((30, 2)))
