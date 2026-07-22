"""Unit tests verifying Logistic Regression and XGBoost baseline classifiers."""

import pytest
import pandas as pd
import numpy as np

from src.models.baseline import LogisticRegressionBaseline, XGBoostBaseline


def test_logistic_regression_baseline() -> None:
    X = pd.DataFrame({
        "f1": np.random.randn(30),
        "f2": np.random.randn(30)
    })
    y = pd.Series([1, 0, 1] * 10)

    lr_model = LogisticRegressionBaseline(random_state=42)
    lr_model.fit(X, y)

    assert "fit_time_seconds" in lr_model.fit_metrics_
    assert "memory_delta_mb" in lr_model.fit_metrics_

    preds = lr_model.predict(X)
    assert len(preds) == 30
    probs = lr_model.predict_proba(X)
    assert probs.shape == (30, 2)

    # Empty check
    with pytest.raises(ValueError):
        lr_model.fit(pd.DataFrame(), y)

    # Fallback predictability on prediction exception
    lr_model.pipeline = None  # force error
    fallback_preds = lr_model.predict(X)
    np.testing.assert_array_equal(fallback_preds, np.zeros(30, dtype=int))


def test_xgboost_baseline() -> None:
    X = pd.DataFrame({
        "f1": np.random.randn(30),
        "f2": np.random.randn(30)
    })
    y = pd.Series([1, 0, 1] * 10)

    xgb_model = XGBoostBaseline(random_state=42)
    xgb_model.fit(X, y)

    assert "fit_time_seconds" in xgb_model.fit_metrics_
    assert "memory_delta_mb" in xgb_model.fit_metrics_

    preds = xgb_model.predict(X)
    assert len(preds) == 30
    probs = xgb_model.predict_proba(X)
    assert probs.shape == (30, 2)

    # Empty check
    with pytest.raises(ValueError):
        xgb_model.fit(pd.DataFrame(), y)

    # Fallback check
    xgb_model.model = None  # force error
    fallback_probs = xgb_model.predict_proba(X)
    np.testing.assert_array_equal(fallback_probs, np.zeros((30, 2)))
