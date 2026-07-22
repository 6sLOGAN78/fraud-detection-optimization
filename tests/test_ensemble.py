"""Unit tests verifying EnsembleClassifier blended estimator."""

import pytest
import pandas as pd
import numpy as np

from src.models.ensemble import EnsembleClassifier


def test_ensemble_estimator() -> None:
    X = pd.DataFrame({
        "f1": np.random.randn(30),
        "f2": np.random.randn(30)
    })
    y = pd.Series([1, 0, 1] * 10)

    # Instantiate ensemble and base mock estimators
    ensemble = EnsembleClassifier(weights=[0.4, 0.3, 0.3], threshold=0.5)
    
    class DummyEstimator:
        def predict_proba(self, X):
            probs = np.zeros((len(X), 2))
            probs[:, 0] = 0.5
            probs[:, 1] = 0.5
            return probs

    ensemble.lgbm_model = DummyEstimator()
    ensemble.xgb_model = DummyEstimator()
    ensemble.cat_model = DummyEstimator()
    
    ensemble.fit(X, y)
    
    assert "fit_time_seconds" in ensemble.fit_metrics_
    assert "memory_delta_mb" in ensemble.fit_metrics_
    
    probs = ensemble.predict_proba(X)
    assert probs.shape == (30, 2)
    np.testing.assert_array_almost_equal(probs[:, 1], 0.5 * np.ones(30))
    
    preds = ensemble.predict(X)
    assert len(preds) == 30
    
    with pytest.raises(ValueError):
        ensemble.fit(pd.DataFrame(), y)
        
    # Check predictions fallback
    ensemble.lgbm_model = None
    fallback_probs = ensemble.predict_proba(X)
    np.testing.assert_array_equal(fallback_probs, np.zeros((30, 2)))
