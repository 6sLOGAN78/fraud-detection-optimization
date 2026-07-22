"""Unit tests verifying ThresholdOptimizer decision cutoff search."""

import pytest
import pandas as pd
import numpy as np

from src.evaluation.thresholds import ThresholdOptimizer


def test_threshold_optimizer() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    
    optimizer = ThresholdOptimizer(target_metric="f1")
    best_thresh = optimizer.optimize(y_true, y_probs)
    
    assert 0.3 < best_thresh < 0.7
    assert optimizer.optimization_metrics_["best_score"] == 1.0
    
    y_empty = np.array([])
    with pytest.raises(ValueError):
        optimizer.optimize(y_empty, y_empty)
