"""Unit tests verifying TimeSeriesCrossValidator temporal splitter."""

import pytest
import pandas as pd
import numpy as np

from src.validation.cv import TimeSeriesCrossValidator


def test_cv_validator() -> None:
    X = pd.DataFrame({
        "f1": np.arange(100),
        "f2": np.arange(100)
    })
    
    cv = TimeSeriesCrossValidator(n_splits=5)
    folds = cv.split(X)
    
    assert len(folds) == 5
    assert len(cv.split_metrics_) == 1
    
    # Train-test split indices validation logic checks
    for train_idx, val_idx in folds:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        # Time-series splits are strictly sequential: train index max should be less than val index min
        assert train_idx[-1] < val_idx[0]
        
    with pytest.raises(ValueError):
        cv.split(pd.DataFrame())
