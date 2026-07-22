"""Unit tests verifying ProbabilityCalibrator calibration calculations."""

import pytest
import pandas as pd
import numpy as np

from src.evaluation.calibration import ProbabilityCalibrator


def test_probability_calibrator() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    
    calibrator = ProbabilityCalibrator(n_bins=3)
    metrics = calibrator.analyze(y_true, y_probs)
    
    assert metrics["brier_score_loss"] < 0.1
    assert len(metrics["true_probabilities"]) == 2
    
    y_empty = np.array([])
    with pytest.raises(ValueError):
        calibrator.analyze(y_empty, y_empty)
