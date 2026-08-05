"""Unit tests for Part 8 — Advanced Model Evaluation Framework."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.evaluation import (
    BusinessKPIEvaluator,
    CalibrationAnalysisEngine,
    ConfusionMatrixEngine,
    ErrorAnalysisEngine,
    EvaluationFrameworkDesign,
    EvaluationImplementationStandards,

    EvaluationPreExecutionGate,
    F1ScoreEngine,
    KSStatisticEngine,
    LiftGainEngine,
    MCCEngine,
    PRAUCEngine,
    PrecisionEngine,
    ROCAUCEngine,
    RecallEngine,
    RobustnessEvaluationEngine,
    ThresholdAnalysisEngine,
)


@pytest.fixture
def sample_eval_data():
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.85, 0.3, 0.15, 0.7, 0.95, 0.05, 0.88])
    X = pd.DataFrame(np.random.randn(10, 4), columns=["f1", "f2", "f3", "f4"])
    amounts = np.array([50.0, 120.0, 10.0, 200.0, 45.0, 30.0, 150.0, 300.0, 25.0, 80.0])
    return X, y_true, y_prob, amounts


def test_8_1_framework(sample_eval_data):
    X, yt, yp, _ = sample_eval_data
    design = EvaluationFrameworkDesign()
    clean_yt, clean_yp = design.validate_inputs(yt, yp)
    assert len(clean_yt) == 10
    assert len(clean_yp) == 10

    div = EvaluationImplementationStandards.safe_divide(10.0, 0.0, default=0.0)
    assert div == 0.0


def test_8_2_to_8_8_metrics(sample_eval_data):
    _, yt, yp, _ = sample_eval_data

    roc = ROCAUCEngine().calculate(yt, yp)
    assert roc["roc_auc"] > 0.9

    pr = PRAUCEngine().calculate(yt, yp)
    assert pr["pr_auc"] > 0.8

    prec = PrecisionEngine().calculate(yt, yp, threshold=0.5)
    assert prec == 1.0

    rec = RecallEngine().calculate(yt, yp, threshold=0.5)
    assert rec == 1.0

    f1 = F1ScoreEngine().calculate(yt, yp, threshold=0.5)
    assert f1 == 1.0

    mcc = MCCEngine().calculate(yt, yp, threshold=0.5)
    assert mcc == 1.0

    ks = KSStatisticEngine().calculate(yt, yp)
    assert ks["ks_statistic"] > 0.8


def test_8_9_to_8_12_diagnostics(sample_eval_data):
    _, yt, yp, _ = sample_eval_data

    cal = CalibrationAnalysisEngine().calculate(yt, yp, n_bins=5)
    assert "brier_score" in cal
    assert "expected_calibration_error" in cal

    lift = LiftGainEngine().calculate(yt, yp, n_deciles=2)
    assert lift["total_positives"] == 5

    thresh = ThresholdAnalysisEngine().sweep(yt, yp)
    assert "best_fbeta_threshold" in thresh

    cm = ConfusionMatrixEngine().calculate(yt, yp, threshold=0.5)
    assert cm["true_positives"] == 5
    assert cm["true_negatives"] == 5


def test_8_13_to_8_15_business(sample_eval_data):
    X, yt, yp, amounts = sample_eval_data

    err = ErrorAnalysisEngine().analyze(X, yt, yp, threshold=0.5)
    assert err["total_false_positives"] == 0
    assert err["total_false_negatives"] == 0

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, yt)
    rob = RobustnessEvaluationEngine().evaluate_perturbation(clf, X, yt)
    assert "robustness_score" in rob

    biz = BusinessKPIEvaluator().calculate(yt, yp, amounts)
    assert biz["fraud_prevented_value"] > 0.0
