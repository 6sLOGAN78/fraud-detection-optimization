"""Unit tests for Part 12 — Production MLOps Monitoring, Alerting & Retraining."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.monitoring import (
    AlertingEngine,
    AutomatedRetrainingPipeline,
    ChampionChallengerLifecycleManager,
    ConceptDriftMonitor,
    DataDriftMonitor,
    FeatureHealthMonitor,
    MLOpsPreExecutionGate,
    ModelPerformanceMonitor,
    PredictionDistributionMonitor,
    ServicePerformanceMonitor,
)


@pytest.fixture
def sample_monitoring_data():
    np.random.seed(42)
    ref_df = pd.DataFrame(np.random.randn(50, 4), columns=["f1", "f2", "f3", "f4"])
    cur_df = pd.DataFrame(np.random.randn(50, 4) + 0.5, columns=["f1", "f2", "f3", "f4"])
    y_ref = np.random.randint(0, 2, 50)
    y_cur = np.random.randint(0, 2, 50)
    return ref_df, cur_df, y_ref, y_cur


def test_12_1_and_12_7_service_monitoring():
    gate = MLOpsPreExecutionGate()
    assert gate.verify() is True

    mon = ServicePerformanceMonitor()
    mon.record_request(10.0)
    mon.record_request(50.0)
    mon.record_request(100.0, is_error=True)

    metrics = mon.get_service_metrics()
    assert metrics["total_requests"] == 3
    assert metrics["error_rate"] == round(1 / 3, 4)
    assert metrics["p50_latency_ms"] == 50.0


def test_12_2_to_12_6_drift_monitoring(sample_monitoring_data):
    ref_df, cur_df, y_ref, y_cur = sample_monitoring_data

    data_mon = DataDriftMonitor()
    drift = data_mon.evaluate_drift(ref_df, cur_df, psi_threshold=0.1)
    assert "drifted_features_count" in drift

    concept_mon = ConceptDriftMonitor()
    concept = concept_mon.evaluate_target_drift(y_ref, y_cur)
    assert "rate_difference" in concept

    pred_mon = PredictionDistributionMonitor()
    pred_res = pred_mon.evaluate_prediction_drift(np.random.rand(50), np.random.rand(50))
    assert "prediction_psi" in pred_res

    health_mon = FeatureHealthMonitor()
    health = health_mon.evaluate_feature_health(cur_df)
    assert health["total_columns"] == 4


def test_12_8_alerting():
    alert_engine = AlertingEngine()
    event = alert_engine.trigger_alert("TestAlert", "Test message", severity="WARNING")
    assert event["alert_name"] == "TestAlert"
    assert event["severity"] == "WARNING"


def test_12_9_and_12_10_lifecycle_and_retraining(sample_monitoring_data):
    ref_df, cur_df, y_ref, y_cur = sample_monitoring_data

    champ = RandomForestClassifier(n_estimators=5, random_state=42)
    champ.fit(ref_df, y_ref)

    chall = RandomForestClassifier(n_estimators=20, random_state=42)
    chall.fit(ref_df, y_ref)

    lifecycle = ChampionChallengerLifecycleManager(champion_model=champ, challenger_model=chall)
    res = lifecycle.evaluate_shadow_promotion(cur_df, y_cur)
    assert "promoted" in res

    retrain = AutomatedRetrainingPipeline()
    should_retrain = retrain.should_trigger_retraining({"has_severe_data_drift": True}, {"is_performance_decayed": False})
    assert should_retrain is True

    new_model = retrain.execute_retrain(ref_df, y_ref)
    assert hasattr(new_model, "predict")
