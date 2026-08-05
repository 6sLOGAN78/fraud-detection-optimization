"""Unit tests for Part 11 — Production Model Deployment, FastAPI, & Inference Engines."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.deployment import (
    BatchInferenceEngine,
    DeploymentPreExecutionGate,
    DeploymentValidator,
    InferencePipeline,
    ModelPackager,
    ModelSerializer,
    RealTimeInferenceEngine,
    RollbackManager,
)
from src.deployment.app import app


@pytest.fixture
def sample_deployment_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(20, 3), columns=["TransactionAmt", "card1", "card2"])
    y = np.random.randint(0, 2, 20)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    return model, X, y


def test_11_1_to_11_4_packaging_and_inference(sample_deployment_data):
    model, X, _ = sample_deployment_data

    with tempfile.TemporaryDirectory() as tmpdir:
        packager = ModelPackager()
        bundle_dir = packager.create_bundle(model, feature_names=["TransactionAmt", "card1", "card2"], bundle_dir=tmpdir)
        assert (bundle_dir / "model.joblib").exists()
        assert (bundle_dir / "metadata.json").exists()

        loaded_model = ModelSerializer.load_artifact(bundle_dir / "model.joblib")
        pipeline = InferencePipeline(model=loaded_model, feature_names=["TransactionAmt", "card1", "card2"])

        preds, probs = pipeline.predict(X)
        assert len(preds) == 20
        assert len(probs) == 20


def test_11_7_to_11_8_engines(sample_deployment_data):
    model, X, _ = sample_deployment_data
    pipeline = InferencePipeline(model=model, feature_names=["TransactionAmt", "card1", "card2"])

    rt_engine = RealTimeInferenceEngine(pipeline=pipeline)
    res = rt_engine.score_transaction({"TransactionAmt": 150.0, "card1": 1000, "card2": 500})
    assert "fraud_probability" in res
    assert "latency_ms" in res

    batch_engine = BatchInferenceEngine(pipeline=pipeline, chunk_size=5)
    df_out = batch_engine.predict_dataframe(X)
    assert "fraud_probability" in df_out.columns
    assert "is_fraud_prediction" in df_out.columns


def test_11_10_to_11_11_validation_and_rollback(sample_deployment_data):
    model, X, _ = sample_deployment_data
    pipeline = InferencePipeline(model=model, feature_names=["TransactionAmt", "card1", "card2"])

    validator = DeploymentValidator()
    val_res = validator.validate_pipeline(pipeline, {"TransactionAmt": 150.0, "card1": 1000, "card2": 500})
    assert val_res["smoke_test_passed"] is True

    rollback = RollbackManager(primary_pipeline=pipeline)
    preds, probs = rollback.predict_safe(X)
    assert len(preds) == 20


def test_11_5_fastapi_endpoints():
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "HEALTHY"

    response = client.get("/v1/model_info")
    assert response.status_code == 200
    assert response.json()["version"] == "v1"

    predict_req = {"TransactionAmt": 150.0, "card1": 1000, "card2": 500}
    response = client.post("/v1/predict", json=predict_req)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
