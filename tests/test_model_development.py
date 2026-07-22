"""Unit tests for Model Development Architecture modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.models.development import (
    ModelDevelopmentArchitectureDesign,
    CoreModelEngine,
    InputOutputProcessor,
    ModelImplementationStandards,
    ModelDevelopmentPipeline
)


def test_model_development_architecture_design() -> None:
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0],
        "feature2": [10.0, 20.0, 30.0, 40.0]
    })
    y = pd.Series([0, 1, 0, 1])

    design = ModelDevelopmentArchitectureDesign()
    design.fit(df, y)

    assert design.scaler_mean_ is not None
    assert design.scaler_std_ is not None

    transformed = design.transform(df)
    assert transformed.shape == df.shape
    pd.testing.assert_frame_equal(transformed, (df - df.mean()) / df.std().replace(0.0, 1.0))


def test_core_model_engine() -> None:
    X = pd.DataFrame({
        "f1": np.random.randn(20),
        "f2": np.random.randn(20)
    })
    y = pd.Series([0, 1] * 10)

    engine = CoreModelEngine(random_state=42)
    engine.fit(X, y)

    assert "fit_time_seconds" in engine.fit_metrics_
    assert "memory_delta_mb" in engine.fit_metrics_

    preds = engine.predict(X)
    assert len(preds) == 20
    probs = engine.predict_proba(X)
    assert probs.shape == (20, 2)


def test_input_output_processor() -> None:
    X = pd.DataFrame({"f": range(10)})
    y = pd.Series(range(10))

    processor = InputOutputProcessor()
    X_tr, X_val, y_tr, y_val = processor.split_train_val(X, y, val_ratio=0.2)

    assert len(X_tr) == 8
    assert len(X_val) == 2
    assert len(y_tr) == 8
    assert len(y_val) == 2
    assert X_tr.iloc[-1]["f"] == 7
    assert X_val.iloc[0]["f"] == 8


def test_model_implementation_standards() -> None:
    X = pd.DataFrame({
        "f1": np.random.randn(10),
        "f2": np.random.randn(10)
    })
    y = pd.Series([0, 1] * 5)
    
    engine = CoreModelEngine(random_state=42)
    engine.fit(X, y)

    standards = ModelImplementationStandards()

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "model_bundle.pkl"
        bundle = {"model": engine, "metric": 0.95}

        standards.serialize(bundle, bundle_path)
        assert bundle_path.exists()

        loaded_bundle = standards.deserialize(bundle_path)
        assert loaded_bundle["metric"] == 0.95
        assert isinstance(loaded_bundle["model"], CoreModelEngine)

    # Safe predict checks
    probs = standards.predict_safe(engine, X)
    assert len(probs) == 10

    # Trigger fallback check
    bad_probs = standards.predict_safe(None, X, fallback_val=0.55)
    np.testing.assert_array_equal(bad_probs, np.full(10, 0.55))


def test_model_development_pipeline() -> None:
    X = pd.DataFrame({
        "f1": np.random.randn(100),
        "f2": np.random.randn(100)
    })
    y = pd.Series([0, 1] * 50)

    pipeline = ModelDevelopmentPipeline(random_state=42)
    summary = pipeline.fit_and_validate(X, y, val_ratio=0.2)

    assert "train_auc" in summary
    assert "val_auc" in summary
    assert "train_accuracy" in summary
    assert "val_accuracy" in summary
    assert "fit_metrics" in summary
