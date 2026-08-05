"""Unit tests for Part 10 — MLOps Experiment Tracking, Model Registry & Reproducibility Framework."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.monitoring import (
    ArtifactManagementEngine,
    DatasetVersionTracker,
    ExperimentManagementArchitecture,
    ExperimentNamingStandards,
    ExperimentPreExecutionGate,
    MLflowModelRegistryManager,
    MLflowTrackingEngine,
    MetricTrackingEngine,
    ParameterTrackingEngine,
    ReproducibilityFramework,
    RunComparisonEngine,
)


def test_10_1_to_10_3_experiment_architecture():
    gate = ExperimentPreExecutionGate()
    assert gate.verify() is True

    exp_name = ExperimentNamingStandards.generate_experiment_name()
    run_name = ExperimentNamingStandards.generate_run_name("lgb")
    assert "ieee-cis" in exp_name
    assert "lgb" in run_name

    arch = ExperimentManagementArchitecture(experiment_name="test_exp")
    exp_id = arch.initialize_session()
    assert exp_id is not None


def test_10_4_to_10_6_logging_engines():
    param_engine = ParameterTrackingEngine()

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.json"
        test_file.write_text('{"a": 1}')

        art_engine = ArtifactManagementEngine()
        payload_file = art_engine.log_dict_as_artifact({"key": "value"}, "payload.json")
        assert payload_file.exists()


def test_10_7_to_10_9_registry_and_dataset():
    tracker = DatasetVersionTracker()
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    meta = tracker.get_dataset_metadata(df)
    assert meta["n_rows"] == 3
    assert meta["n_columns"] == 2

    comp = RunComparisonEngine()
    res = comp.compare_runs(experiment_name="non_existent_exp")
    assert "error" in res or res["total_runs"] == 0


def test_10_10_reproducibility():
    repro = ReproducibilityFramework(seed=123)
    seed = repro.set_global_seed()
    assert seed == 123
    assert np.random.get_state()[1][0] == 123

    env = repro.capture_environment_snapshot()
    assert env["seed"] == 123
    assert "python_version" in env
