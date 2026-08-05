"""Unit tests for Part 7 — Hyperparameter Optimization Framework."""

import tempfile
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.optimization import (
    BestConfigurationRegistry,
    BayesianOptimizationEngine,
    EarlyStoppingHandler,
    OptimizationArchitectureDesign,
    OptimizationImplementationStandards,
    OptimizationInputOutputProcessor,
    OptimizationMonitor,
    OptimizationPipelineExecution,
    OptunaEarlyStoppingCallback,
    OptunaStudyManager,
    ParallelOptimizationExecutor,
    SearchSpaceBuilder,
    SearchSpaceValidator,
    StatisticalMetricSpec,
    ThresholdTuningCriteria,
    TrialPruningEngine,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100), name="isFraud")
    return X, y


def test_7_1_optimization_architecture(sample_data):
    X, y = sample_data
    arch = OptimizationArchitectureDesign()
    arch.fit(X, y)
    X_trans = arch.transform(X)
    assert X_trans.shape == X.shape

    io_proc = OptimizationInputOutputProcessor()
    assert io_proc.validate_inputs(X, y) is True

    standards = OptimizationImplementationStandards()
    assert standards.verify_stability(0.95) is True
    assert standards.verify_stability(float("nan")) is False


def test_7_2_search_space_design():
    space = SearchSpaceBuilder.get_default_space("lightgbm")
    assert "n_estimators" in space
    assert "learning_rate" in space

    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1e-3,
        "reg_lambda": 1e-3,
    }
    assert SearchSpaceValidator.validate(params, space) is True

    metric_spec = StatisticalMetricSpec("roc_auc")
    assert metric_spec.direction == "maximize"
    assert metric_spec.is_better(0.90, 0.85) is True

    thresh = ThresholdTuningCriteria()
    assert len(thresh.get_candidate_thresholds()) > 0


def test_7_3_bayesian_optimization(sample_data):
    X, y = sample_data
    engine = BayesianOptimizationEngine(n_trials=3, n_splits=2)

    def model_factory(params):
        return RandomForestClassifier(**params, random_state=42)

    score = engine.evaluate_cv_objective(model_factory, {"n_estimators": 10}, X, y)
    assert 0.0 <= score <= 1.0

    engine.record_trial(1, {"n_estimators": 10}, score, 0.5)
    summary = engine.get_summary()
    assert summary["total_trials"] == 1
    assert summary["best_score"] == score


def test_7_4_optuna_framework(sample_data):
    X, y = sample_data
    study_manager = OptunaStudyManager(
        study_name="test_optuna_study", sampler_name="tpe", pruner_name="median"
    )

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 5, 20)
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X, y)
        return float(clf.score(X, y))

    study = study_manager.run_optimization(objective, n_trials=3)
    best = study_manager.get_best_results()
    assert "best_value" in best
    assert best["n_trials"] == 3


def test_7_5_early_stopping():
    handler = EarlyStoppingHandler(patience=2, min_delta=0.01, mode="maximize")
    assert handler.update(0.80) is False
    assert handler.update(0.805) is False  # Delta < 0.01
    assert handler.update(0.805) is True   # Patience reached

    cb = OptunaEarlyStoppingCallback(patience=2)
    assert cb.no_improvement_counter == 0


def test_7_6_trial_pruning():
    engine = TrialPruningEngine(pruner_type="median")
    assert engine.pruner is not None


def test_7_7_parallel_optimization():
    executor = ParallelOptimizationExecutor(n_jobs=1)
    seed = executor.get_worker_seed(2)
    assert seed == 42 + 20014


def test_7_8_monitoring():
    monitor = OptimizationMonitor(experiment_name="test_exp")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2)
    summary = monitor.log_study_summary(study, model_type="test_model")
    assert summary["completed_trials"] == 2


def test_7_9_best_configuration_registry():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = BestConfigurationRegistry(registry_dir=tmpdir)
        json_path = registry.register_configuration(
            model_name="lightgbm",
            best_params={"learning_rate": 0.05, "n_estimators": 100},
            best_score=0.925,
            metric_name="roc_auc",
        )
        assert json_path.exists()
        loaded = registry.load_configuration("lightgbm")
        assert loaded["best_score"] == 0.925
        assert loaded["hyperparameters"]["learning_rate"] == 0.05
