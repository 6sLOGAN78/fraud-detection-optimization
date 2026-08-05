"""Unit tests for Part 13 — Enterprise Testing Strategy, QA & Acceptance Quality Gate."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.deployment.app import app
from src.utils import (
    APIContractTestRunner,
    AcceptanceQualityGate,
    DataValidationTestRunner,
    EndToEndPipelineTestRunner,
    IntegrationTestRunner,
    ModelInvariantTestRunner,
    PerformanceLoadTestRunner,
    RegressionTestRunner,
    TestingStrategyPreExecutionGate,
    UnitTestFrameworkRunner,
)


def test_13_1_to_13_5_testing_framework():
    gate = TestingStrategyPreExecutionGate()
    assert gate.verify() is True

    runner = IntegrationTestRunner()
    assert runner.verify_pipeline_stage_connection("data/interim/train_merged.parquet", "train_cleaned.parquet") is True

    df_x = pd.DataFrame(np.random.randn(10, 2), columns=["TransactionAmt", "card1"])
    y = np.random.randint(0, 2, 10)
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(df_x, y)

    reg = RegressionTestRunner()
    res = reg.evaluate_regression(clf, df_x, y, expected_min_auc=0.5)
    assert res["regression_passed"] is True

    load_runner = PerformanceLoadTestRunner()
    load_res = load_runner.simulate_concurrent_load(lambda req: {"p": 0.1}, {"amt": 10}, n_requests=10)
    assert load_res["sla_p95_passed"] is True


def test_13_6_to_13_8_validations():
    df = pd.DataFrame({"isFraud": [0, 1], "TransactionAmt": [10.0, 50.0]})
    data_val = DataValidationTestRunner()
    res = data_val.validate_dataset_contract(df, required_cols=["isFraud"])
    assert res["contract_passed"] is True

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(df[["TransactionAmt"]], df["isFraud"])
    invariant = ModelInvariantTestRunner()
    inv_res = invariant.test_monotonicity(clf, df.iloc[[0]][["TransactionAmt"]], "TransactionAmt")
    assert "monotonicity_passed" in inv_res

    api_runner = APIContractTestRunner()
    api_res = api_runner.test_api_contracts(app)
    assert api_res["api_contract_passed"] is True


def test_13_9_to_13_10_acceptance():
    e2e = EndToEndPipelineTestRunner()
    e2e_res = e2e.validate_e2e_pipeline()
    assert "e2e_passed" in e2e_res

    gate = AcceptanceQualityGate()
    res = gate.evaluate_quality_gate(test_suite_passed=True, model_auc_score=0.90, p95_latency_ms=10.0)
    assert res["production_accepted"] is True
    assert res["status"] == "RELEASE_APPROVED"
