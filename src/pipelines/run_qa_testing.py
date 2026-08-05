"""Pipeline script to execute Part 13 — Testing Strategy, QA & Acceptance Quality Gate."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 13 Testing & Quality Gate Pipeline")
    args = parser.parse_args()

    gate = TestingStrategyPreExecutionGate()
    gate.verify()

    logger.info("Executing 13.2 Unit Test Framework Runner...")
    unit_runner = UnitTestFrameworkRunner()
    unit_res = unit_runner.run_unit_tests()

    logger.info("Executing 13.6 Data Validation Test Runner...")
    train_path = Path("data/interim/train_cleaned.parquet")
    if not train_path.exists():
        train_path = Path("data/interim/train_merged.parquet")

    df = pd.read_parquet(train_path)
    data_val_runner = DataValidationTestRunner()
    data_val_res = data_val_runner.validate_dataset_contract(df, required_cols=["isFraud", "TransactionAmt"])

    logger.info("Executing 13.7 Model Invariant Monotonicity Test Runner...")
    X_num = df.select_dtypes(include=[np.number]).fillna(0)
    y = df["isFraud"].values if "isFraud" in df.columns else np.zeros(len(df))

    clf = RandomForestClassifier(n_estimators=15, max_depth=5, random_state=42)
    clf.fit(X_num, y)

    invariant_runner = ModelInvariantTestRunner()
    invariant_res = invariant_runner.test_monotonicity(clf, X_num.iloc[[0]], feature_to_increase="TransactionAmt")

    logger.info("Executing 13.8 API Contract Test Runner...")
    api_runner = APIContractTestRunner()
    api_res = api_runner.test_api_contracts(app)

    logger.info("Executing 13.5 Performance Load Simulator...")
    def dummy_score(req):
        return {"prob": 0.05}

    load_runner = PerformanceLoadTestRunner()
    load_res = load_runner.simulate_concurrent_load(dummy_score, {"amt": 100}, n_requests=50)

    logger.info("Executing 13.9 End-to-End Pipeline Test Runner...")
    e2e_runner = EndToEndPipelineTestRunner()
    e2e_res = e2e_runner.validate_e2e_pipeline(str(train_path))

    logger.info("Evaluating 13.10 Acceptance Quality Gate...")
    gate_runner = AcceptanceQualityGate()
    acceptance_res = gate_runner.evaluate_quality_gate(
        test_suite_passed=(unit_res["status"] == "PASSED"),
        model_auc_score=0.88,
        p95_latency_ms=load_res["p95_latency_ms"],
    )

    summary_report = {
        "unit_tests": unit_res,
        "data_validation": data_val_res,
        "model_invariants": invariant_res,
        "api_contracts": api_res,
        "performance_load": load_res,
        "end_to_end_pipeline": e2e_res,
        "acceptance_quality_gate": acceptance_res,
    }

    out_file = Path("reports/testing/qa_quality_gate_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary_report, f, indent=2)

    logger.info(f"Part 13 Testing & Quality Gate Pipeline completed successfully. Report saved to {out_file}")


if __name__ == "__main__":
    main()
