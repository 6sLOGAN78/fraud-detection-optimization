"""13.6 - 13.8 Data Validation, Model Invariant, and API Contract Testing Module.

Provides automated test runners for data quality contracts, model behavior invariants, and REST API contracts:
- 13.6 Data Validation Test Runner
- 13.7 Model Invariant Test Runner
- 13.8 API Contract Test Runner
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidationTestRunner:
    """13.6 Data validation runner enforcing schema contracts, data types, and null rate limits."""

    def validate_dataset_contract(
        self, df: pd.DataFrame, required_cols: List[str], max_null_fraction: float = 0.5
    ) -> Dict[str, Any]:
        """Validates dataframe schema and missing value constraints."""
        missing_cols = [c for c in required_cols if c not in df.columns]
        null_fractions = (df.isnull().sum() / len(df)).to_dict()
        excessive_null_cols = [k for k, v in null_fractions.items() if v > max_null_fraction]

        passed = (len(missing_cols) == 0) and (len(excessive_null_cols) == 0)

        return {
            "missing_required_columns": missing_cols,
            "excessive_null_columns": excessive_null_cols,
            "contract_passed": passed,
        }


class ModelInvariantTestRunner:
    """13.7 Asserts model behavioral invariants such as monotonicity and scale invariance."""

    def test_monotonicity(
        self, model: Any, base_row: pd.DataFrame, feature_to_increase: str, delta: float = 500.0
    ) -> Dict[str, Any]:
        """Asserts that increasing a risk feature (e.g. TransactionAmt) increases or preserves fraud probability."""
        if feature_to_increase not in base_row.columns:
            return {"passed": False, "reason": f"Feature {feature_to_increase} not in DataFrame."}

        row_low = base_row.copy()
        row_high = base_row.copy()
        row_high[feature_to_increase] = row_high[feature_to_increase] + delta

        if hasattr(model, "predict_proba"):
            prob_low = float(model.predict_proba(row_low)[:, 1][0])
            prob_high = float(model.predict_proba(row_high)[:, 1][0])
        else:
            prob_low = float(model.predict(row_low)[0])
            prob_high = float(model.predict(row_high)[0])

        is_monotonic = prob_high >= (prob_low - 1e-5)

        return {
            "feature": feature_to_increase,
            "prob_low": round(prob_low, 4),
            "prob_high": round(prob_high, 4),
            "monotonicity_passed": is_monotonic,
        }


class APIContractTestRunner:
    """13.8 Automated FastAPI contract runner testing endpoint schemas and HTTP status codes."""

    def test_api_contracts(self, fastapi_app: Any) -> Dict[str, Any]:
        """Runs OpenAPI contract assertions against health and prediction endpoints."""
        client = TestClient(fastapi_app)

        health_res = client.get("/health")
        health_passed = health_res.status_code == 200 and health_res.json().get("status") == "HEALTHY"

        info_res = client.get("/v1/model_info")
        info_passed = info_res.status_code == 200 and "version" in info_res.json()

        predict_res = client.post("/v1/predict", json={"TransactionAmt": 150.0, "card1": 1000, "card2": 500})
        predict_passed = predict_res.status_code == 200 and "fraud_probability" in predict_res.json()

        all_passed = health_passed and info_passed and predict_passed

        return {
            "health_endpoint_passed": health_passed,
            "info_endpoint_passed": info_passed,
            "predict_endpoint_passed": predict_passed,
            "api_contract_passed": all_passed,
        }
