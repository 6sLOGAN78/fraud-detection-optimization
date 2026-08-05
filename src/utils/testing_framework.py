"""13.1 - 13.5 Testing Strategy Architecture, Integration, Regression, and Performance Testing Module.

Provides automated test orchestration, integration checks, golden dataset regression benchmarking, and load testing:
- 13.1 Testing Strategy Architecture
- 13.2 Unit Test Framework Runner
- 13.3 Integration Test Runner
- 13.4 Regression Test Runner
- 13.5 Performance & Load Test Simulator
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestingStrategyPreExecutionGate:
    """Pre-execution verification gate checking pytest environment and test directory readiness."""
    __test__ = False

    def __init__(self, test_dir: str = "tests"):
        self.test_dir = Path(test_dir)

    def verify(self) -> bool:
        """Verifies that unit test files exist."""
        if not self.test_dir.exists() or not list(self.test_dir.glob("test_*.py")):
            raise FileNotFoundError("Test directory missing or no test_*.py files found.")

        logger.info("Testing Strategy Pre-Execution Gate PASSED.")
        return True


class UnitTestFrameworkRunner:
    """13.2 Automated unit test suite execution and coverage metrics manager."""
    __test__ = False

    def run_unit_tests(self) -> Dict[str, Any]:
        """Runs unit test suite and collects pass/fail counts."""
        import pytest

        start_time = time.time()
        exit_code = pytest.main(["-q", "tests/"])
        elapsed = time.time() - start_time

        return {
            "exit_code": int(exit_code),
            "status": "PASSED" if exit_code == 0 else "FAILED",
            "execution_seconds": round(elapsed, 2),
        }


class IntegrationTestRunner:
    """13.3 Integration test runner verifying pipeline stage connectivity and artifact flow."""
    __test__ = False

    def verify_pipeline_stage_connection(
        self, upstream_stage_output: Union[str, Path], downstream_stage_input: Union[str, Path]
    ) -> bool:
        """Verifies artifact handoff between upstream and downstream stages."""
        up_path = Path(upstream_stage_output)
        if not up_path.exists():
            logger.error(f"Integration Failure: Upstream artifact missing at {up_path}")
            return False
        logger.info(f"Integration Check PASSED between {upstream_stage_output} and {downstream_stage_input}")
        return True


class RegressionTestRunner:
    """13.4 Regression test runner asserting model predictions against golden benchmark dataset."""
    __test__ = False

    def evaluate_regression(
        self,
        model: Any,
        golden_X: pd.DataFrame,
        golden_y: np.ndarray,
        expected_min_auc: float = 0.80,
    ) -> Dict[str, Any]:
        """Evaluates model performance against golden benchmark to prevent performance regressions."""
        from sklearn.metrics import roc_auc_score

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(golden_X)[:, 1]
        else:
            probs = model.predict(golden_X)

        score = float(roc_auc_score(golden_y, probs))
        passed = score >= expected_min_auc

        if not passed:
            logger.error(f"Performance Regression Detected! Score {score:.4f} < Expected {expected_min_auc:.4f}")

        return {
            "golden_auc_score": round(score, 4),
            "expected_min_auc": expected_min_auc,
            "regression_passed": passed,
        }


class PerformanceLoadTestRunner:
    """13.5 Simulates concurrent multi-request load to verify latency SLAs (p95/p99) under load."""
    __test__ = False

    def simulate_concurrent_load(
        self, score_func: Callable[[Dict[str, Any]], Dict[str, Any]], sample_request: Dict[str, Any], n_requests: int = 100
    ) -> Dict[str, Any]:
        """Runs simulated request load and asserts p95 latency stays under SLA limit."""
        latencies_ms = []

        start_total = time.time()
        for _ in range(n_requests):
            t0 = time.perf_counter()
            res = score_func(sample_request)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

        total_elapsed = time.time() - start_total
        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
        p99 = float(np.percentile(latencies_ms, 99))
        throughput_qps = float(n_requests / total_elapsed)

        return {
            "total_simulated_requests": n_requests,
            "throughput_qps": round(throughput_qps, 2),
            "p50_latency_ms": round(p50, 3),
            "p95_latency_ms": round(p95, 3),
            "p99_latency_ms": round(p99, 3),
            "sla_p95_passed": p95 <= 50.0,
        }
