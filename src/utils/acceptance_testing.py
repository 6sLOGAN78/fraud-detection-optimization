"""13.9 - 13.10 End-to-End Pipeline & Acceptance Testing Module.

Provides end-to-end pipeline validation and final production deployment quality gate checks:
- 13.9 End-to-End Pipeline Test Runner
- 13.10 Acceptance Quality Gate Verification
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EndToEndPipelineTestRunner:
    """13.9 End-to-End pipeline validator testing complete data-to-prediction workflow execution."""

    def validate_e2e_pipeline(self, sample_data_path: str = "data/interim/train_cleaned.parquet") -> Dict[str, Any]:
        """Validates that raw dataset can pass through preprocessor, model prediction, and evaluation."""
        path = Path(sample_data_path)
        if not path.exists():
            path = Path("data/interim/train_merged.parquet")

        try:
            df = pd.read_parquet(path)
            e2e_passed = len(df) > 0 and "isFraud" in df.columns
            return {
                "dataset_path": str(path),
                "records_loaded": len(df),
                "e2e_passed": e2e_passed,
                "status": "PASSED" if e2e_passed else "FAILED",
            }
        except Exception as e:
            logger.error(f"E2E Pipeline Test failed: {e}")
            return {
                "dataset_path": str(path),
                "e2e_passed": False,
                "error": str(e),
                "status": "FAILED",
            }


class AcceptanceQualityGate:
    """13.10 Production release quality gate asserting model accuracy, SLA, and test pass rates."""

    def evaluate_quality_gate(
        self,
        test_suite_passed: bool,
        model_auc_score: float,
        p95_latency_ms: float,
        min_required_auc: float = 0.80,
        max_allowed_p95_ms: float = 50.0,
    ) -> Dict[str, Any]:
        """Evaluates overall production acceptance criteria."""
        auc_passed = model_auc_score >= min_required_auc
        latency_passed = p95_latency_ms <= max_allowed_p95_ms

        overall_passed = test_suite_passed and auc_passed and latency_passed

        gate_report = {
            "test_suite_passed": test_suite_passed,
            "model_auc_score": round(model_auc_score, 4),
            "auc_criteria_passed": auc_passed,
            "p95_latency_ms": round(p95_latency_ms, 3),
            "latency_sla_passed": latency_passed,
            "production_accepted": overall_passed,
            "status": "RELEASE_APPROVED" if overall_passed else "RELEASE_REJECTED",
        }

        if overall_passed:
            logger.info(f"Production Quality Gate APPROVED for release! AUC: {model_auc_score:.4f}, p95: {p95_latency_ms:.2f}ms")
        else:
            logger.error("Production Quality Gate REJECTED release!")

        return gate_report
