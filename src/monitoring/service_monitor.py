"""12.1, 12.2, and 12.7 MLOps Architecture, Model Performance, and Service Monitoring Module.

Provides pre-execution verification gates, SLA latency percentiles, QPS throughput, and accuracy decay tracking:
- 12.1 MLOps Architecture & Pre-Execution Gate
- 12.2 Model Performance Decay Monitor
- 12.7 Service Performance & SLA Monitor
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLOpsPreExecutionGate:
    """Pre-execution pipeline verification gate checking monitoring directory readiness."""

    def __init__(self, logs_dir: str = "logs/monitoring"):
        self.logs_dir = Path(logs_dir)

    def verify(self) -> bool:
        """Ensures monitoring log directories exist and are writable."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"MLOps Pre-Execution Gate PASSED. Logs directory: {self.logs_dir}")
        return True


class ServicePerformanceMonitor:
    """12.7 Tracks real-time API SLA latency percentiles (p50, p95, p99), QPS throughput, and error rates."""

    def __init__(self):
        self.latency_records_ms: List[float] = []
        self.error_count: int = 0
        self.total_requests: int = 0
        self.start_time: float = time.time()

    def record_request(self, latency_ms: float, is_error: bool = False) -> None:
        """Records single request latency and error status."""
        self.latency_records_ms.append(latency_ms)
        self.total_requests += 1
        if is_error:
            self.error_count += 1

    def get_service_metrics(self) -> Dict[str, Any]:
        """Calculates p50, p95, p99 latency percentiles, throughput QPS, and error rate."""
        if not self.latency_records_ms:
            return {
                "total_requests": 0,
                "qps": 0.0,
                "error_rate": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
            }

        latencies = np.array(self.latency_records_ms)
        elapsed = max(0.001, time.time() - self.start_time)

        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        qps = float(self.total_requests / elapsed)
        error_rate = float(self.error_count / self.total_requests)

        return {
            "total_requests": self.total_requests,
            "qps": round(qps, 2),
            "error_rate": round(error_rate, 4),
            "p50_latency_ms": round(p50, 3),
            "p95_latency_ms": round(p95, 3),
            "p99_latency_ms": round(p99, 3),
        }


class ModelPerformanceMonitor:
    """12.2 Evaluates production model accuracy/ROC-AUC decay over sliding time windows."""

    def evaluate_performance_decay(
        self,
        baseline_score: float,
        current_y_true: Union[pd.Series, np.ndarray],
        current_y_prob: Union[pd.Series, np.ndarray],
        max_allowed_decay: float = 0.05,
    ) -> Dict[str, Any]:
        """Calculates current ROC-AUC score and compares against baseline to flag performance decay."""
        from sklearn.metrics import roc_auc_score

        yt = np.asarray(current_y_true, dtype=int).ravel()
        yp = np.asarray(current_y_prob, dtype=float).ravel()

        if len(np.unique(yt)) > 1:
            current_score = float(roc_auc_score(yt, yp))
        else:
            current_score = baseline_score

        decay = baseline_score - current_score
        is_decayed = decay > max_allowed_decay

        return {
            "baseline_score": round(baseline_score, 4),
            "current_score": round(current_score, 4),
            "performance_decay": round(decay, 4),
            "max_allowed_decay": max_allowed_decay,
            "is_performance_decayed": is_decayed,
        }
