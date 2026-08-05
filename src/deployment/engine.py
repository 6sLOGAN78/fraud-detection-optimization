"""11.7 - 11.8 Real-Time and Batch Prediction Engines Module.

Provides low-latency single transaction scoring and chunked high-throughput batch inference:
- 11.7 Batch Prediction Engine
- 11.8 Real-Time Low-Latency Prediction Engine
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.deployment.inference import InferencePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeInferenceEngine:
    """11.8 Low-latency single transaction inference engine measuring sub-10ms scoring SLA."""

    def __init__(self, pipeline: InferencePipeline):
        self.pipeline = pipeline

    def score_transaction(self, transaction_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Scores a single transaction dictionary in real time with SLA latency logging."""
        start_time = time.perf_counter()

        df_single = pd.DataFrame([transaction_dict])
        preds, probs = self.pipeline.predict(df_single)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        fraud_prob = float(probs[0])
        is_fraud = bool(preds[0] == 1)

        return {
            "is_fraud": is_fraud,
            "fraud_probability": round(fraud_prob, 5),
            "decision_threshold": self.pipeline.decision_threshold,
            "latency_ms": round(latency_ms, 3),
            "status": "APPROVED" if not is_fraud else "FLAGGED_FOR_REVIEW",
        }


class BatchInferenceEngine:
    """11.7 High-throughput batch inference engine processing datasets in configurable chunks."""

    def __init__(self, pipeline: InferencePipeline, chunk_size: int = 5000):
        self.pipeline = pipeline
        self.chunk_size = chunk_size

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes input DataFrame in chunks to avoid memory bottlenecks."""
        start_time = time.time()
        logger.info(f"Starting batch prediction on {len(df)} records (Chunk size: {self.chunk_size})...")

        results_probs = []
        results_preds = []

        n_chunks = int(np.ceil(len(df) / self.chunk_size))
        for i in range(n_chunks):
            chunk = df.iloc[i * self.chunk_size : (i + 1) * self.chunk_size]
            preds, probs = self.pipeline.predict(chunk)
            results_probs.extend(probs)
            results_preds.extend(preds)

        df_out = df.copy()
        df_out["fraud_probability"] = results_probs
        df_out["is_fraud_prediction"] = results_preds

        elapsed = time.time() - start_time
        logger.info(f"Batch prediction completed in {elapsed:.3f}s ({len(df)/elapsed:.1f} records/sec).")
        return df_out
