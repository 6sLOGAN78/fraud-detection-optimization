"""11.5 - 11.6 & 11.9 FastAPI Service, Endpoints, Pydantic Schemas, and API Versioning.

Provides FastAPI REST web service endpoints:
- GET /health (Health check)
- POST /v1/predict (Real-time single transaction scoring)
- POST /v1/predict_batch (Batch transaction scoring)
- GET /v1/model_info (Model metadata and features)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from src.deployment.inference import InferencePipeline, ModelPackager, ModelSerializer
from src.deployment.engine import RealTimeInferenceEngine, BatchInferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Schemas
class TransactionRequest(BaseModel):
    TransactionAmt: float = Field(..., example=150.0, description="Transaction amount in USD")
    ProductCD: Optional[str] = Field("W", example="W")
    card1: Optional[int] = Field(1000, example=13926)
    card2: Optional[float] = Field(500.0, example=150.0)
    extra_features: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional feature key-value pairs")


class PredictResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    decision_threshold: float
    latency_ms: float
    status: str
    version: str = "v1"


class BatchTransactionRequest(BaseModel):
    transactions: List[Dict[str, Any]] = Field(..., description="List of transaction dictionaries")


class BatchPredictResponse(BaseModel):
    total_records: int
    predictions: List[PredictResponse]
    total_latency_ms: float
    version: str = "v1"


# Initialize FastAPI app
app = FastAPI(
    title="IEEE-CIS Fraud Detection Production API",
    description="High-throughput real-time and batch fraud probability scoring API.",
    version="1.0.0",
)

# Mock baseline pipeline for service testing
_DEFAULT_FEATURES = ["TransactionAmt", "card1", "card2"]


class _DummyModel:
    def predict_proba(self, X: pd.DataFrame):
        amt = X.get("TransactionAmt", pd.Series([0] * len(X))).values
        probs = 1.0 / (1.0 + np.exp(- (amt - 100.0) / 50.0))
        return np.column_stack([1 - probs, probs])


import numpy as np
global_pipeline = InferencePipeline(model=_DummyModel(), feature_names=_DEFAULT_FEATURES)
realtime_engine = RealTimeInferenceEngine(pipeline=global_pipeline)
batch_engine = BatchInferenceEngine(pipeline=global_pipeline)


@app.get("/health")
def health_check():
    """Health check endpoint for Kubernetes liveness/readiness probes."""
    return {"status": "HEALTHY", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


@app.get("/v1/model_info")
def get_model_info():
    """Returns metadata for the currently loaded model version."""
    return {
        "version": "v1",
        "features": global_pipeline.feature_names,
        "decision_threshold": global_pipeline.decision_threshold,
    }


@app.post("/v1/predict", response_model=PredictResponse)
def predict_single(request: TransactionRequest):
    """v1 Real-Time Single Transaction Fraud Scoring Endpoint."""
    start_time = time.perf_counter()
    try:
        tx_dict = {"TransactionAmt": request.TransactionAmt, "card1": request.card1, "card2": request.card2}
        tx_dict.update(request.extra_features)

        result = realtime_engine.score_transaction(tx_dict)
        return PredictResponse(
            is_fraud=result["is_fraud"],
            fraud_probability=result["fraud_probability"],
            decision_threshold=result["decision_threshold"],
            latency_ms=result["latency_ms"],
            status=result["status"],
            version="v1",
        )
    except Exception as e:
        logger.error(f"Error processing transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/predict_batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchTransactionRequest):
    """v1 Batch Transaction Fraud Scoring Endpoint."""
    start_time = time.perf_counter()
    try:
        results = []
        for tx in request.transactions:
            res = realtime_engine.score_transaction(tx)
            results.append(
                PredictResponse(
                    is_fraud=res["is_fraud"],
                    fraud_probability=res["fraud_probability"],
                    decision_threshold=res["decision_threshold"],
                    latency_ms=res["latency_ms"],
                    status=res["status"],
                    version="v1",
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return BatchPredictResponse(
            total_records=len(results),
            predictions=results,
            total_latency_ms=round(elapsed_ms, 2),
            version="v1",
        )
    except Exception as e:
        logger.error(f"Error processing batch transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
