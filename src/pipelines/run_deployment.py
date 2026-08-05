"""Pipeline script to execute Part 11 — Production Model Deployment & Validation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.deployment import (
    BatchInferenceEngine,
    DeploymentPreExecutionGate,
    DeploymentValidator,
    InferencePipeline,
    ModelPackager,
    ModelSerializer,
    RealTimeInferenceEngine,
    RollbackManager,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 11 Production Deployment Pipeline")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples to evaluate for batch engine")
    args = parser.parse_args()

    train_path = Path("data/interim/train_cleaned.parquet")
    if not train_path.exists():
        train_path = Path("data/interim/train_merged.parquet")

    gate = DeploymentPreExecutionGate(required_artifacts=[str(train_path)])
    gate.verify()

    logger.info(f"Loading data from {train_path}...")
    df = pd.read_parquet(train_path)

    if "isFraud" not in df.columns:
        logger.error("Target column 'isFraud' not found.")
        sys.exit(1)

    if len(df) > args.n_samples:
        df = df.sample(n=args.n_samples, random_state=42).reset_index(drop=True)

    y = df["isFraud"].values
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    X_num = X[num_cols].fillna(0)

    # Train model for packaging
    logger.info("Training production candidate model...")
    clf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    clf.fit(X_num, y)

    # Package Model
    packager = ModelPackager()
    bundle_path = packager.create_bundle(clf, feature_names=num_cols, version="v1")

    # Load Model Pipeline
    loaded_model = ModelSerializer.load_artifact(bundle_path / "model.joblib")
    pipeline = InferencePipeline(model=loaded_model, feature_names=num_cols, decision_threshold=0.5)

    # Pre-flight Smoke Test Validation
    validator = DeploymentValidator()
    sample_tx = X_num.iloc[0].to_dict()
    val_res = validator.validate_pipeline(pipeline, sample_tx)
    logger.info(f"Deployment Validation Smoke Test: {val_res['status']}")

    # Real-Time Scoring Engine
    rt_engine = RealTimeInferenceEngine(pipeline=pipeline)
    tx_res = rt_engine.score_transaction(sample_tx)
    logger.info(f"Real-Time Scoring Latency: {tx_res['latency_ms']} ms | Probability: {tx_res['fraud_probability']}")

    # Batch Scoring Engine
    batch_engine = BatchInferenceEngine(pipeline=pipeline, chunk_size=500)
    df_batch_out = batch_engine.predict_dataframe(X_num)

    out_file = Path("data/processed/batch_predictions_v1.parquet")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_batch_out.to_parquet(out_file)

    logger.info(f"Part 11 Production Deployment Pipeline completed successfully. Output saved to {out_file}")


if __name__ == "__main__":
    main()
