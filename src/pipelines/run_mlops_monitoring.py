"""Pipeline script to execute Part 12 — Production MLOps Monitoring, Alerting & Retraining."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.monitoring import (
    AlertingEngine,
    AutomatedRetrainingPipeline,
    ChampionChallengerLifecycleManager,
    ConceptDriftMonitor,
    DataDriftMonitor,
    FeatureHealthMonitor,
    MLOpsPreExecutionGate,
    ModelPerformanceMonitor,
    PredictionDistributionMonitor,
    ServicePerformanceMonitor,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 12 MLOps Monitoring & Retraining Pipeline")
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of samples to evaluate")
    args = parser.parse_args()

    gate = MLOpsPreExecutionGate()
    gate.verify()

    train_path = Path("data/interim/train_cleaned.parquet")
    if not train_path.exists():
        train_path = Path("data/interim/train_merged.parquet")

    logger.info(f"Loading reference & current datasets from {train_path}...")
    df = pd.read_parquet(train_path)

    if len(df) > args.n_samples * 2:
        ref_df = df.iloc[: args.n_samples].reset_index(drop=True)
        cur_df = df.iloc[args.n_samples : args.n_samples * 2].reset_index(drop=True)
    else:
        ref_df = df.sample(frac=0.5, random_state=42).reset_index(drop=True)
        cur_df = df.drop(ref_df.index, errors="ignore").reset_index(drop=True)

    y_ref = ref_df["isFraud"].values if "isFraud" in ref_df.columns else np.zeros(len(ref_df))
    y_cur = cur_df["isFraud"].values if "isFraud" in cur_df.columns else np.zeros(len(cur_df))

    X_ref_num = ref_df.select_dtypes(include=[np.number]).fillna(0)
    X_cur_num = cur_df.select_dtypes(include=[np.number]).fillna(0)

    # Train Champion & Challenger models
    logger.info("Training Champion & Challenger models for monitoring...")
    champ = RandomForestClassifier(n_estimators=15, max_depth=5, random_state=42)
    champ.fit(X_ref_num, y_ref)

    chall = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42)
    chall.fit(X_ref_num, y_ref)

    ref_probs = champ.predict_proba(X_ref_num)[:, 1]
    cur_probs = champ.predict_proba(X_cur_num)[:, 1]

    # Service SLA performance monitoring
    service_mon = ServicePerformanceMonitor()
    for lat in [12.5, 15.0, 8.2, 45.0, 9.1, 110.0]:
        service_mon.record_request(lat)
    service_metrics = service_mon.get_service_metrics()

    # Data & Prediction Drift
    data_drift_mon = DataDriftMonitor()
    drift_res = data_drift_mon.evaluate_drift(X_ref_num, X_cur_num, psi_threshold=0.2)

    concept_drift_mon = ConceptDriftMonitor()
    concept_res = concept_drift_mon.evaluate_target_drift(y_ref, y_cur)

    pred_drift_mon = PredictionDistributionMonitor()
    pred_res = pred_drift_mon.evaluate_prediction_drift(ref_probs, cur_probs)

    feature_health_mon = FeatureHealthMonitor()
    health_res = feature_health_mon.evaluate_feature_health(X_cur_num)

    perf_mon = ModelPerformanceMonitor()
    perf_res = perf_mon.evaluate_performance_decay(baseline_score=0.85, current_y_true=y_cur, current_y_prob=cur_probs)

    # Alerting Engine
    alert_engine = AlertingEngine()
    alerts = alert_engine.evaluate_drift_and_alert(drift_res)

    # Champion vs Challenger Shadow Testing & Promotion
    lifecycle_mgr = ChampionChallengerLifecycleManager(champion_model=champ, challenger_model=chall)
    promo_res = lifecycle_mgr.evaluate_shadow_promotion(X_cur_num, y_cur)

    # Automated Retraining Pipeline
    retrain_pipe = AutomatedRetrainingPipeline(alerting_engine=alert_engine)
    triggered = retrain_pipe.should_trigger_retraining(drift_res, perf_res)

    summary_report = {
        "service_metrics": service_metrics,
        "data_drift": drift_res,
        "concept_drift": concept_res,
        "prediction_drift": pred_res,
        "feature_health": health_res,
        "performance_decay": perf_res,
        "champion_promotion": promo_res,
        "retraining_triggered": triggered,
    }

    out_file = Path("reports/monitoring/mlops_monitoring_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary_report, f, indent=2)

    logger.info(f"Part 12 MLOps Monitoring Pipeline completed successfully. Report saved to {out_file}")


if __name__ == "__main__":
    main()
