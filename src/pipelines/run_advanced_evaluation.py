"""Pipeline script to execute Part 8 — Advanced Model Evaluation Framework."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.evaluation import (
    BusinessKPIEvaluator,
    CalibrationAnalysisEngine,
    ConfusionMatrixEngine,
    ErrorAnalysisEngine,
    EvaluationFrameworkDesign,
    EvaluationPreExecutionGate,
    F1ScoreEngine,
    KSStatisticEngine,
    LiftGainEngine,
    MCCEngine,
    PRAUCEngine,
    PrecisionEngine,
    ROCAUCEngine,
    RecallEngine,
    RobustnessEvaluationEngine,
    ThresholdAnalysisEngine,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 8 Advanced Evaluation Pipeline")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of samples to evaluate")
    args = parser.parse_args()

    train_path = Path("data/interim/train_cleaned.parquet")
    if not train_path.exists():
        train_path = Path("data/interim/train_merged.parquet")

    gate = EvaluationPreExecutionGate(required_files=[str(train_path)])
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
    num_cols = X.select_dtypes(include=[np.number]).columns
    X_num = X[num_cols].fillna(0)

    # Train a fast model for evaluation testing
    logger.info("Training evaluator model...")
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(X_num, y)
    y_prob = clf.predict_proba(X_num)[:, 1]

    val_design = EvaluationFrameworkDesign()
    y_true_clean, y_prob_clean = val_design.validate_inputs(y, y_prob)

    logger.info("Executing Metric Evaluations (ROC-AUC, PR-AUC, Precision, Recall, F1, MCC, KS)...")
    roc = ROCAUCEngine().calculate(y_true_clean, y_prob_clean)
    pr = PRAUCEngine().calculate(y_true_clean, y_prob_clean)
    prec = PrecisionEngine().calculate(y_true_clean, y_prob_clean)
    rec = RecallEngine().calculate(y_true_clean, y_prob_clean)
    f1 = F1ScoreEngine().calculate(y_true_clean, y_prob_clean)
    mcc = MCCEngine().calculate(y_true_clean, y_prob_clean)
    ks = KSStatisticEngine().calculate(y_true_clean, y_prob_clean)

    logger.info("Executing Diagnostic Evaluations (Calibration, Lift/Gain, Threshold Sweep, Confusion Matrix)...")
    cal = CalibrationAnalysisEngine().calculate(y_true_clean, y_prob_clean)
    lift = LiftGainEngine().calculate(y_true_clean, y_prob_clean)
    thresh = ThresholdAnalysisEngine().sweep(y_true_clean, y_prob_clean)
    cm = ConfusionMatrixEngine().calculate(y_true_clean, y_prob_clean)

    logger.info("Executing Error Analysis, Robustness & Business KPI Evaluation...")
    error_analysis = ErrorAnalysisEngine().analyze(X_num, y_true_clean, y_prob_clean)
    robustness = RobustnessEvaluationEngine().evaluate_perturbation(clf, X_num, y_true_clean)

    amounts = df["TransactionAmt"].values if "TransactionAmt" in df.columns else np.full(len(y_true_clean), 100.0)
    biz_kpi = BusinessKPIEvaluator().calculate(y_true_clean, y_prob_clean, amounts)

    evaluation_report = {
        "roc_auc": roc["roc_auc"],
        "pr_auc": pr["pr_auc"],
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "mcc": mcc,
        "ks_statistic": ks["ks_statistic"],
        "expected_calibration_error": cal["expected_calibration_error"],
        "brier_score": cal["brier_score"],
        "best_fbeta_threshold": thresh["best_fbeta_threshold"],
        "false_positive_rate": cm["false_positive_rate"],
        "robustness_score": robustness["robustness_score"],
        "net_financial_savings": biz_kpi["net_financial_savings"],
    }

    out_dir = Path("reports/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "advanced_evaluation_report.json"

    with open(report_path, "w") as f:
        json.dump(evaluation_report, f, indent=2)

    logger.info(f"Part 8 Advanced Model Evaluation completed successfully. Report saved to {report_path}")


if __name__ == "__main__":
    main()
