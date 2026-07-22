"""Pipeline script executing pre-validation, probability calibration calculations, and monochromatic HUD report generation."""

from __future__ import annotations

import logging
import json
import gc
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import mlflow

from src.evaluation.calibration import ProbabilityCalibrator
from src.models.ensemble import EnsembleClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pre_execution_verification_gate() -> None:
    """Verifies that all preceding pipeline outputs exist before running training."""
    logger.info("Executing Pre-Execution Pipeline Verification Gate...")
    dependencies = [
        "data/feature_store_engineered/v1/train_selected_features.parquet",
        "data/feature_store_engineered/v1/test_selected_features.parquet",
        "data/feature_store_engineered/v1/feature_registry.json",
        "data/interim/train_merged.parquet",
        "data/models/v1/ensemble_model.pkl"
    ]
    missing = [dep for dep in dependencies if not Path(dep).exists()]
    if missing:
        err = f"Pipeline Dependency Failure: Missing upstream outputs: {missing}"
        logger.error(err)
        raise FileNotFoundError(err)
    logger.info("Prior stage verification check successfully passed.")


def generate_calibration_hud_report(cal_summary: dict, output_path: Path) -> None:
    """Renders a sleek, monochromatic dashboard report for Probability Calibration results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    o_m = cal_summary.get("calibration_metrics", {})
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Probability Calibration HUD // Performance Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #050505;
            --panel-bg: rgba(26, 26, 26, 0.45);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-color: #f5f5f5;
            --text-muted: #888888;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 40px;
            display: flex;
            justify-content: center;
        }}
        .hud-container {{
            width: 100%;
            max-width: 800px;
            border: 1px solid var(--border-color);
            background: var(--panel-bg);
            backdrop-filter: blur(16px) saturate(120%);
            padding: 30px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 24px;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin: 0;
        }}
        .sys-info {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 11px;
            color: var(--text-muted);
            text-align: right;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .panel {{
            border: 1px solid var(--border-color);
            background: rgba(255, 255, 255, 0.01);
            padding: 20px;
        }}
        .panel-title {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 14px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }}
        .stat-row:last-child {{
            border-bottom: none;
        }}
        .stat-label {{
            color: var(--text-muted);
        }}
        .stat-value {{
            font-family: 'Share Tech Mono', monospace;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="hud-container">
        <div class="header">
            <div>
                <h1>Probability Calibration HUD</h1>
                <div style="font-size: 12px; color: var(--text-muted);">V1.0 // Brier Score Loss Evaluation</div>
            </div>
            <div class="sys-info">
                STATUS: ACTIVE<br>
                DATE: {now_str}<br>
                STRATEGY: Binning Reliability
            </div>
        </div>
        
        <div class="grid">
            <div class="panel">
                <div class="panel-title">Calibration Scores</div>
                <div class="stat-row">
                    <span class="stat-label">Brier Score Loss:</span>
                    <span class="stat-value">{o_m.get('brier_score_loss', 0.0):.5f}</span>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">Runtime Performance</div>
                <div class="stat-row">
                    <span class="stat-label">Analysis Duration:</span>
                    <span class="stat-value">{o_m.get('elapsed_time_seconds', 0.0):.4f}s</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Memory Delta:</span>
                    <span class="stat-value">{o_m.get('memory_delta_mb', 0.0):.2f}MB</span>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html_content)
    logger.info("Probability Calibration HUD report saved: %s", output_path)


def main() -> None:
    run_pre_execution_verification_gate()
    
    train_in = Path("data/feature_store_engineered/v1/train_selected_features.parquet")
    target_in = Path("data/interim/train_merged.parquet")
    registry_in = Path("data/feature_store_engineered/v1/feature_registry.json")
    
    with open(registry_in, "r") as f:
        registry = json.load(f)
    selected_cols = list(registry["features"].keys())
    
    logger.info("Loading pruned feature matrix...")
    df_features = pd.read_parquet(train_in, columns=["TransactionID"] + selected_cols)
    df_raw = pd.read_parquet(target_in, columns=["TransactionID", "isFraud"])
    
    df_merged = pd.merge(df_features, df_raw, on="TransactionID", how="inner")
    
    del df_features
    del df_raw
    gc.collect()
    
    X = df_merged[selected_cols]
    y = df_merged["isFraud"]
    
    split_idx = int(len(X) * 0.8)
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    # Load EnsembleClassifier
    with open(Path("data/models/v1/ensemble_model.pkl"), "rb") as f:
        ensemble = pickle.load(f)
        
    logger.info("Generating predictions on validation split...")
    y_probs = ensemble.predict_proba(X_val)[:, 1]
    
    calibrator = ProbabilityCalibrator(n_bins=10)
    metrics = calibrator.analyze(y_val, y_probs)
    
    cal_summary = {
        "calibration_metrics": metrics
    }
    
    out_report = Path("reports/models/probability_calibration_report.html")
    generate_calibration_hud_report(cal_summary, out_report)
    
    logger.info("Logging Probability Calibration parameters and metrics to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="probability_calibration")
        started = True
        
    try:
        mlflow.log_params({
            "stage": "probability_calibration",
            "n_bins": 10
        })
        mlflow.log_metrics({
            "brier_score_loss": metrics["brier_score_loss"]
        })
        mlflow.log_artifact(str(out_report), artifact_path="probability_calibration")
    except Exception as e:
        logger.warning("MLflow logging raised exception: %s", e)
    finally:
        if started:
            mlflow.end_run()
            
    logger.info("Probability Calibration stage execution complete.")


if __name__ == "__main__":
    main()
