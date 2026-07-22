"""Pipeline script executing pre-validation, candidate model promotion, and monochromatic HUD report generation."""

from __future__ import annotations

import logging
import json
import shutil
import gc
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import mlflow

from src.evaluation.selection import CandidateModelSelector
from src.evaluation.comparison import ModelComparator

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
        "data/models/v1/baseline_lr.pkl",
        "data/models/v1/baseline_xgb.pkl",
        "data/models/v1/lightgbm_model.pkl",
        "data/models/v1/xgboost_model.pkl",
        "data/models/v1/catboost_model.pkl",
        "data/models/v1/ensemble_model.pkl",
        "reports/models/model_comparison_report.html"
    ]
    missing = [dep for dep in dependencies if not Path(dep).exists()]
    if missing:
        err = f"Pipeline Dependency Failure: Missing upstream outputs: {missing}"
        logger.error(err)
        raise FileNotFoundError(err)
    logger.info("Prior stage verification check successfully passed.")


def generate_selection_hud_report(champion_name: str, champion_metrics: dict, output_path: Path) -> None:
    """Renders a sleek, monochromatic dashboard report showing the promoted candidate model statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Candidate Model Selection HUD</title>
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
                <h1>Candidate Selection HUD</h1>
                <div style="font-size: 12px; color: var(--text-muted);">V1.0 // Production Promotion Verification</div>
            </div>
            <div class="sys-info">
                STATUS: PROMOTED<br>
                DATE: {now_str}<br>
                TARGET: validation_f2_score
            </div>
        </div>
        
        <div class="grid">
            <div class="panel">
                <div class="panel-title">Promoted Champion</div>
                <div class="stat-row">
                    <span class="stat-label">Model Name:</span>
                    <span class="stat-value">{champion_name}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Threshold:</span>
                    <span class="stat-value">{champion_metrics.get('threshold', 0.5):.2f}</span>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">Champion Validation Stats</div>
                <div class="stat-row">
                    <span class="stat-label">F2-score:</span>
                    <span class="stat-value">{champion_metrics.get('f2_score', 0.0):.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">F1-score:</span>
                    <span class="stat-value">{champion_metrics.get('f1_score', 0.0):.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">AUC:</span>
                    <span class="stat-value">{champion_metrics.get('auc', 0.0):.5f}</span>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html_content)
    logger.info("Candidate Selection HUD report saved: %s", output_path)


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
    
    # Load all models to build comparison matrix
    model_paths = {
        "Baseline_LR": "data/models/v1/baseline_lr.pkl",
        "Baseline_XGB": "data/models/v1/baseline_xgb.pkl",
        "LightGBM": "data/models/v1/lightgbm_model.pkl",
        "XGBoost": "data/models/v1/xgboost_model.pkl",
        "CatBoost": "data/models/v1/catboost_model.pkl",
        "Ensemble_Blended": "data/models/v1/ensemble_model.pkl"
    }
    
    models = {}
    for name, p in model_paths.items():
        with open(Path(p), "rb") as f:
            models[name] = pickle.load(f)
            
    comparator = ModelComparator()
    results = comparator.compare(models, X_val, y_val)
    
    selector = CandidateModelSelector(optimize_metric="f2_score")
    champion_name, champion_metrics = selector.select_best_model(results)
    
    # Promote champion
    champion_source_path = model_paths[champion_name]
    promoted_dest_path = Path("data/models/v1/candidate_model.pkl")
    promoted_dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(champion_source_path), promoted_dest_path)
    logger.info("Champion model %s serialized copy promoted successfully to: %s", champion_name, promoted_dest_path)
    
    out_report = Path("reports/models/candidate_selection_report.html")
    generate_selection_hud_report(champion_name, champion_metrics, out_report)
    
    logger.info("Logging Candidate Selection metrics to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="candidate_selection")
        started = True
        
    try:
        mlflow.log_params({
            "stage": "candidate_selection",
            "promoted_champion_name": champion_name,
            "promoted_champion_threshold": champion_metrics.get("threshold", 0.5)
        })
        mlflow.log_metrics({
            "champion_validation_auc": champion_metrics["auc"],
            "champion_validation_f1": champion_metrics["f1_score"],
            "champion_validation_f2": champion_metrics["f2_score"]
        })
        mlflow.log_artifact(str(out_report), artifact_path="candidate_selection")
    except Exception as e:
        logger.warning("MLflow logging raised exception: %s", e)
    finally:
        if started:
            mlflow.end_run()
            
    logger.info("Candidate Model Selection stage execution complete.")


if __name__ == "__main__":
    main()
