"""Pipeline script executing pre-validation, side-by-side model comparison, and monochromatic HUD report generation."""

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
        "data/models/v1/ensemble_model.pkl"
    ]
    missing = [dep for dep in dependencies if not Path(dep).exists()]
    if missing:
        err = f"Pipeline Dependency Failure: Missing upstream outputs: {missing}"
        logger.error(err)
        raise FileNotFoundError(err)
    logger.info("Prior stage verification check successfully passed.")


def generate_comparison_hud_report(comp_summary: dict, output_path: Path) -> None:
    """Renders a sleek, monochromatic dashboard report comparing all model architectures side-by-side."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    comps = comp_summary.get("comparisons", {})
    
    table_rows = ""
    for name, metrics in comps.items():
        table_rows += f"""
        <tr>
            <td style="text-align: left; font-weight: 600; padding: 12px 10px;">{name}</td>
            <td style="font-family: 'Share Tech Mono', monospace; padding: 12px 10px;">{metrics.get('auc', 0.0):.5f}</td>
            <td style="font-family: 'Share Tech Mono', monospace; padding: 12px 10px;">{metrics.get('accuracy', 0.0):.5f}</td>
            <td style="font-family: 'Share Tech Mono', monospace; padding: 12px 10px;">{metrics.get('precision', 0.0):.5f}</td>
            <td style="font-family: 'Share Tech Mono', monospace; padding: 12px 10px;">{metrics.get('recall', 0.0):.5f}</td>
            <td style="font-family: 'Share Tech Mono', monospace; padding: 12px 10px;">{metrics.get('f1_score', 0.0):.5f}</td>
            <td style="font-family: 'Share Tech Mono', monospace; padding: 12px 10px;">{metrics.get('f2_score', 0.0):.5f}</td>
            <td style="font-family: 'Share Tech Mono', monospace; padding: 12px 10px;">{metrics.get('threshold', 0.5):.2f}</td>
        </tr>
        """
        
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model Comparison Framework HUD</title>
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
            max-width: 900px;
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
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 13px;
        }}
        th {{
            font-family: 'Share Tech Mono', monospace;
            font-weight: normal;
            color: var(--text-muted);
            text-transform: uppercase;
            border-bottom: 2px solid var(--border-color);
            padding: 10px;
            text-align: right;
        }}
        th:first-child {{
            text-align: left;
        }}
        td {{
            border-bottom: 1px solid var(--border-color);
            text-align: right;
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
    </style>
</head>
<body>
    <div class="hud-container">
        <div class="header">
            <div>
                <h1>Model Suite Evaluation comparison</h1>
                <div style="font-size: 12px; color: var(--text-muted);">V1.0 // Holistic Validation Benchmarking</div>
            </div>
            <div class="sys-info">
                STATUS: COMPETE<br>
                DATE: {now_str}<br>
                METRICS: Multi-Dimensional
            </div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th style="text-align: left;">Architecture</th>
                    <th>AUC</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-score</th>
                    <th>F2-score</th>
                    <th>Threshold</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html_content)
    logger.info("Model Comparison Framework HUD report saved: %s", output_path)


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
    
    # Load all models
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
    
    out_report = Path("reports/models/model_comparison_report.html")
    generate_comparison_hud_report(results, out_report)
    
    logger.info("Logging Model Comparison statistics to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="model_comparison")
        started = True
        
    try:
        mlflow.log_params({
            "stage": "model_comparison"
        })
        for name, metrics in results["comparisons"].items():
            mlflow.log_metrics({
                f"{name}_auc": metrics["auc"],
                f"{name}_f1": metrics["f1_score"],
                f"{name}_f2": metrics["f2_score"]
            })
        mlflow.log_artifact(str(out_report), artifact_path="model_comparison")
    except Exception as e:
        logger.warning("MLflow logging raised exception: %s", e)
    finally:
        if started:
            mlflow.end_run()
            
    logger.info("Model Comparison stage execution complete.")


if __name__ == "__main__":
    main()
