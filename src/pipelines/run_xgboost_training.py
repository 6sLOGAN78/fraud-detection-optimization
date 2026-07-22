"""Pipeline script executing pre-validation, XGBoost Model fit, evaluation, and monochromatic HUD report generation."""

from __future__ import annotations

import logging
import json
import gc
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import mlflow

from src.models.xgboost_model import XGBoostClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pre_execution_verification_gate() -> None:
    """Verifies that all preceding pipeline outputs exist before running training."""
    logger.info("Executing Pre-Execution Pipeline Verification Gate...")
    dependencies = [
        "data/feature_store_engineered/v1/train_selected_features.parquet",
        "data/feature_store_engineered/v1/test_selected_features.parquet",
        "data/feature_store_engineered/v1/feature_registry.json"
    ]
    missing = [dep for dep in dependencies if not Path(dep).exists()]
    if missing:
        err = f"Pipeline Dependency Failure: Missing upstream outputs: {missing}"
        logger.error(err)
        raise FileNotFoundError(err)
    logger.info("Prior stage verification check successfully passed.")


def generate_xgb_hud_report(xgb_summary: dict, output_path: Path) -> None:
    """Renders a sleek, monochromatic dashboard report for the XGBoost model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fit_m = xgb_summary.get("fit_metrics", {})
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>XGBoost Classifier HUD // Performance Report</title>
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
                <h1>XGBoost Classifier HUD</h1>
                <div style="font-size: 12px; color: var(--text-muted);">V1.0 // Optimized Scalable Booster</div>
            </div>
            <div class="sys-info">
                STATUS: ACTIVE<br>
                DATE: {now_str}<br>
                ENGINE: XGBoost
            </div>
        </div>
        
        <div class="grid">
            <div class="panel">
                <div class="panel-title">Model Performance</div>
                <div class="stat-row">
                    <span class="stat-label">Train AUC:</span>
                    <span class="stat-value">{xgb_summary['train_auc']:.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Validation AUC:</span>
                    <span class="stat-value">{xgb_summary['val_auc']:.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Train Accuracy:</span>
                    <span class="stat-value">{xgb_summary['train_accuracy']:.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Validation Accuracy:</span>
                    <span class="stat-value">{xgb_summary['val_accuracy']:.5f}</span>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">System Metrics</div>
                <div class="stat-row">
                    <span class="stat-label">Fit Duration:</span>
                    <span class="stat-value">{fit_m.get('fit_time_seconds', 0.0):.4f}s</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Memory Delta:</span>
                    <span class="stat-value">{fit_m.get('memory_delta_mb', 0.0):.2f}MB</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Sample Count:</span>
                    <span class="stat-value">{fit_m.get('samples_count', 0)} rows</span>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html_content)
    logger.info("XGBoost monochromatic HUD report saved: %s", output_path)


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
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Downsample train set dynamically to prevent OOM
    if len(X_train) > 50000:
        rng = np.random.RandomState(42)
        fit_idx = rng.choice(X_train.index, size=50000, replace=False)
        X_train_fit = X_train.loc[fit_idx]
        y_train_fit = y_train.loc[fit_idx]
    else:
        X_train_fit = X_train
        y_train_fit = y_train
        
    logger.info("Training XGBoost Classifier...")
    model = XGBoostClassifier(threshold=0.05, random_state=42, n_jobs=-1)
    model.fit(X_train_fit, y_train_fit)
    
    train_pred = model.predict_proba(X_train_fit)[:, 1]
    val_pred = model.predict_proba(X_val)[:, 1]
    
    train_auc = roc_auc_score(y_train_fit, train_pred) if len(np.unique(y_train_fit)) > 1 else 0.5
    val_auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) > 1 else 0.5
    
    train_acc = accuracy_score(y_train_fit, (train_pred >= 0.5).astype(int))
    val_acc = accuracy_score(y_val, (val_pred >= 0.5).astype(int))
    
    xgb_summary = {
        "train_auc": train_auc,
        "val_auc": val_auc,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "fit_metrics": model.fit_metrics_
    }
    
    out_dir = Path("data/models/v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_model = out_dir / "xgboost_model.pkl"
    out_report = Path("reports/models/xgboost_report.html")
    
    with open(out_model, "wb") as f:
        pickle.dump(model, f)
        
    generate_xgb_hud_report(xgb_summary, out_report)
    
    logger.info("Logging XGBoost metrics and artifacts to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="xgboost_training")
        started = True
        
    try:
        mlflow.log_params({
            "stage": "xgboost_model",
            "estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        })
        mlflow.log_metrics({
            "xgb_train_auc": train_auc,
            "xgb_val_auc": val_auc,
            "xgb_train_accuracy": train_acc,
            "xgb_val_accuracy": val_acc
        })
        mlflow.log_artifact(str(out_model), artifact_path="xgboost_model")
        mlflow.log_artifact(str(out_report), artifact_path="xgboost_model")
    except Exception as e:
        logger.warning("MLflow logging raised exception: %s", e)
    finally:
        if started:
            mlflow.end_run()
            
    logger.info("XGBoost training pipeline stage execution complete.")


if __name__ == "__main__":
    main()
