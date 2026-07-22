"""Pipeline script executing pre-validation, Ensemble Model evaluation, validation threshold tuning, and monochromatic HUD report generation."""

from __future__ import annotations

import logging
import json
import gc
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import mlflow

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
        "data/models/v1/lightgbm_model.pkl",
        "data/models/v1/xgboost_model.pkl",
        "data/models/v1/catboost_model.pkl"
    ]
    missing = [dep for dep in dependencies if not Path(dep).exists()]
    if missing:
        err = f"Pipeline Dependency Failure: Missing upstream outputs: {missing}"
        logger.error(err)
        raise FileNotFoundError(err)
    logger.info("Prior stage verification check successfully passed.")


def generate_ensemble_hud_report(ens_summary: dict, output_path: Path) -> None:
    """Renders a sleek, monochromatic dashboard report for the Ensemble model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fit_m = ens_summary.get("fit_metrics", {})
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Ensemble Classifier HUD // Performance Report</title>
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
                <h1>Ensemble Blended HUD</h1>
                <div style="font-size: 12px; color: var(--text-muted);">V1.0 // LGBM + XGB + CatBoost Blending</div>
            </div>
            <div class="sys-info">
                STATUS: ACTIVE<br>
                DATE: {now_str}<br>
                ENGINE: Blended Ensemble
            </div>
        </div>
        
        <div class="grid">
            <div class="panel">
                <div class="panel-title">Model Performance</div>
                <div class="stat-row">
                    <span class="stat-label">Train AUC:</span>
                    <span class="stat-value">{ens_summary['train_auc']:.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Validation AUC:</span>
                    <span class="stat-value">{ens_summary['val_auc']:.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Tuned Threshold:</span>
                    <span class="stat-value">{ens_summary['best_threshold']:.2f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Validation F1-score:</span>
                    <span class="stat-value">{ens_summary['best_f1']:.5f}</span>
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
    logger.info("Ensemble monochromatic HUD report saved: %s", output_path)


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
        
    logger.info("Initializing and blending Ensemble Classifier...")
    model = EnsembleClassifier(weights=[0.4, 0.3, 0.3])
    model.fit(X_train_fit, y_train_fit)
    
    # Validation threshold tuning
    train_probs = model.predict_proba(X_train_fit)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_thresh = 0.5
    best_f1 = -1.0
    
    for t in thresholds:
        f1 = f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
        logger.info("Threshold: %.1f, F1-score: %.5f", t, f1)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    logger.info("Best threshold identified: %.1f with F1-score of %.5f", best_thresh, best_f1)
    model.threshold = best_thresh
    
    train_auc = roc_auc_score(y_train_fit, train_probs) if len(np.unique(y_train_fit)) > 1 else 0.5
    val_auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.5
    
    ens_summary = {
        "train_auc": train_auc,
        "val_auc": val_auc,
        "best_threshold": best_thresh,
        "best_f1": best_f1,
        "fit_metrics": model.fit_metrics_
    }
    
    out_dir = Path("data/models/v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_model = out_dir / "ensemble_model.pkl"
    out_report = Path("reports/models/ensemble_report.html")
    
    with open(out_model, "wb") as f:
        pickle.dump(model, f)
        
    generate_ensemble_hud_report(ens_summary, out_report)
    
    logger.info("Logging Ensemble metrics and artifacts to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="ensemble_training")
        started = True
        
    try:
        mlflow.log_params({
            "stage": "ensemble_model",
            "weights": str(model.weights),
            "best_threshold": best_thresh
        })
        mlflow.log_metrics({
            "ens_train_auc": train_auc,
            "ens_val_auc": val_auc,
            "ens_val_f1": best_f1
        })
        mlflow.log_artifact(str(out_model), artifact_path="ensemble_model")
        mlflow.log_artifact(str(out_report), artifact_path="ensemble_model")
    except Exception as e:
        logger.warning("MLflow logging raised exception: %s", e)
    finally:
        if started:
            mlflow.end_run()
            
    logger.info("Ensemble evaluation and threshold tuning pipeline stage complete.")


if __name__ == "__main__":
    main()
