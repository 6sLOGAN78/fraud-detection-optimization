"""Pipeline orchestration script executing pre-validation, baseline Model fit, evaluation, and comparative HUD reporting."""

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

from src.models.baseline import LogisticRegressionBaseline, XGBoostBaseline

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
    
    missing = []
    for dep in dependencies:
        p = Path(dep)
        if not p.exists():
            missing.append(dep)
            
    if missing:
        err = f"Pipeline Dependency Failure: Missing upstream outputs: {missing}"
        logger.error(err)
        raise FileNotFoundError(err)
        
    logger.info("Prior stage verification check successfully passed.")


def generate_baseline_hud_report(lr_summary: dict, xgb_summary: dict, output_path: Path) -> None:
    """Renders a sleek, monochromatic dashboard report comparing Logistic Regression and XGBoost baselines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lr_fit = lr_summary.get("fit_metrics", {})
    xgb_fit = xgb_summary.get("fit_metrics", {})
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Baseline Models HUD // Performance Comparison</title>
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
        
        .comparison-layout {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .model-panel {{
            border: 1px solid var(--border-color);
            background: rgba(255, 255, 255, 0.01);
            padding: 20px;
        }}
        
        .model-title {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 16px;
            font-weight: 600;
            text-transform: uppercase;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
            margin-bottom: 15px;
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
        
        .section-label {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-muted);
            margin-bottom: 15px;
            border-left: 2px solid white;
            padding-left: 8px;
        }}
    </style>
</head>
<body>
    <div class="hud-container">
        <div class="header">
            <div>
                <h1>Baseline Models HUD</h1>
                <div style="font-size: 12px; color: var(--text-muted);">V1.0 // Comparison Matrix</div>
            </div>
            <div class="sys-info">
                STATUS: COMPLETED<br>
                RUN DATE: {now_str}<br>
                TARGET: isFraud
            </div>
        </div>
        
        <div class="section-label">Baseline Classifiers Benchmarks</div>
        <div class="comparison-layout">
            <!-- Logistic Regression Panel -->
            <div class="model-panel">
                <div class="model-title">LogisticRegression (Linear)</div>
                <div class="stat-row">
                    <span class="stat-label">Train AUC:</span>
                    <span class="stat-value">{lr_summary['train_auc']:.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Validation AUC:</span>
                    <span class="stat-value">{lr_summary['val_auc']:.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Train Accuracy:</span>
                    <span class="stat-value">{lr_summary['train_accuracy']:.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Validation Accuracy:</span>
                    <span class="stat-value">{lr_summary['val_accuracy']:.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Fit Duration:</span>
                    <span class="stat-value">{lr_fit.get('fit_time_seconds', 0.0):.4f}s</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Memory Rise:</span>
                    <span class="stat-value">{lr_fit.get('memory_delta_mb', 0.0):.2f}MB</span>
                </div>
            </div>
            
            <!-- XGBoost Panel -->
            <div class="model-panel">
                <div class="model-title">XGBoost (Tree Booster)</div>
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
                <div class="stat-row">
                    <span class="stat-label">Fit Duration:</span>
                    <span class="stat-value">{xgb_fit.get('fit_time_seconds', 0.0):.4f}s</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Memory Rise:</span>
                    <span class="stat-value">{xgb_fit.get('memory_delta_mb', 0.0):.2f}MB</span>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html_content)
    logger.info("Monochromatic baseline comparison HTML HUD dashboard saved: %s", output_path)


def main() -> None:
    # 1. Verification Gate
    run_pre_execution_verification_gate()
    
    # 2. Files paths
    train_in = Path("data/feature_store_engineered/v1/train_selected_features.parquet")
    target_in = Path("data/interim/train_merged.parquet")
    registry_in = Path("data/feature_store_engineered/v1/feature_registry.json")
    
    with open(registry_in, "r") as f:
        registry = json.load(f)
    selected_cols = list(registry["features"].keys())
    
    # 3. Read dataset memory-efficiently
    logger.info("Loading pruned feature matrix...")
    df_features = pd.read_parquet(train_in, columns=["TransactionID"] + selected_cols)
    df_raw = pd.read_parquet(target_in, columns=["TransactionID", "isFraud"])
    
    df_merged = pd.merge(df_features, df_raw, on="TransactionID", how="inner")
    
    del df_features
    del df_raw
    gc.collect()
    
    X = df_merged[selected_cols]
    y = df_merged["isFraud"]
    
    from src.config.config import ConfigurationManager
    config = ConfigurationManager().get_config()

    # Data Preparation Spec: time-based train/validation split (e.g. 80/20)
    split_idx = int(len(X) * (1.0 - config.training.val_ratio))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Fast train limits to prevent OOM
    if len(X_train) > config.training.max_samples:
        rng = np.random.RandomState(config.seed)
        fit_idx = rng.choice(X_train.index, size=config.training.max_samples, replace=False)
        X_train_fit = X_train.loc[fit_idx]
        y_train_fit = y_train.loc[fit_idx]
    else:
        X_train_fit = X_train
        y_train_fit = y_train
        
    # 4. Train Logistic Regression Baseline
    logger.info("Training Logistic Regression baseline...")
    lr_baseline = LogisticRegressionBaseline(
        threshold=config.training.decision_threshold, 
        random_state=config.seed, 
        n_jobs=-1
    )
    lr_baseline.fit(X_train_fit, y_train_fit)
    
    # Predict and evaluate
    lr_train_pred = lr_baseline.predict_proba(X_train_fit)[:, 1]
    lr_val_pred = lr_baseline.predict_proba(X_val)[:, 1]
    
    lr_train_auc = roc_auc_score(y_train_fit, lr_train_pred) if len(np.unique(y_train_fit)) > 1 else 0.5
    lr_val_auc = roc_auc_score(y_val, lr_val_pred) if len(np.unique(y_val)) > 1 else 0.5
    
    lr_train_acc = accuracy_score(y_train_fit, (lr_train_pred >= 0.5).astype(int))
    lr_val_acc = accuracy_score(y_val, (lr_val_pred >= 0.5).astype(int))
    
    lr_summary = {
        "train_auc": lr_train_auc,
        "val_auc": lr_val_auc,
        "train_accuracy": lr_train_acc,
        "val_accuracy": lr_val_acc,
        "fit_metrics": lr_baseline.fit_metrics_
    }
    
    # 5. Train XGBoost Baseline
    logger.info("Training XGBoost baseline...")
    xgb_baseline = XGBoostBaseline(
        threshold=config.training.decision_threshold, 
        random_state=config.seed, 
        n_jobs=-1
    )
    xgb_baseline.fit(X_train_fit, y_train_fit)
    
    xgb_train_pred = xgb_baseline.predict_proba(X_train_fit)[:, 1]
    xgb_val_pred = xgb_baseline.predict_proba(X_val)[:, 1]
    
    xgb_train_auc = roc_auc_score(y_train_fit, xgb_train_pred) if len(np.unique(y_train_fit)) > 1 else 0.5
    xgb_val_auc = roc_auc_score(y_val, xgb_val_pred) if len(np.unique(y_val)) > 1 else 0.5
    
    xgb_train_acc = accuracy_score(y_train_fit, (xgb_train_pred >= 0.5).astype(int))
    xgb_val_acc = accuracy_score(y_val, (xgb_val_pred >= 0.5).astype(int))
    
    xgb_summary = {
        "train_auc": xgb_train_auc,
        "val_auc": xgb_val_auc,
        "train_accuracy": xgb_train_acc,
        "val_accuracy": xgb_val_acc,
        "fit_metrics": xgb_baseline.fit_metrics_
    }
    
    # Save outputs
    output_dir = Path("data/models/v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out_lr = output_dir / "baseline_lr.pkl"
    out_xgb = output_dir / "baseline_xgb.pkl"
    out_report = Path("reports/models/baseline_models_report.html")
    
    with open(out_lr, "wb") as f:
        pickle.dump(lr_baseline, f)
    with open(out_xgb, "wb") as f:
        pickle.dump(xgb_baseline, f)
        
    logger.info("Saved baseline LR model to: %s", out_lr)
    logger.info("Saved baseline XGB model to: %s", out_xgb)
    
    # Render HUD
    generate_baseline_hud_report(lr_summary, xgb_summary, out_report)
    
    # MLflow tracking
    logger.info("Logging baseline metrics & artifacts to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="baseline_models_training")
        started = True
        
    try:
        # Params
        mlflow.log_params({
            "stage": "baseline_models",
            "lr_random_state": 42,
            "xgb_estimators": 100,
            "xgb_max_depth": 5,
            "xgb_learning_rate": 0.1
        })
        
        # Metrics
        mlflow.log_metrics({
            "lr_train_auc": lr_summary["train_auc"],
            "lr_val_auc": lr_summary["val_auc"],
            "lr_train_accuracy": lr_summary["train_accuracy"],
            "lr_val_accuracy": lr_summary["val_accuracy"],
            "xgb_train_auc": xgb_summary["train_auc"],
            "xgb_val_auc": xgb_summary["val_auc"],
            "xgb_train_accuracy": xgb_summary["train_accuracy"],
            "xgb_val_accuracy": xgb_summary["val_accuracy"]
        })
        
        mlflow.log_artifact(str(out_lr), artifact_path="baseline_models")
        mlflow.log_artifact(str(out_xgb), artifact_path="baseline_models")
        mlflow.log_artifact(str(out_report), artifact_path="baseline_models")
    except Exception as e:
        logger.warning("MLflow logging exception: %s", e)
    finally:
        if started:
            mlflow.end_run()
            
    logger.info("Baseline models training execution finished.")


if __name__ == "__main__":
    main()
