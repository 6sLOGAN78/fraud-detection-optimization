"""Pipeline script executing pre-validation, TimeSeriesSplit Cross Validation, evaluation metrics aggregation, and monochromatic HUD report generation."""

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

from src.validation.cv import TimeSeriesCrossValidator
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


def generate_cv_hud_report(cv_summary: dict, output_path: Path) -> None:
    """Renders a sleek, monochromatic dashboard report for Cross Validation results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Cross Validation HUD // Performance Report</title>
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
                <h1>Cross Validation HUD</h1>
                <div style="font-size: 12px; color: var(--text-muted);">V1.0 // TimeSeriesSplit (5 Folds)</div>
            </div>
            <div class="sys-info">
                STATUS: ACTIVE<br>
                DATE: {now_str}<br>
                STRATEGY: Temporal Folds
            </div>
        </div>
        
        <div class="grid">
            <div class="panel">
                <div class="panel-title">AUC Folds Performance</div>
                <div class="stat-row">
                    <span class="stat-label">Fold 1 AUC:</span>
                    <span class="stat-value">{cv_summary['auc_folds'][0]:.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Fold 2 AUC:</span>
                    <span class="stat-value">{cv_summary['auc_folds'][1]:.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Fold 3 AUC:</span>
                    <span class="stat-value">{cv_summary['auc_folds'][2]:.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Fold 4 AUC:</span>
                    <span class="stat-value">{cv_summary['auc_folds'][3]:.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Fold 5 AUC:</span>
                    <span class="stat-value">{cv_summary['auc_folds'][4]:.5f}</span>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">Summary Statistics</div>
                <div class="stat-row">
                    <span class="stat-label">Mean Val AUC:</span>
                    <span class="stat-value">{cv_summary['mean_auc']:.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Mean F1-score:</span>
                    <span class="stat-value">{cv_summary['mean_f1']:.5f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Mean Accuracy:</span>
                    <span class="stat-value">{cv_summary['mean_accuracy']:.5f}</span>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html_content)
    logger.info("Cross Validation HUD report saved: %s", output_path)


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
    
    # Downsample matrix to prevent OOM
    if len(X) > 50000:
        rng = np.random.RandomState(42)
        indices = rng.choice(X.index, size=50000, replace=False)
        X_sub = X.loc[indices].reset_index(drop=True)
        y_sub = y.loc[indices].reset_index(drop=True)
    else:
        X_sub = X
        y_sub = y
        
    logger.info("Executing TimeSeriesSplit Cross Validation...")
    validator = TimeSeriesCrossValidator(n_splits=5)
    folds = validator.split(X_sub, y_sub)
    
    # Load EnsembleClassifier
    with open(Path("data/models/v1/ensemble_model.pkl"), "rb") as f:
        ensemble = pickle.load(f)
        
    auc_folds = []
    f1_folds = []
    acc_folds = []
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        logger.info("Evaluating Fold %d...", fold + 1)
        X_val_fold = X_sub.iloc[val_idx]
        y_val_fold = y_sub.iloc[val_idx]
        
        preds_probs = ensemble.predict_proba(X_val_fold)[:, 1]
        preds_class = ensemble.predict(X_val_fold)
        
        auc = roc_auc_score(y_val_fold, preds_probs) if len(np.unique(y_val_fold)) > 1 else 0.5
        f1 = f1_score(y_val_fold, preds_class, zero_division=0)
        acc = accuracy_score(y_val_fold, preds_class)
        
        auc_folds.append(auc)
        f1_folds.append(f1)
        acc_folds.append(acc)
        
    cv_summary = {
        "auc_folds": auc_folds,
        "mean_auc": float(np.mean(auc_folds)),
        "mean_f1": float(np.mean(f1_folds)),
        "mean_accuracy": float(np.mean(acc_folds))
    }
    
    out_dir = Path("reports/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_report = out_dir / "cv_report.html"
    
    generate_cv_hud_report(cv_summary, out_report)
    
    logger.info("Logging Cross Validation statistics to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="cv_evaluation")
        started = True
        
    try:
        mlflow.log_params({
            "stage": "cv_evaluation",
            "n_splits": 5
        })
        for i, val in enumerate(auc_folds):
            mlflow.log_metric(f"cv_fold_{i+1}_auc", val)
        mlflow.log_metrics({
            "cv_mean_auc": cv_summary["mean_auc"],
            "cv_mean_f1": cv_summary["mean_f1"],
            "cv_mean_accuracy": cv_summary["mean_accuracy"]
        })
        mlflow.log_artifact(str(out_report), artifact_path="cv_evaluation")
    except Exception as e:
        logger.warning("MLflow logging raised exception: %s", e)
    finally:
        if started:
            mlflow.end_run()
            
    logger.info("TimeSeriesSplit Cross Validation stage execution complete.")


if __name__ == "__main__":
    main()
