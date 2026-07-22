"""Pipeline orchestration script executing pre-validation, model development training/evaluation, and monochromatic HTML reporting."""

from __future__ import annotations

import logging
import json
import gc
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import mlflow

from src.models.development import ModelDevelopmentPipeline

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
        err = f"Pipeline Dependency Failure: The following previous stage outputs are missing: {missing}"
        logger.error(err)
        raise FileNotFoundError(err)
        
    logger.info("Prior stage verification check successfully passed.")


def generate_monochromatic_hud_report(summary: dict, output_path: Path) -> None:
    """Renders a futuristic minimal monochromatic HTML summary of model development performance."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fit_metrics = summary.get("fit_metrics", {})
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model Development HUD // Performance Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #050505;
            --panel-bg: rgba(26, 26, 26, 0.45);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-color: #f5f5f5;
            --text-muted: #888888;
            --accent-glow: rgba(255, 255, 255, 0.15);
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
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            border: 1px solid var(--border-color);
            background: rgba(255, 255, 255, 0.02);
            padding: 15px;
            text-align: center;
        }}
        
        .metric-card .caption {{
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}
        
        .metric-card .value {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 22px;
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
        
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'Share Tech Mono', monospace;
            font-size: 13px;
            margin-bottom: 30px;
        }}
        
        .stats-table th, .stats-table td {{
            border: 1px solid var(--border-color);
            padding: 10px;
            text-align: left;
        }}
        
        .stats-table th {{
            background: rgba(255, 255, 255, 0.05);
            text-transform: uppercase;
        }}
    </style>
</head>
<body>
    <div class="hud-container">
        <div class="header">
            <div>
                <h1>Model Development HUD</h1>
                <div style="font-size: 12px; color: var(--text-muted);">V1.0 // Gradient Boosted Baseline</div>
            </div>
            <div class="sys-info">
                STATUS: ACTIVE<br>
                DATE: {now_str}<br>
                ENGINE: Random Forest Classifier
            </div>
        </div>
        
        <div class="section-label">Pipeline Score Grid</div>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="caption">Train AUC</div>
                <div class="value">{summary['train_auc']:.5f}</div>
            </div>
            <div class="metric-card">
                <div class="caption">Validation AUC</div>
                <div class="value">{summary['val_auc']:.5f}</div>
            </div>
            <div class="metric-card">
                <div class="caption">Train Acc</div>
                <div class="value">{summary['train_accuracy']:.5f}</div>
            </div>
            <div class="metric-card">
                <div class="caption">Validation Acc</div>
                <div class="value">{summary['val_accuracy']:.5f}</div>
            </div>
        </div>
        
        <div class="section-label">Computation Profile Matrix</div>
        <table class="stats-table">
            <thead>
                <tr>
                    <th>Metric Parameter</th>
                    <th>Runtime Measurement</th>
                    <th>Optimal Margin</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Fit Duration (seconds)</td>
                    <td>{fit_metrics.get('fit_time_seconds', 0.0):.4f}s</td>
                    <td>Under 10.0s</td>
                </tr>
                <tr>
                    <td>Memory Delta (MB)</td>
                    <td>{fit_metrics.get('memory_delta_mb', 0.0):.2f}MB</td>
                    <td>Under 500MB</td>
                </tr>
                <tr>
                    <td>Sample Cardinality</td>
                    <td>{fit_metrics.get('samples_count', 0)} rows</td>
                    <td>50k - 100k</td>
                </tr>
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html_content)
    logger.info("Monochromatic model development HUD report saved to: %s", output_path)


def main() -> None:
    # 1. Run dependencies validation gate
    run_pre_execution_verification_gate()
    
    # 2. Input/Output Processing Specs files
    train_in = Path("data/feature_store_engineered/v1/train_selected_features.parquet")
    target_in = Path("data/interim/train_merged.parquet")
    registry_in = Path("data/feature_store_engineered/v1/feature_registry.json")
    
    # Read the feature registry to find selected columns
    with open(registry_in, "r") as f:
        registry = json.load(f)
    selected_cols = list(registry["features"].keys())
    logger.info("Loaded final feature store registry features: %s", selected_cols)
    
    # 3. Read dataset memory-efficiently (ONLY selected features + index)
    logger.info("Loading training dataset features...")
    df_features = pd.read_parquet(train_in, columns=["TransactionID"] + selected_cols)
    logger.info("Loading target labels...")
    df_raw = pd.read_parquet(target_in, columns=["TransactionID", "isFraud"])
    
    # Merge on TransactionID
    df_merged = pd.merge(df_features, df_raw, on="TransactionID", how="inner")
    
    # Free memory
    del df_features
    del df_raw
    gc.collect()
    
    X = df_merged[selected_cols]
    y = df_merged["isFraud"]
    
    # Limit to 50k samples for fast training runs & zero OOM
    if len(X) > 50000:
        logger.info("Sampling 50,000 samples for development fit...")
        rng = np.random.RandomState(42)
        fit_idx = rng.choice(X.index, size=50000, replace=False)
        X_fit = X.loc[fit_idx]
        y_fit = y.loc[fit_idx]
    else:
        X_fit = X
        y_fit = y
        
    # 4. Fit Pipeline
    pipeline = ModelDevelopmentPipeline(threshold=0.05, random_state=42, n_jobs=-1, log_level="INFO")
    summary = pipeline.fit_and_validate(X_fit, y_fit, val_ratio=0.2)
    
    # Save output artifacts
    output_dir = Path("data/models/v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out_model = output_dir / "model_bundle.pkl"
    out_report = Path("reports/models/model_development_report.html")
    
    # Serialize model bundle
    bundle = {
        "pipeline": pipeline,
        "summary": summary,
        "selected_features": selected_cols
    }
    pipeline.standards.serialize(bundle, out_model)
    logger.info("Saved trained model bundle to: %s", out_model)
    
    # Generate the HUD dashboard
    generate_monochromatic_hud_report(summary, out_report)
    
    # 5. Log to MLflow
    logger.info("Logging model training run to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="model_development_pipeline")
        started = True
        
    try:
        mlflow.log_params({
            "pipeline_stage": "model_development",
            "threshold": 0.05,
            "random_state": 42,
            "n_jobs": -1
        })
        
        mlflow.log_metrics({
            "train_auc": summary["train_auc"],
            "val_auc": summary["val_auc"],
            "train_accuracy": summary["train_accuracy"],
            "val_accuracy": summary["val_accuracy"],
            "fit_time_seconds": summary["fit_metrics"].get("fit_time_seconds", 0.0),
            "memory_delta_mb": summary["fit_metrics"].get("memory_delta_mb", 0.0)
        })
        
        mlflow.log_artifact(str(out_model), artifact_path="model_development")
        mlflow.log_artifact(str(out_report), artifact_path="model_development")
    except Exception as e:
        logger.warning("MLflow logging warning: %s", e)
    finally:
        if started:
            mlflow.end_run()
            
    logger.info("Model development pipeline executed successfully.")


if __name__ == "__main__":
    main()
