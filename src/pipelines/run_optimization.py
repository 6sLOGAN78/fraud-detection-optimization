"""Pipeline script to execute Part 7 — Hyperparameter Optimization Framework."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

from src.optimization import (
    BestConfigurationRegistry,
    OptunaEarlyStoppingCallback,
    OptimizationArchitectureDesign,
    OptimizationMonitor,
    OptimizationPipelineExecution,
    OptunaStudyManager,
    SearchSpaceBuilder,
    StatisticalMetricSpec,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_model_hpo(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 10,
    n_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Runs Optuna HPO for a specific model type using cross-validation."""
    logger.info(f"--- Starting HPO for Model Type: {model_type} ---")
    space = SearchSpaceBuilder.get_default_space(model_type)
    study_manager = OptunaStudyManager(
        study_name=f"hpo_{model_type}",
        direction="maximize",
        sampler_name="tpe",
        pruner_name="median",
        random_state=random_state,
    )

    def objective(trial: optuna.Trial) -> float:
        params = SearchSpaceBuilder.sample_optuna_params(trial, space)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            if model_type in ["lgb", "lightgbm"]:
                model = lgb.LGBMClassifier(**params, random_state=random_state, verbose=-1)
            elif model_type in ["xgb", "xgboost"]:
                model = xgb.XGBClassifier(**params, random_state=random_state, eval_metric="logloss")
            elif model_type in ["cat", "catboost"]:
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(**params, random_seed=random_state, verbose=0)
            elif model_type in ["rf", "random_forest"]:
                model = RandomForestClassifier(**params, random_state=random_state)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            model.fit(X_tr, y_tr)
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_val)[:, 1]
            else:
                preds = model.predict(X_val)

            score = roc_auc_score(y_val, preds)
            scores.append(score)

            # Check trial pruning at fold step
            trial.report(float(np.mean(scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    early_stopping = OptunaEarlyStoppingCallback(patience=5, min_delta=1e-4)
    study_manager.study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[early_stopping],
        show_progress_bar=False,
    )

    monitor = OptimizationMonitor()
    summary = monitor.log_study_summary(study_manager.study, model_type=model_type)

    registry = BestConfigurationRegistry()
    registry.register_configuration(
        model_name=model_type,
        best_params=study_manager.study.best_params,
        best_score=study_manager.study.best_value,
        metric_name="roc_auc",
        metadata=summary,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 7 HPO Pipeline")
    parser.add_argument("--n-trials", type=int, default=5, help="Number of trials per model")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of samples to train on for HPO")
    args = parser.parse_args()

    # Pre-execution pipeline verification gate
    train_path = Path("data/interim/train_cleaned.parquet")
    if not train_path.exists():
        # Fallback to train_merged if train_cleaned isn't present
        train_path = Path("data/interim/train_merged.parquet")

    gate = OptimizationPipelineExecution(required_artifacts=[str(train_path)])
    gate.verify_prerequisites()

    logger.info(f"Loading data from {train_path}...")
    df = pd.read_parquet(train_path)

    if "isFraud" not in df.columns:
        logger.error("Target column 'isFraud' not found in dataset.")
        sys.exit(1)

    # Subsample for faster HPO if dataset is huge
    if len(df) > args.n_samples:
        df = df.sample(n=args.n_samples, random_state=42).reset_index(drop=True)

    y = df["isFraud"]
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols].fillna(0)

    # Architecture Design fit/transform check
    arch = OptimizationArchitectureDesign()
    arch.fit(X, y)
    X = arch.transform(X)

    # Run HPO for LightGBM and XGBoost models
    models_to_optimize = ["lightgbm", "xgboost"]
    for m_type in models_to_optimize:
        run_model_hpo(model_type=m_type, X=X, y=y, n_trials=args.n_trials)

    logger.info("Part 7 Hyperparameter Optimization Pipeline completed successfully.")


if __name__ == "__main__":
    main()
