"""7.3 Bayesian Optimization Engine Module.

Provides Bayesian optimization orchestration, objective function evaluation,
acquisition function handling, and surrogate search logic.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.optimization.search_space import StatisticalMetricSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianOptimizationEngine:
    """Bayesian Optimization engine orchestrating parameter search, cross-validated objective

    evaluations, and trial history management.
    """

    def __init__(
        self,
        metric_spec: Optional[StatisticalMetricSpec] = None,
        n_trials: int = 20,
        random_state: int = 42,
        n_splits: int = 3,
    ):
        self.metric_spec = metric_spec or StatisticalMetricSpec("roc_auc")
        self.n_trials = n_trials
        self.random_state = random_state
        self.n_splits = n_splits
        self.trials_history: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = (
            -float("inf") if self.metric_spec.direction == "maximize" else float("inf")
        )

    def evaluate_cv_objective(
        self,
        model_factory: Callable[[Dict[str, Any]], Any],
        params: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """Evaluates mean cross-validation score for a set of hyperparameters."""
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = model_factory(params)
            model.fit(X_tr, y_tr)

            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_val)[:, 1]
            elif hasattr(model, "predict"):
                preds = model.predict(X_val)
            else:
                raise AttributeError("Model has neither predict_proba nor predict method.")

            score = roc_auc_score(y_val, preds)
            scores.append(score)

        mean_score = float(np.mean(scores))
        return mean_score

    def record_trial(
        self, trial_id: int, params: Dict[str, Any], score: float, duration: float
    ) -> None:
        """Records trial results into trial history and updates best score/parameters."""
        trial_record = {
            "trial_id": trial_id,
            "params": params,
            "score": score,
            "duration_seconds": round(duration, 4),
        }
        self.trials_history.append(trial_record)

        if self.metric_spec.is_better(score, self.best_score):
            self.best_score = score
            self.best_params = params.copy()
            logger.info(
                f"New best score found at trial {trial_id}: {score:.5f} (Params: {params})"
            )

    def get_summary(self) -> Dict[str, Any]:
        """Returns summary of optimization session."""
        return {
            "total_trials": len(self.trials_history),
            "best_score": self.best_score,
            "best_params": self.best_params,
            "metric": self.metric_spec.metric_name,
            "direction": self.metric_spec.direction,
        }
