"""7.2 Search Space Design Core Module.

Defines search space builders, validators, statistical metric specifications,
and threshold tuning criteria for hyperparameter optimization across LightGBM,
XGBoost, CatBoost, and Baseline models.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchSpaceBuilder:
    """Builds hyperparameter search space definitions for supported model families."""

    @staticmethod
    def get_default_space(model_type: str) -> Dict[str, Dict[str, Any]]:
        """Returns standard search space configuration bounds for a given model type."""
        model_type = model_type.lower()
        if model_type in ["lgb", "lightgbm"]:
            return {
                "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
                "max_depth": {"type": "int", "low": 3, "high": 12},
                "num_leaves": {"type": "int", "low": 15, "high": 255},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
                "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
                "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            }
        elif model_type in ["xgb", "xgboost"]:
            return {
                "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
                "gamma": {"type": "float", "low": 1e-8, "high": 5.0, "log": True},
                "min_child_weight": {"type": "int", "low": 1, "high": 10},
            }
        elif model_type in ["catboost", "cat"]:
            return {
                "iterations": {"type": "int", "low": 50, "high": 500, "step": 50},
                "depth": {"type": "int", "low": 4, "high": 10},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0},
                "random_strength": {"type": "float", "low": 1e-9, "high": 10.0, "log": True},
            }
        elif model_type in ["rf", "random_forest"]:
            return {
                "n_estimators": {"type": "int", "low": 50, "high": 300, "step": 50},
                "max_depth": {"type": "int", "low": 3, "high": 15},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            }
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    @staticmethod
    def sample_optuna_params(trial: Any, space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Samples hyperparameters from an Optuna trial given a search space definition dict."""
        params = {}
        for param_name, config in space.items():
            p_type = config["type"]
            if p_type == "int":
                step = config.get("step", 1)
                log = config.get("log", False)
                params[param_name] = trial.suggest_int(
                    param_name, config["low"], config["high"], step=step, log=log
                )
            elif p_type == "float":
                log = config.get("log", False)
                step = config.get("step", None)
                params[param_name] = trial.suggest_float(
                    param_name, config["low"], config["high"], step=step, log=log
                )
            elif p_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, config["choices"])
            else:
                raise ValueError(f"Unknown parameter type: {p_type}")
        return params


class SearchSpaceValidator:
    """Validates parameters generated against bounding rules."""

    @staticmethod
    def validate(params: Dict[str, Any], space: Dict[str, Dict[str, Any]]) -> bool:
        """Checks whether sampled parameters conform to space bounds."""
        for name, config in space.items():
            if name not in params:
                raise ValueError(f"Missing required parameter: {name}")
            val = params[name]
            p_type = config["type"]
            if p_type in ["int", "float"]:
                if val < config["low"] or val > config["high"]:
                    raise ValueError(
                        f"Parameter {name}={val} out of bounds [{config['low']}, {config['high']}]"
                    )
            elif p_type == "categorical":
                if val not in config["choices"]:
                    raise ValueError(
                        f"Parameter {name}={val} not in allowed choices {config['choices']}"
                    )
        return True


class StatisticalMetricSpec:
    """Defines target optimization metrics and direction (maximize or minimize)."""

    def __init__(self, metric_name: str = "roc_auc", direction: Optional[str] = None):
        self.metric_name = metric_name.lower()
        if direction is None:
            if self.metric_name in ["roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall"]:
                self.direction = "maximize"
            elif self.metric_name in ["logloss", "brier_score", "rmse", "mae", "mse"]:
                self.direction = "minimize"
            else:
                self.direction = "maximize"
        else:
            self.direction = direction.lower()

    def is_better(self, score: float, best_score: float) -> bool:
        """Determines if new score is an improvement over best_score."""
        if self.direction == "maximize":
            return score > best_score
        else:
            return score < best_score


class ThresholdTuningCriteria:
    """Specifies bounds and step granularity for threshold optimization within search space tuning."""

    def __init__(self, low: float = 0.01, high: float = 0.99, step: float = 0.01):
        self.low = low
        self.high = high
        self.step = step
        self.grid = np.arange(low, high + step / 2.0, step)

    def get_candidate_thresholds(self) -> np.ndarray:
        """Returns array of threshold candidates."""
        return self.grid.copy()
