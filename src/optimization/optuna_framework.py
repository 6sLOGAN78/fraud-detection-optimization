"""7.4 Optuna Framework Integration Module.

Provides Optuna study management, sampler & pruner configuration,
storage backend configuration, and optimization execution.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import optuna
from optuna.pruners import BasePruner, HyperbandPruner, MedianPruner, NopPruner, PercentilePruner
from optuna.samplers import BaseSampler, CmaEsSampler, RandomSampler, TPESampler
import pandas as pd

from src.optimization.search_space import SearchSpaceBuilder, StatisticalMetricSpec

logging.basicConfig(level=logging.INFO)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class OptunaStudyManager:
    """Manages Optuna study creation, sampling, pruning, and optimization execution."""

    def __init__(
        self,
        study_name: str = "fraud_detection_hpo",
        direction: str = "maximize",
        sampler_name: str = "tpe",
        pruner_name: str = "median",
        storage: Optional[str] = None,
        random_state: int = 42,
    ):
        self.study_name = study_name
        self.direction = direction
        self.sampler_name = sampler_name.lower()
        self.pruner_name = pruner_name.lower()
        self.storage = storage
        self.random_state = random_state

        self.sampler = self._init_sampler()
        self.pruner = self._init_pruner()
        self.study = self._init_study()

    def _init_sampler(self) -> BaseSampler:
        if self.sampler_name == "tpe":
            return TPESampler(seed=self.random_state)
        elif self.sampler_name == "random":
            return RandomSampler(seed=self.random_state)
        elif self.sampler_name in ["cmaes", "cma-es"]:
            return CmaEsSampler(seed=self.random_state)
        else:
            logger.warning(f"Unknown sampler {self.sampler_name}, defaulting to TPESampler.")
            return TPESampler(seed=self.random_state)

    def _init_pruner(self) -> BasePruner:
        if self.pruner_name == "median":
            return MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        elif self.pruner_name == "hyperband":
            return HyperbandPruner()
        elif self.pruner_name == "percentile":
            return PercentilePruner(percentile=50.0)
        elif self.pruner_name in ["none", "nop"]:
            return NopPruner()
        else:
            return MedianPruner()

    def _init_study(self) -> optuna.Study:
        return optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage,
            load_if_exists=True,
        )

    def run_optimization(
        self,
        objective_func: Callable[[optuna.Trial], float],
        n_trials: int = 20,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
    ) -> optuna.Study:
        """Runs hyperparameter optimization with the given objective function."""
        logger.info(
            f"Starting Optuna optimization: study='{self.study_name}', n_trials={n_trials}, n_jobs={n_jobs}"
        )
        self.study.optimize(
            objective_func,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=False,
        )
        logger.info(
            f"Optuna optimization finished. Best value ({self.direction}): {self.study.best_value:.5f}"
        )
        return self.study

    def get_best_results(self) -> Dict[str, Any]:
        """Returns best trial parameters and value."""
        return {
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "best_trial_number": self.study.best_trial.number,
            "n_trials": len(self.study.trials),
        }
