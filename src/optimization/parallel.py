"""7.7 Parallel Optimization Module.

Provides parallel and multi-process execution management for hyperparameter optimization studies.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from typing import Any, Callable, Dict, Optional

import optuna

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParallelOptimizationExecutor:
    """Manages parallel execution of hyperparameter optimization trials across CPU cores."""

    def __init__(self, n_jobs: int = -1, random_state: int = 42):
        if n_jobs == -1 or n_jobs is None:
            self.n_jobs = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.n_jobs = max(1, n_jobs)
        self.random_state = random_state

    def get_worker_seed(self, worker_id: int) -> int:
        """Derives a deterministic random seed per parallel worker process."""
        return self.random_state + (worker_id * 10007)

    def execute_parallel_study(
        self,
        study: optuna.Study,
        objective_func: Callable[[optuna.Trial], float],
        n_trials: int = 20,
        timeout: Optional[float] = None,
    ) -> optuna.Study:
        """Executes trials in parallel across configured CPU workers."""
        logger.info(
            f"Executing parallel optimization study '{study.study_name}' using {self.n_jobs} workers for {n_trials} trials."
        )
        study.optimize(
            objective_func,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=False,
        )
        logger.info(
            f"Parallel optimization finished. Best value: {study.best_value:.5f} (Best params: {study.best_params})"
        )
        return study
