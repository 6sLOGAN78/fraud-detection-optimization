"""7.6 Trial Pruning System Module.

Provides trial pruner initialization, intermediate evaluation reporting,
and trial pruning condition evaluation for hyperparameter tuning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import optuna
from optuna.pruners import (
    BasePruner,
    HyperbandPruner,
    MedianPruner,
    NopPruner,
    PercentilePruner,
    PatientPruner,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrialPruningEngine:
    """Configures trial pruners and manages intermediate step reporting and pruning checks."""

    def __init__(
        self,
        pruner_type: str = "median",
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
    ):
        self.pruner_type = pruner_type.lower()
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.interval_steps = interval_steps
        self.pruner = self._build_pruner()

    def _build_pruner(self) -> BasePruner:
        if self.pruner_type == "median":
            return MedianPruner(
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=self.n_warmup_steps,
                interval_steps=self.interval_steps,
            )
        elif self.pruner_type == "percentile":
            return PercentilePruner(
                percentile=50.0,
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=self.n_warmup_steps,
                interval_steps=self.interval_steps,
            )
        elif self.pruner_type == "hyperband":
            return HyperbandPruner(
                min_resource=1,
                max_resource="auto",
                reduction_factor=3,
            )
        elif self.pruner_type in ["none", "nop"]:
            return NopPruner()
        else:
            logger.warning(f"Unknown pruner_type '{self.pruner_type}'. Falling back to MedianPruner.")
            return MedianPruner(
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=self.n_warmup_steps,
                interval_steps=self.interval_steps,
            )

    def report_step(self, trial: optuna.Trial, value: float, step: int) -> bool:
        """Reports intermediate evaluation metric to Optuna trial and returns True if trial should be pruned."""
        trial.report(value, step)
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned at step {step} with value {value:.5f}")
            return True
        return False
