"""7.5 Early Stopping Mechanism Module.

Provides early stopping callbacks for model training iterations and Optuna optimization studies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import optuna

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStoppingHandler:
    """Tracks consecutive non-improving evaluation metrics for early stopping during model training."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "maximize"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()

        self.best_score: float = (
            -float("inf") if self.mode == "maximize" else float("inf")
        )
        self.counter: int = 0
        self.should_stop: bool = False

    def update(self, current_score: float) -> bool:
        """Updates internal state with new score and returns True if training should stop."""
        if self.mode == "maximize":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            self.should_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered! Counter reached patience ({self.patience}). Best score: {self.best_score:.5f}"
                )

        return self.should_stop


class OptunaEarlyStoppingCallback:
    """Optuna study callback that stops the study if no improvement is observed for `patience` trials."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: Optional[float] = None
        self.no_improvement_counter: int = 0

    def __call__(self, study: optuna.Study, trial: optuna.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        direction = study.direction
        current_value = trial.value

        if current_value is None:
            return

        if self.best_value is None:
            self.best_value = current_value
            self.no_improvement_counter = 0
            return

        if direction == optuna.study.StudyDirection.MAXIMIZE:
            improved = current_value > (self.best_value + self.min_delta)
        else:
            improved = current_value < (self.best_value - self.min_delta)

        if improved:
            self.best_value = current_value
            self.no_improvement_counter = 0
        else:
            self.no_improvement_counter += 1
            if self.no_improvement_counter >= self.patience:
                logger.info(
                    f"Optuna Early Stopping: No improvement for {self.no_improvement_counter} consecutive trials. Stopping study."
                )
                study.stop()
