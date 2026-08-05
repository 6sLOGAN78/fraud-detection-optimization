"""Optimization package for IEEE-CIS Fraud Detection project (Part 7)."""

from src.optimization.architecture import (
    OptimizationArchitectureDesign,
    CoreOptimizationEngine,
    OptimizationInputOutputProcessor,
    OptimizationImplementationStandards,
    OptimizationPipelineExecution,
)
from src.optimization.search_space import (
    SearchSpaceBuilder,
    SearchSpaceValidator,
    StatisticalMetricSpec,
    ThresholdTuningCriteria,
)
from src.optimization.bayesian import BayesianOptimizationEngine
from src.optimization.optuna_framework import OptunaStudyManager
from src.optimization.early_stopping import EarlyStoppingHandler, OptunaEarlyStoppingCallback
from src.optimization.pruning import TrialPruningEngine
from src.optimization.parallel import ParallelOptimizationExecutor
from src.optimization.monitoring import OptimizationMonitor
from src.optimization.registry import BestConfigurationRegistry

__all__ = [
    "OptimizationArchitectureDesign",
    "CoreOptimizationEngine",
    "OptimizationInputOutputProcessor",
    "OptimizationImplementationStandards",
    "OptimizationPipelineExecution",
    "SearchSpaceBuilder",
    "SearchSpaceValidator",
    "StatisticalMetricSpec",
    "ThresholdTuningCriteria",
    "BayesianOptimizationEngine",
    "OptunaStudyManager",
    "EarlyStoppingHandler",
    "OptunaEarlyStoppingCallback",
    "TrialPruningEngine",
    "ParallelOptimizationExecutor",
    "OptimizationMonitor",
    "BestConfigurationRegistry",
]
