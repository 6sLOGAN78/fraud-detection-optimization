"""Evaluation package for IEEE-CIS Fraud Detection project (Part 8)."""

from src.evaluation.framework import (
    EvaluationFrameworkDesign,
    EvaluationPreExecutionGate,
    EvaluationInputOutputProcessor,
    EvaluationImplementationStandards,
)
from src.evaluation.metrics import (
    ROCAUCEngine,
    PRAUCEngine,
    PrecisionEngine,
    RecallEngine,
    F1ScoreEngine,
    MCCEngine,
    KSStatisticEngine,
)
from src.evaluation.diagnostics import (
    CalibrationAnalysisEngine,
    LiftGainEngine,
    ThresholdAnalysisEngine,
    ConfusionMatrixEngine,
)
from src.evaluation.business import (
    ErrorAnalysisEngine,
    RobustnessEvaluationEngine,
    BusinessKPIEvaluator,
)

__all__ = [
    "EvaluationFrameworkDesign",
    "EvaluationPreExecutionGate",
    "EvaluationInputOutputProcessor",
    "EvaluationImplementationStandards",
    "ROCAUCEngine",
    "PRAUCEngine",
    "PrecisionEngine",
    "RecallEngine",
    "F1ScoreEngine",
    "MCCEngine",
    "KSStatisticEngine",
    "CalibrationAnalysisEngine",
    "LiftGainEngine",
    "ThresholdAnalysisEngine",
    "ConfusionMatrixEngine",
    "ErrorAnalysisEngine",
    "RobustnessEvaluationEngine",
    "BusinessKPIEvaluator",
]
