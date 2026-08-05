"""Explainability package for IEEE-CIS Fraud Detection project (Part 9)."""

from src.explainability.architecture import (
    ExplainabilityArchitectureDesign,
    ExplainabilityPreExecutionGate,
    ExplainabilityInputOutputProcessor,
    ExplainabilityImplementationStandards,
)
from src.explainability.shap_engine import (
    GlobalFeatureImportanceEngine,
    SHAPAnalysisEngine,
    LocalExplanationsEngine,
    WaterfallPlotsEngine,
    ForcePlotsEngine,
)
from src.explainability.partial_dependence import (
    PartialDependenceEngine,
    IndividualConditionalExpectationEngine,
    FeatureInteractionExplanationsEngine,
)
from src.explainability.transparency import (
    ModelTransparencyEngine,
    FairnessBiasAssessmentEngine,
    ExplainabilityReporter,
)

__all__ = [
    "ExplainabilityArchitectureDesign",
    "ExplainabilityPreExecutionGate",
    "ExplainabilityInputOutputProcessor",
    "ExplainabilityImplementationStandards",
    "GlobalFeatureImportanceEngine",
    "SHAPAnalysisEngine",
    "LocalExplanationsEngine",
    "WaterfallPlotsEngine",
    "ForcePlotsEngine",
    "PartialDependenceEngine",
    "IndividualConditionalExpectationEngine",
    "FeatureInteractionExplanationsEngine",
    "ModelTransparencyEngine",
    "FairnessBiasAssessmentEngine",
    "ExplainabilityReporter",
]
