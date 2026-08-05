"""Unit tests for Part 9 — Explainability & Model Transparency Framework."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.explainability import (
    ExplainabilityArchitectureDesign,
    ExplainabilityImplementationStandards,
    ExplainabilityPreExecutionGate,
    FairnessBiasAssessmentEngine,
    GlobalFeatureImportanceEngine,
    IndividualConditionalExpectationEngine,
    LocalExplanationsEngine,
    ModelTransparencyEngine,
    PartialDependenceEngine,
    SHAPAnalysisEngine,
    WaterfallPlotsEngine,
)


@pytest.fixture
def sample_explain_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 4), columns=["f1", "f2", "f3", "f4"])
    y = np.random.randint(0, 2, 50)
    model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    model.fit(X, y)
    return model, X, y


def test_9_1_architecture(sample_explain_data):
    model, X, _ = sample_explain_data
    arch = ExplainabilityArchitectureDesign()
    m_clean, X_clean = arch.validate_inputs(model, X)
    assert X_clean.shape == X.shape

    norm = ExplainabilityImplementationStandards.normalize_importance({"a": 10.0, "b": 10.0})
    assert norm["a"] == 0.5


def test_9_2_to_9_6_shap(sample_explain_data):
    model, X, _ = sample_explain_data

    global_imp = GlobalFeatureImportanceEngine().calculate(model, X)
    assert len(global_imp) == 4
    assert sum(global_imp.values()) == pytest.approx(1.0, abs=1e-3)

    shap_matrix = SHAPAnalysisEngine().calculate_shap_matrix(model, X)
    assert len(shap_matrix["shap_values"]) == 50

    local_exp = LocalExplanationsEngine().explain_sample(model, X, sample_idx=0)
    assert local_exp["sample_index"] == 0
    assert len(local_exp["contributions"]) == 4

    waterfall = WaterfallPlotsEngine().generate_plot_data(model, X, sample_idx=0)
    assert "waterfall_steps" in waterfall


def test_9_7_to_9_9_partial_dependence(sample_explain_data):
    model, X, _ = sample_explain_data

    pdp = PartialDependenceEngine().calculate_pdp(model, X, feature_name="f1", grid_resolution=5)
    assert pdp["feature"] == "f1"
    assert len(pdp["grid_values"]) == 5

    ice = IndividualConditionalExpectationEngine().calculate_ice(model, X, feature_name="f1", grid_resolution=5)
    assert ice["feature"] == "f1"
    assert len(ice["ice_curves"]) == 50


def test_9_10_to_9_12_transparency(sample_explain_data):
    model, X, y = sample_explain_data

    transparency = ModelTransparencyEngine().extract_surrogate_rules(model, X, max_depth=2)
    assert "extracted_rules_text" in transparency
    assert transparency["surrogate_fidelity_score"] > 0.5

    protected_attr = np.random.choice(["Group_A", "Group_B"], size=len(y))
    y_prob = model.predict_proba(X)[:, 1]
    fairness = FairnessBiasAssessmentEngine().evaluate_fairness(y, y_prob, protected_attr)
    assert "disparate_impact_ratio" in fairness
