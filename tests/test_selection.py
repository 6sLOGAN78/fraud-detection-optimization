"""Unit tests verifying CandidateModelSelector champion selection resolution."""

import pytest
from src.evaluation.selection import CandidateModelSelector


def test_candidate_model_selector() -> None:
    comp_results = {
        "comparisons": {
            "ModelA": {"f2_score": 0.82, "f1_score": 0.80, "auc": 0.85},
            "ModelB": {"f2_score": 0.88, "f1_score": 0.82, "auc": 0.89}
        }
    }
    
    selector = CandidateModelSelector(optimize_metric="f2_score")
    best_name, best_metrics = selector.select_best_model(comp_results)
    
    assert best_name == "ModelB"
    assert best_metrics["auc"] == 0.89
    
    empty_results = {"comparisons": {}}
    with pytest.raises(ValueError):
        selector.select_best_model(empty_results)
