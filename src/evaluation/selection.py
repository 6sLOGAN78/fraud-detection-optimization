"""Candidate Model Selection module implementing the champion model determination strategy."""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CandidateModelSelector:
    """Wrapper class to determine champion model from evaluation metrics comparison."""
    def __init__(self, optimize_metric: str = "f2_score", log_level: str = "INFO"):
        self.optimize_metric = optimize_metric
        self.log_level = log_level

    def select_best_model(self, comparison_results: dict) -> tuple[str, dict]:
        """Resolves the best scoring model path or name based on comparative metrics."""
        comps = comparison_results.get("comparisons", {})
        if not comps:
            raise ValueError("No comparisons dict found in comparison results")
            
        best_name = None
        best_score = -1.0
        best_metrics = {}
        
        for name, metrics in comps.items():
            if self.optimize_metric not in metrics:
                raise KeyError(f"Optimize metric '{self.optimize_metric}' not found in metrics for '{name}'")
            score = metrics[self.optimize_metric]
            if score > best_score:
                best_score = score
                best_name = name
                best_metrics = metrics
                
        if best_name is None:
            raise ValueError("Could not determine any champion candidate model")
            
        logger.info("CandidateModelSelector champion promoted: %s with %s = %.5f",
                    best_name, self.optimize_metric, best_score)
        return best_name, best_metrics
