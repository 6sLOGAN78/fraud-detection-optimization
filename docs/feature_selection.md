# Feature Selection

This document describes the automated feature filtering layer, preventing overfitting and reducing runtime.

## Selection Stages

1. **Information Completeness Filter**:
   - Removes features with missing rates higher than the configurable threshold (default: 90%).

2. **Zero Variance Filter**:
   - Drops features with zero or near-zero variance since they provide no classifier discriminative power.

3. **Collinearity Filter**:
   - Calculates correlation matrices on numerical features.
   - Removes one of a pair of features with correlation above the threshold (default: 0.95), retaining the one with the higher univariate importance.

4. **Relative Importance Filter**:
   - Executes a lightweight model (e.g. LightGBM baseline) to evaluate relative feature importances.
   - Drops features with relative importance below the threshold (default: 0.05).
