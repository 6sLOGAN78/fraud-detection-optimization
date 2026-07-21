# MLflow Experiment Tracking

This document outlines standard parameters, performance metrics, and artifacts logged by the experimental tracking suite.

## Logged Components

### 1. Parameters
- **Architecture Specs**: Model name (LightGBM/CatBoost/XGBoost), Random seed, Cross-validation fold size.
- **Hyperparameters**: Learning rates, tree estimators, depths, regularization terms, bagging fractions.
- **Optimization run**: Optuna trial sequence index, objective objectives.

### 2. Metrics
- **Performance**: Area under ROC curve (ROC-AUC), Area under Precision-Recall Curve (PR-AUC), F1 Score, Matthews Correlation Coefficient (MCC).
- **Quality**: Log Loss, Brier Score (accuracy of probabilities), Expected Calibration Error (ECE), Kolmogorov-Smirnov (KS) statistic.

### 3. Artifacts
- **Model Files**: Serialized model binaries (`.pkl`).
- **Feature Importance**: Text file containing relative feature weights and visualization plots.
- **Interactive Reports**: SHAP values, summary plots, calibration curves, confusion matrix.
- **Submission Output**: Logged copy of prediction files ready for Kaggle submission.

## Offline Fallback

The `MLflowTracker` (configured in `src/utils/mlflow_helper.py`) automatically detects server availability. If the remote server specified in `configs/config.yaml` is offline, tracking redirects output locally to `mlruns/` to prevent training crashes and guarantee experiment persistence.
