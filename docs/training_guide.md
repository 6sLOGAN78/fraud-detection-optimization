# Model Training and Hyperparameter Optimization Guide

This document details model training strategies, cross-validation setup, and Bayesian hyperparameter tuning with Optuna.

---

## ⚙️ Model Development Architecture

The candidate model framework supports:
- **Baseline Models**: Logistic Regression & Random Forest
- **Gradient Boosting Models**: LightGBM, XGBoost, CatBoost
- **Ensemble Classifier**: Weighted probability blending of gradient boosted trees

---

## 🎯 Stratified K-Fold Cross Validation
Cross-validation uses 5-fold Stratified K-Fold with fixed random seed (`42`) to preserve class imbalance ratio (~3.5% fraud rate) across all folds.

---

## ⚡ Bayesian Optimization with Optuna
Hyperparameter tuning utilizes Optuna TPE (`TPESampler`) and CMA-ES (`CmaEsSampler`) with automated trial pruning (`MedianPruner`).

To launch hyperparameter tuning:
```bash
python3 -m src.pipelines.run_optimization --n-trials 50
```
Best hyperparameters are automatically serialized to `configs/optimized/` and logged to MLflow.
