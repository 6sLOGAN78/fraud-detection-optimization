# Source Package Architecture (`src/`)

Welcome to the core codebase of the **IEEE-CIS Financial Fraud Detection & Optimization System**. The `src/` directory is structured into decoupled, modular sub-packages with strict input/output contracts and clean separation of concerns.

---

## 🏛️ Sub-Package Architecture & Distributed Data Flow

```
                                  [ Raw CSV Data ]
                                         │
                                         ▼
                             ┌──────────────────────┐
                             │      src/data/       │  (Ingestion, Downcasting, Schema Validation)
                             └──────────┬───────────┘
                                        │
                       ┌────────────────┴────────────────┐
                       ▼                                 ▼
           ┌──────────────────────┐          ┌──────────────────────┐
           │       src/eda/       │          │    src/features/     │  (Feature Engineering & Store)
           │ (17 EDA Analyzers)   │          └──────────┬───────────┘
           └──────────────────────┘                     │
                                                        ▼
                                             ┌──────────────────────┐
                                             │     src/models/      │  (LightGBM, XGBoost, CatBoost, GNN)
                                             └──────────┬───────────┘
                                                        │
                         ┌──────────────────────────────┼──────────────────────────────┐
                         ▼                              ▼                              ▼
             ┌──────────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
             │   src/optimization/  │       │   src/evaluation/    │       │ src/explainability/  │
             │   (Optuna HPO)       │       │ (Financial Matrix)   │       │ (SHAP & Fairness)    │
             └──────────────────────┘       └──────────────────────┘       └──────────────────────┘
                         │                              │                              │
                         └──────────────────────────────┼──────────────────────────────┘
                                                        ▼
                                            ┌──────────────────────┐
                                            │   src/monitoring/    │  (MLflow & PSI Drift Engine)
                                            └──────────┬───────────┘
                                                       │
                                                       ▼
                                            ┌──────────────────────┐
                                            │   src/deployment/    │  (FastAPI Sub-10ms Serving)
                                            └──────────────────────┘
```

---

## 📦 Sub-Package Directory Index

| Directory | Core Purpose | Key Modules |
| :--- | :--- | :--- |
| [`src/data/`](file:///home/logan78/efforts/projects/ongoing/fraud-detection-optimization/src/data) | Data ingestion, memory optimization, type downcasting & cleaning | `ingestion.py`, `cleaning.py`, `pipeline.py` |
| [`src/eda/`](file:///home/logan78/efforts/projects/ongoing/fraud-detection-optimization/src/eda) | 17 Automated EDA Analyzers & HTML Report Generator | `profiling.py`, `target.py`, `drift.py`, `leakage.py` |
| [`src/features/`](file:///home/logan78/efforts/projects/ongoing/fraud-detection-optimization/src/features) | Enterprise Feature Store & Feature Selection Engine | `feature_store.py`, `feature_selection.py`, `encodings.py` |
| [`src/models/`](file:///home/logan78/efforts/projects/ongoing/fraud-detection-optimization/src/models) | Candidate Models, Stacking, Deep Learning, GNN & RL | `lightgbm_model.py`, `ensemble.py`, `advanced_ai.py` |
| [`src/optimization/`](file:///home/logan78/efforts/projects/ongoing/fraud-detection-optimization/src/optimization) | Optuna Bayesian Hyperparameter Optimization & Pruning | `bayesian.py`, `optuna_framework.py`, `pruning.py` |
| [`src/evaluation/`](file:///home/logan78/efforts/projects/ongoing/fraud-detection-optimization/src/evaluation) | Metrics, Calibration, Diagnostics & Financial Net Savings Matrix | `framework.py`, `metrics.py`, `business.py` |
| [`src/explainability/`](file:///home/logan78/efforts/projects/ongoing/fraud-detection-optimization/src/explainability) | SHAP Global/Local, PDP, ICE & Disparate Impact Fairness | `shap_engine.py`, `transparency.py` |
| [`src/monitoring/`](file:///home/logan78/efforts/projects/ongoing/fraud-detection-optimization/src/monitoring) | MLflow Tracking, PSI Drift, Service SLA & Auto-Retraining | `experiment_tracker.py`, `drift_engine.py`, `alerting.py` |
| [`src/deployment/`](file:///home/logan78/efforts/projects/ongoing/fraud-detection-optimization/src/deployment) | Production FastAPI REST Microservice & Batch Scoring Engine | `app.py`, `inference.py`, `engine.py` |
| [`src/pipelines/`](file:///home/logan78/efforts/projects/ongoing/fraud-detection-optimization/src/pipelines) | DVC Stage Execution Command Runners | `run_data_pipeline.py`, `run_deployment.py` |
| [`src/utils/`](file:///home/logan78/efforts/projects/ongoing/fraud-detection-optimization/src/utils) | QA Framework, CI/CD Automation, Security & Governance | `security_framework.py`, `testing_framework.py` |

---

## 🛠️ Code Conventions & Design Principles

1. **Object-Oriented & Functional Synergy**: Core algorithms are encapsulated in clean, stateless classes with type annotations.
2. **Pre-Execution Gates**: Every major pipeline module initiates a verification gate asserting inputs before execution.
3. **Immutability & Safety**: Intermediate transformations never mutate input dataframes in place.
