# ADR 001: Overall System Architecture and Pipeline Framework

* **Status**: Accepted
* **Date**: 2026-08-06
* **Deciders**: Antigravity Lead Architect

---

## CONTEXT
We needed a scalable, reproducible, and production-grade Machine Learning architecture to handle financial fraud detection on the IEEE-CIS dataset containing over 500,000 transactions and 400+ raw features under extreme class imbalance (~3.5% fraud rate).

---

## DECISION
We decided to build a modular Python 3.10 framework backed by DVC (Data Version Control) pipeline orchestration, LightGBM/XGBoost/CatBoost candidate models, Optuna hyperparameter optimization, MLflow experiment tracking, SHAP explainability, FastAPI microservice deployment, and MLOps monitoring with automated drift-based retraining triggers.

---

## CONSEQUENCES

### Positive
- Strict separation of concerns (Ingestion, EDA, Feature Store, Model Dev, Optimization, Evaluation, Explainability, Deployment, Monitoring, QA).
- Full reproducibility with deterministic seed locking and MLflow tracking.
- Sub-10ms real-time scoring SLA via FastAPI.
- Continuous quality assurance with 295+ automated unit tests.

### Negative
- High memory requirements (~16 GB RAM) when loading full parquet datasets.
