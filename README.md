# IEEE-CIS Fraud Detection & Optimization Production System

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Code Quality](https://img.shields.io/badge/quality-A%2B-blue.svg)]()
[![Tests Pass Rate](https://img.shields.io/badge/tests-295%2F295%20(100%25)-success.svg)]()
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

Production-grade, scalable End-to-End Machine Learning System for **IEEE-CIS Financial Fraud Detection**, featuring automated data pipelines, enterprise feature store, multi-model ensemble candidates, hyperparameter optimization, model explainability, real-time FastAPI deployment, MLOps monitoring, and CI/CD quality gates.

---

## 🚀 Key Features & Architecture

```
[Raw Data] -> [Ingestion & Cleaning] -> [EDA Engine (17 Modules)] 
                     |
                     v
   [Feature Store & Feature Selection Engine]
                     |
                     v
 [Model Development: LightGBM / XGBoost / CatBoost] 
                     |
                     v
  [Bayesian Optuna Tuning & Advanced Evaluation] 
                     |
                     v
  [SHAP Explainability & MLOps MLflow Tracking] 
                     |
                     v
  [FastAPI Microservice & Sub-10ms Real-Time Engine] 
                     |
                     v
 [MLOps Monitoring, Alerting & Auto-Retraining]
```

---

## 🛠️ Quickstart

### 1. Installation & Environment Setup
```bash
# Clone repository
git clone https://github.com/6sLOGAN78/fraud-detection-optimization.git
cd fraud-detection-optimization

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full DVC Pipeline
```bash
dvc repro
```

### 3. Run FastAPI Real-Time Scoring Microservice
```bash
uvicorn src.deployment.app:app --host 0.0.0.0 --port 8000
```

### 4. Test API Health & Prediction Endpoints
```bash
curl -X GET http://localhost:8000/health
curl -X POST http://localhost:8000/v1/predict -H "Content-Type: application/json" -d '{"TransactionAmt": 150.0, "card1": 1000, "card2": 500}'
```

---

## 📊 Complete Pipeline Overview

1. **Part 1 & 2**: Data Ingestion, Cleaning & Memory Downcasting
2. **Part 3**: Automated EDA Suite (17 Comprehensive Analyzers & HTML Reports)
3. **Part 4**: Enterprise Feature Store & Transformation Pipeline
4. **Part 5**: Multi-Strategy Feature Selection Engine (SHAP, Mutual Info, Boruta, RFE, Annealing)
5. **Part 6**: Candidate Model Architecture (LightGBM, XGBoost, CatBoost, Blended Ensembles)
6. **Part 7**: Hyperparameter Optimization (Optuna TPE/CMA-ES Samplers & Pruning)
7. **Part 8**: Advanced Evaluation Framework (ROC-AUC, PR-AUC, ECE Calibration, Financial Savings Matrix)
8. **Part 9**: Explainability Framework (SHAP Global/Local, PDP, ICE, Disparate Impact Fairness)
9. **Part 10**: MLOps Experiment Tracking & Model Registry (MLflow & SHA256 Data Versioning)
10. **Part 11**: Real-Time & Batch Deployment Engine (FastAPI & Sub-10ms Latency SLA)
11. **Part 12**: Production MLOps Monitoring (PSI Drift, Performance Decay, Auto-Retraining)
12. **Part 13**: Enterprise QA Strategy & Release Quality Gate
13. **Part 14**: CI/CD Automation & Security Scanning (`.github/workflows/ci_cd.yml`)
14. **Part 15**: System Documentation & Architecture Decision Records (ADRs)

---

## 🧪 Unit Test Suite
Run the full 295+ unit test suite:
```bash
pytest tests/
```

---

## 📄 License
This project is licensed under the MIT License - see the `LICENSE` file for details.
