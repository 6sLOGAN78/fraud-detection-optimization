# Project Structure Specification

```
fraud-detection-optimization/
├── .github/
│   └── workflows/
│       └── ci_cd.yml                 # Production GitHub Actions CI/CD Pipeline
├── artifacts/                        # Model artifacts & deployment packages
├── configs/                          # Experiment configuration YAML/JSON files
├── data/                             # Data directory (ignored in git)
│   ├── raw/                          # Raw IEEE-CIS CSV files
│   ├── interim/                      # Cleaned parquet files
│   └── processed/                    # Feature store parquets
├── docs/                             # Technical documentation suite & ADRs
│   ├── adrs/                         # Architecture Decision Records
│   ├── data_dictionary.md            # Raw & Interim data schemas
│   ├── feature_dictionary.md         # Engineered feature definitions
│   └── ...                           # Guides
├── logs/                             # System logs & monitoring output
├── mlruns/                           # MLflow experiment tracking database
├── reports/                          # Generated HTML/JSON reports
├── src/                              # Main application source code
│   ├── data/                         # Data ingestion & cleaning
│   ├── eda/                          # Automated EDA analyzers
│   ├── features/                     # Feature engineering & Feature Store
│   ├── models/                       # Candidate models & evaluation
│   ├── optimization/                 # Optuna hyperparameter optimization
│   ├── evaluation/                   # Advanced metrics & financial matrix
│   ├── explainability/               # SHAP & transparency engines
│   ├── monitoring/                   # MLOps tracking, drift & alerting
│   ├── deployment/                   # Real-time FastAPI & batch inference
│   ├── pipelines/                    # Pipeline runners
│   └── utils/                        # Utilities, QA & CI/CD automation
├── tests/                            # Unit & Integration test suite (295+ tests)
├── dvc.yaml                          # Complete DVC Data Science Pipeline
├── Dockerfile                        # Production container build script
├── docker-compose.yml                # Microservice orchestration spec
├── README.md                         # Main system README
└── requirements.txt                  # Python dependencies
```
