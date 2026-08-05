# MLOps Monitoring & Alerting Subsystem (`src/monitoring/`)

The `src/monitoring/` package manages experiment tracking with MLflow, population stability index (PSI) feature drift monitoring, service SLA tracking, security alerts, and automated continuous retraining.

---

## 📊 Drift Detection & Data Integrity Visuals

### Kolmogorov-Smirnov (KS) Feature Drift Shift Plots
The monitoring engine tracks feature distribution stability over time using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests to detect concept or covariate drift before accuracy decays.

![KS Drift Shifts](../../reports/eda/drift/plots/ks_drift_shifts.png)

---

## 📈 Monitoring & Retraining Flow

```
[Production Predictions] ──► [PSI Data Drift Monitor] ──► [Alerting Engine]
                                     │                        │
                                     ▼                        ▼
                          [Performance Decay Check] ──► [Auto-Retraining Pipeline]
```

---

## 🛠️ Key Components

1. **`experiment_tracker.py`**: MLflow tracking integration logging parameters, metrics, artifacts, and dataset SHA256 hashes.
2. **`drift_engine.py`**: `DataDriftMonitor` (PSI & KS tests per feature), `ConceptDriftMonitor`, `PredictionDistributionMonitor`, `FeatureHealthMonitor`.
3. **`service_monitor.py`**: SLA latency percentiles (`p50`, `p95`, `p99`), throughput QPS, and error rate tracking.
4. **`alerting.py`**: `AlertingEngine` logging INFO, WARNING, and CRITICAL security alerts.
5. **`lifecycle.py`**: `ChampionChallengerLifecycleManager` (shadow testing & promotion) and `AutomatedRetrainingPipeline`.
