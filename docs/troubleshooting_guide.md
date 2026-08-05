# Operations and Troubleshooting Guide

Guide for diagnosing errors, handling high latency, and responding to drift alerts.

---

## 🚨 Common Issues & Resolution

### 1. High Prediction Latency (>50ms SLA)
* **Symptom**: `p95_latency_ms` exceeds SLA threshold.
* **Resolution**: Check CPU utilization and increase worker thread count in `uvicorn` or scale Kubernetes pods.

### 2. Data Drift Alert Triggered (`DataDriftDetected`)
* **Symptom**: PSI >= 0.2 on features.
* **Resolution**: Run `python3 -m src.pipelines.run_mlops_monitoring` to verify feature drift list and evaluate retraining trigger.

### 3. MLflow File Store Lock Warning
* **Symptom**: `MLflow filestore warning`.
* **Resolution**: Ensure `os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"` is set.
