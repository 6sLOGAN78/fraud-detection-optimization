# API Documentation & REST Interface Spec

OpenAPI 3.0 REST endpoints specification for the Fraud Detection API.

---

## 1. Health Check Endpoint

`GET /health`

### Response (200 OK)
```json
{
  "status": "HEALTHY",
  "timestamp": "2026-08-06 01:00:00"
}
```

---

## 2. Real-Time Predict Endpoint

`POST /v1/predict`

### Request Body
```json
{
  "TransactionAmt": 150.0,
  "card1": 13926,
  "card2": 150.0,
  "extra_features": {}
}
```

### Response (200 OK)
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0215,
  "decision_threshold": 0.5,
  "latency_ms": 3.42,
  "status": "APPROVED",
  "version": "v1"
}
```
