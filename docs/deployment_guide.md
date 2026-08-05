# Production Deployment Guide

Guide for serving real-time and batch prediction engines using FastAPI and Docker containers.

---

## ⚡ FastAPI Production Web Service

Run locally using uvicorn:
```bash
uvicorn src.deployment.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Endpoints:
- `GET /health`: Liveness probe for Kubernetes
- `GET /v1/model_info`: Model version & feature metadata
- `POST /v1/predict`: Real-time single transaction scoring (<10ms SLA)
- `POST /v1/predict_batch`: High-throughput batch scoring
