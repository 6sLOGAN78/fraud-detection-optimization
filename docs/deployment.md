# Model Deployment

This document describes options and commands designed for deployment serving.

## Serving Modes

1. **Real-time API Service**:
   - A FastAPI application exposed at `src/deployment/api.py`.
   - Listens for post payloads containing transaction specs, parses features, and yields fraud probabilities.
   - Run via docker-compose or directly:
     ```bash
     uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000
     ```

2. **Batch Inference Pipeline**:
   - An offline score generator exposed in `src/deployment/batch.py`.
   - Scores transaction lists from data tables (e.g. Parquet) and writes predictions to SQLite or file stores.
