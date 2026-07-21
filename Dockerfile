# Base image with Python 3.12
FROM python:3.12-slim AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and config codes
COPY src/ ./src/
COPY configs/ ./configs/
COPY Makefile .
COPY pyproject.toml .

# Target for API deployment
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Target for script training/batch executions
FROM base AS runner
ENTRYPOINT ["python"]
