# Installation and Setup Guide

This guide details environment setup, dependency management, and system requirements for running the IEEE-CIS Fraud Detection & Optimization system.

---

## 💻 Prerequisites

* **Operating System**: Linux (Ubuntu 20.04+ / Debian / Fedora) or macOS (12.0+)
* **Python**: Python 3.10+
* **RAM**: 16 GB minimum (32 GB recommended for full dataset execution)
* **Disk Space**: 10 GB free space for interim parquets and MLflow artifacts

---

## 🛠️ Step-by-Step Installation

### 1. Clone Repository
```bash
git clone https://github.com/6sLOGAN78/fraud-detection-optimization.git
cd fraud-detection-optimization
```

### 2. Configure Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade Core Build Tools
```bash
pip install --upgrade pip setuptools wheel
```

### 4. Install Project Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Verifying Installation

Run the complete test suite to confirm all system dependencies are working:
```bash
pytest tests/ -v
```
Expected output: **295 passed**.
