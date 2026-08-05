# Data Ingestion & Preprocessing Subsystem (`src/data/`)

The `src/data/` package handles dataset loading, vectorized memory optimization, table merging, missing value handling, and schema validation for the IEEE-CIS Fraud Detection dataset.

---

## 🔄 Processing Architecture

```
[train_transaction.csv] ──┐
                          ├──► [Memory Downcasting & Merging] ──► [Quality Verification] ──► [train_cleaned.parquet]
[train_identity.csv]    ──┘
```

---

## 🛠️ Key Modules

* **`ingestion.py`**: Loads raw CSV files with schema validation.
* **`cleaning.py`**: Executes float64 -> float32 and int64 -> int32 downcasting (reducing RAM memory usage by **~65%**).
* **`pipeline.py`**: Orchestrates the full ingestion pipeline saving outputs to `data/interim/train_cleaned.parquet`.

---

## 📊 Performance Statistics
* **Raw Dataset Size**: ~1.2 GB CSV
* **Optimized Parquet Size**: ~350 MB Parquet
* **Memory Reduction**: From 2.4 GB in RAM down to ~800 MB in RAM.
