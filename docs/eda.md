# Exploratory Data Analysis (EDA)

This document describes the automated EDA pipeline and reports.

## Core Operations

Our EDA script runs automatically over data partitions, recording key results in `reports/eda/`:
1. **Missing Quality Scans**: Scans columns for null values. Low-information columns (missing rate above 90%) are mapped to clean target cuts.
2. **Cardinality Verification**: Maps categorical distributions, capturing skewness and card levels.
3. **Data Quality Profile**: Evaluates infinite values and duplicate rows.
4. **Drift Detection**: Computes target distribution shifts or column drift metrics between train and test datasets.

## Output Targets

- `reports/eda/drift_report.json`: Drift indicators and Kolmogorov-Smirnov stats.
- `reports/eda/missing_report.json`: Per-feature count of missing entries.
