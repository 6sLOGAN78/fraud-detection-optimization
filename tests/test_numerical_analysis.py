# ruff: noqa: E501
"""Unit tests for the Numerical Feature Analysis module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eda.numerical import (
    NumericalFeatureAnalyzer,
    classify_kurtosis,
    classify_skewness,
)


def test_skewness_classification() -> None:
    """Verifies that skewness value classification bins match standard descriptions."""
    assert classify_skewness(0.1) == "Approximately Symmetric"
    assert classify_skewness(0.6) == "Moderately Right-Skewed"
    assert classify_skewness(-0.8) == "Moderately Left-Skewed"
    assert classify_skewness(1.5) == "Highly Right-Skewed"
    assert classify_skewness(-1.5) == "Highly Left-Skewed"
    assert classify_skewness(float("nan")) == "N/A"


def test_kurtosis_classification() -> None:
    """Verifies that kurtosis tails match categorization boundaries."""
    # pandas kurtosis computes excess kurtosis (kurtosis - 3)
    assert classify_kurtosis(0.1) == "Normal"
    assert classify_kurtosis(5.0) == "Heavy-Tailed"
    assert classify_kurtosis(-2.0) == "Light-Tailed"
    assert classify_kurtosis(float("nan")) == "N/A"


def test_numerical_analyzer_execution(tmp_path: Path) -> None:
    """Tests analyzer engine correctness using small mock datasets."""
    # Design mock dataset
    np.random.seed(42)
    n_samples = 200

    # 1. Symmetric Normal feature
    feat_norm = np.random.normal(loc=10.0, scale=2.0, size=n_samples)

    # 2. Right Skewed positive feature (log1p recommendation candidate)
    feat_skewed = np.random.exponential(scale=50.0, size=n_samples)

    # 3. Target label
    is_fraud = np.random.choice([0, 1], p=[0.95, 0.05], size=n_samples)

    df_train = pd.DataFrame({
        "TransactionID": range(n_samples),
        "isFraud": is_fraud,
        "TransactionDT": range(1000, 1000 + n_samples),
        "feat_norm": feat_norm,
        "feat_skewed": feat_skewed,
    })

    # Include outlier in norm to verify outlier detection logic
    df_train.loc[0, "feat_norm"] = 100.0  # Large outlier

    df_test = pd.DataFrame({
        "TransactionID": range(n_samples, 2 * n_samples),
        "TransactionDT": range(2000, 2000 + n_samples),
        "feat_norm": feat_norm + 1.0,
        "feat_skewed": feat_skewed,
    })

    # Initialize analyzer
    analyzer = NumericalFeatureAnalyzer(
        df_train=df_train,
        df_test=df_test,
        target_col="isFraud",
    )

    # Validate automatic column classifications (excludes ID, DT, target)
    assert "feat_norm" in analyzer.numerical_cols
    assert "feat_skewed" in analyzer.numerical_cols
    assert "TransactionID" not in analyzer.numerical_cols
    assert "TransactionDT" not in analyzer.numerical_cols
    assert "isFraud" not in analyzer.numerical_cols

    # Customize plot columns specifically for testing so it matches dummy data
    analyzer.plot_cols = ["feat_norm", "feat_skewed"]

    # Run complete sweep
    analyzer.analyze_all(tmp_path)

    # Check generated files
    assert (tmp_path / "numerical_features.csv").exists()
    assert (tmp_path / "numerical_feature_summary.json").exists()
    assert (tmp_path / "distribution_statistics.csv").exists()
    assert (tmp_path / "outlier_analysis.csv").exists()
    assert (tmp_path / "outlier_summary.json").exists()
    assert (tmp_path / "skewness_report.csv").exists()
    assert (tmp_path / "kurtosis_report.csv").exists()
    assert (tmp_path / "transformation_recommendations.csv").exists()
    assert (tmp_path / "numerical_analysis_report.html").exists()
    assert (tmp_path / "numerical_analysis.json").exists()

    # Verify visual asset directories and file outputs
    assert (tmp_path / "kde_plots" / "feat_norm_kde.png").exists()
    assert (tmp_path / "histograms" / "feat_norm_hist.png").exists()
    assert (tmp_path / "boxplots" / "feat_norm_boxplot.png").exists()
    assert (tmp_path / "violin_plots" / "feat_norm_violin.png").exists()

    # Load summary state validation
    with (tmp_path / "numerical_analysis.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["total_numerical_features"] == 2
    assert summary["highly_skewed_count"] >= 1  # feat_skewed exceeds 1.0 skew, feat_norm might as well due to outlier
    assert summary["outlier_summary"]["total_iqr_outliers"] > 0

    # Verify recommendations logic details
    df_recs = pd.read_csv(tmp_path / "transformation_recommendations.csv")
    rec_skewed = df_recs[df_recs["feature"] == "feat_skewed"]
    rec_norm = df_recs[df_recs["feature"] == "feat_norm"]

    assert rec_skewed["suggested_transformation"].values[0] == "Log Transformation (log1p)"
    assert rec_norm["suggested_transformation"].values[0] in ["No Transformation", "Log Transformation (log1p)", "Robust Scaler (IQR-based)"]
