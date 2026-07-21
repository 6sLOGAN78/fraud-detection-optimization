"""Unit tests for Target Variable Analysis module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.target import TargetVariableAnalyzer, classify_imbalance_severity


def test_classify_imbalance_severity() -> None:
    """Tests the severity categorization method."""
    assert classify_imbalance_severity(2.0) == "Low"
    assert classify_imbalance_severity(5.0) == "Moderate"
    assert classify_imbalance_severity(12.0) == "Moderate"
    assert classify_imbalance_severity(20.0) == "Moderate"
    assert classify_imbalance_severity(21.0) == "Severe"


def test_target_variable_analyzer_calculations(tmp_path: Path) -> None:
    """Tests analyzer engine metric correctness using a mock dataset."""
    # Construct mock dataset
    np.random.seed(42)
    n_samples = 100

    # 5% fraud, 95% legit
    is_fraud = [1] * 5 + [0] * 95
    tx_amt = np.random.exponential(scale=100.0, size=n_samples)
    tx_dt = np.linspace(1000, 900000, n_samples)  # Spans multiple days/weeks

    # Add identity cols
    id_01 = np.random.choice([-10.0, -5.0, 0.0, np.nan], size=n_samples)
    device_type = np.random.choice(["mobile", "desktop", None], size=n_samples)

    df_train = pd.DataFrame({
        "TransactionID": range(n_samples),
        "isFraud": is_fraud,
        "TransactionAmt": tx_amt,
        "TransactionDT": tx_dt,
        "id_01": id_01,
        "DeviceType": device_type,
    })

    df_test = pd.DataFrame({
        "TransactionID": range(100, 100 + n_samples),
        "TransactionAmt": tx_amt,
        "TransactionDT": tx_dt + 1000000,
        "id_01": id_01,
        "DeviceType": device_type,
    })

    analyzer = TargetVariableAnalyzer(
        df_train=df_train,
        df_test=df_test,
    )

    # Verify column classifications
    assert "id_01" in analyzer.identity_cols
    assert "DeviceType" in analyzer.identity_cols

    # Run analysis
    analyzer.analyze_all(tmp_path)

    # 1. Verify files exist
    assert (tmp_path / "fraud_distribution.csv").exists()
    assert (tmp_path / "fraud_distribution.json").exists()
    assert (tmp_path / "class_imbalance_report.csv").exists()
    assert (tmp_path / "fraud_rate_summary.csv").exists()
    assert (tmp_path / "fraud_by_time.csv").exists()
    assert (tmp_path / "fraud_identity_analysis.csv").exists()
    assert (tmp_path / "fraud_amount_analysis.csv").exists()
    assert (tmp_path / "fraud_statistics.csv").exists()
    assert (tmp_path / "target_analysis.json").exists()
    assert (tmp_path / "target_analysis_report.html").exists()

    # Verify visual assets saved
    assert (tmp_path / "fraud_distribution_plot.png").exists()
    assert (tmp_path / "fraud_by_time_plot.png").exists()
    assert (tmp_path / "fraud_identity_plot.png").exists()
    assert (tmp_path / "fraud_amount_plot.png").exists()

    # 2. Check metrics math
    with (tmp_path / "target_analysis.json").open(encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["total_transactions"] == 100
    assert summary["fraud_count"] == 5
    assert summary["legit_count"] == 95
    assert pytest.approx(summary["fraud_pct"]) == 5.0
    assert summary["imbalance_ratio"] == 95 / 5
    assert summary["imbalance_severity"] == "Moderate"

    df_dist = pd.read_csv(tmp_path / "fraud_distribution.csv")
    assert len(df_dist) == 3  # Total, Fraud, Legitimate
    assert df_dist.loc[df_dist["class"] == "Total", "count"].values[0] == 100

    df_imb = pd.read_csv(tmp_path / "class_imbalance_report.csv")
    assert df_imb["severity"].values[0] == "Moderate"
    assert df_imb["class_ratio"].values[0] == "19.00:1"

    # Verify time conversion mapping ranges
    df_time = pd.read_csv(tmp_path / "fraud_by_time.csv")
    assert "time_bucket" in df_time.columns
    assert set(df_time["time_bucket"].unique()) == {"hour", "day", "week"}

    # Verify recommendations structure
    recs = summary["recommendations"]
    assert "evaluation_metrics" in recs
    assert "cross_validation_strategy" in recs
    assert "imbalance_mitigation" in recs
    assert "feature_engineering_suggestions" in recs
