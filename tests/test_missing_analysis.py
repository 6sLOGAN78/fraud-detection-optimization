"""Unit tests for the MissingValueAnalyzer engine classes and algorithms."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.missing import MissingValueAnalyzer, classify_missing_family


@pytest.fixture
def sample_missing_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates synthetic datasets with controlled missingness patterns."""
    np.random.seed(42)
    n_rows = 100

    # Build columns belonging to different families
    data_train = {
        "TransactionID": list(range(1, n_rows + 1)),
        "TransactionDT": list(range(100, 100 + n_rows)),
        "isFraud": np.random.choice([0, 1], size=n_rows, p=[0.9, 0.1]),
        # card1: 10% missing
        "card1": [123 if i % 10 != 0 else None for i in range(n_rows)],
        # addr1: 20% missing
        "addr1": [456 if i % 5 != 0 else None for i in range(n_rows)],
        # dist1: 25% missing
        "dist1": [1.2 if i % 4 != 0 else None for i in range(n_rows)],
        # C1: 0% missing
        "C1": [1.0] * n_rows,
        # D1: 50% missing
        "D1": [10.0 if i % 2 == 0 else None for i in range(n_rows)],
        # M1: 50% missing (correlated with D1)
        "M1": ["F" if i % 2 == 0 else None for i in range(n_rows)],
        # V1: 5% missing
        "V1": [9.0 if i % 20 != 0 else None for i in range(n_rows)],
        # DeviceInfo: 50% missing
        "DeviceInfo": [
            "Desktop" if i % 2 == 0 else None for i in range(n_rows)
        ],
    }

    # High fraud when DeviceInfo is missing
    df_train = pd.DataFrame(data_train)
    df_train.loc[df_train["DeviceInfo"].isnull(), "isFraud"] = 1

    data_test = {
        "TransactionID": list(range(n_rows + 1, 2 * n_rows + 1)),
        "TransactionDT": list(range(200, 200 + n_rows)),
        "card1": [123 if i % 10 != 0 else None for i in range(n_rows)],
        # addr1: 0% missing (drift vs train!)
        "addr1": [456] * n_rows,
        "dist1": [1.2 if i % 4 != 0 else None for i in range(n_rows)],
        "C1": [1.0] * n_rows,
        "D1": [10.0 if i % 2 == 0 else None for i in range(n_rows)],
        "M1": ["F" if i % 2 == 0 else None for i in range(n_rows)],
        "V1": [9.0 if i % 20 != 0 else None for i in range(n_rows)],
        "DeviceInfo": ["Desktop" if i % 2 == 0 else None for i in range(n_rows)],
    }
    df_test = pd.DataFrame(data_test)

    return df_train, df_test


def test_classify_missing_family() -> None:
    """Verifies feature family mapping according to 11-family setup."""
    assert classify_missing_family("card2") == "Card"
    assert classify_missing_family("addr_state") == "Address"
    assert classify_missing_family("dist2") == "Distance"
    assert classify_missing_family("P_emaildomain") == "Email"
    assert classify_missing_family("DeviceInfo") == "Device"
    assert classify_missing_family("id_02") == "Identity"
    assert classify_missing_family("C10") == "C Features"
    assert classify_missing_family("D3") == "D Features"
    assert classify_missing_family("M4") == "M Features"
    assert classify_missing_family("V104") == "V Features"
    assert classify_missing_family("ProductCD") == "Transaction"


def test_missingness_percentages(
    sample_missing_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Validates missing calculation metrics and bucket categorizations."""
    df_train, df_test = sample_missing_data
    analyzer = MissingValueAnalyzer(df_train, df_test)

    df_pct = analyzer.analyze_missing_percentages(tmp_path)

    # Validate output files exist
    assert (tmp_path / "missing_percentage.csv").exists()
    assert (tmp_path / "missing_summary.json").exists()
    assert (tmp_path / "missing_percentage_bar.png").exists()

    # Validate statistics
    with (tmp_path / "missing_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["total_features"] == 8
    # C1 is complete
    assert summary["complete_count"] == 1
    assert "C1" in df_pct[df_pct["category_train"] == "Complete"]["column"].values


def test_missing_correlation(
    sample_missing_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Verifies Pearson/Phi missingness correlation and heatmap logging."""
    df_train, df_test = sample_missing_data
    analyzer = MissingValueAnalyzer(df_train, df_test)

    df_corr = analyzer.analyze_missing_correlations(tmp_path)

    # Validate output files exist
    assert (tmp_path / "missing_correlation.csv").exists()
    assert (tmp_path / "missing_correlation_heatmap.png").exists()

    # D1 and M1 have exact same missing indices (index % 2 != 0).
    # Their missing correlation coefficient should be exactly 1.0.
    pair = df_corr[
        ((df_corr["feature_1"] == "D1") & (df_corr["feature_2"] == "M1"))
        | ((df_corr["feature_1"] == "M1") & (df_corr["feature_2"] == "D1"))
    ]
    assert not pair.empty
    assert pytest.approx(pair["correlation"].iloc[0], abs=0.01) == 1.0


def test_missing_vs_fraud(
    sample_missing_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Validates target fraud rate difference and Relative Risk metrics."""
    df_train, df_test = sample_missing_data
    analyzer = MissingValueAnalyzer(df_train, df_test)

    df_fraud = analyzer.analyze_missing_vs_target(tmp_path)

    assert (tmp_path / "missing_vs_fraud.csv").exists()

    # DeviceInfo has missing rows set to fraud.
    dev_info = df_fraud[df_fraud["column"] == "DeviceInfo"].iloc[0]
    # Missing DeviceInfo => 100% fraud
    assert dev_info["missing_fraud_pct"] == 100.0
    assert dev_info["difference"] > 0


def test_missing_patterns(
    sample_missing_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Verifies missing pattern combinations, row counts, ratios, and summary files."""
    df_train, df_test = sample_missing_data
    analyzer = MissingValueAnalyzer(df_train, df_test)

    df_row, df_pat = analyzer.analyze_missing_patterns(tmp_path)

    assert (tmp_path / "row_missing_statistics.csv").exists()
    assert (tmp_path / "missing_patterns.csv").exists()

    # Test top row statistics
    assert len(df_row) == len(df_train)
    assert not df_pat.empty


def test_family_missingness(
    sample_missing_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Validates missing statistics grouped by the 11 feature families."""
    df_train, df_test = sample_missing_data
    analyzer = MissingValueAnalyzer(df_train, df_test)

    df_fam = analyzer.analyze_family_missingness(tmp_path)

    assert (tmp_path / "feature_family_missing.csv").exists()
    assert "Card" in df_fam["family"].values
    assert "Distance" in df_fam["family"].values


def test_train_test_missing_drift(
    sample_missing_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Checks drift warnings for train/test missing value discrepancies."""
    df_train, df_test = sample_missing_data
    analyzer = MissingValueAnalyzer(df_train, df_test)

    df_comp = analyzer.compare_train_test_missingness(tmp_path)

    assert (tmp_path / "train_test_missing_comparison.csv").exists()

    # addr1 is 20% missing in train, but 0% in test => 20% difference => drift_detected
    addr = df_comp[df_comp["column"] == "addr1"].iloc[0]
    assert addr["drift_detected"]
    assert addr["absolute_difference"] == 20.0


def test_missing_recommendations(
    sample_missing_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Validates rule-based handling strategy recommendation selections."""
    df_train, df_test = sample_missing_data
    analyzer = MissingValueAnalyzer(df_train, df_test)

    df_pct = analyzer.analyze_missing_percentages(tmp_path)
    df_recs = analyzer.generate_recommendations(tmp_path, df_pct)

    assert (tmp_path / "missing_recommendations.csv").exists()

    # D1 has 50% missing => Create Missing Indicator
    d1_rec = df_recs[df_recs["column"] == "D1"].iloc[0]
    assert d1_rec["recommendation"] == "Create Missing Indicator"

    # C1 has 0% missing => Keep
    c1_rec = df_recs[df_recs["column"] == "C1"].iloc[0]
    assert c1_rec["recommendation"] == "Keep"
