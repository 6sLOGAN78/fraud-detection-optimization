"""Unit tests for the DataQualityAssessor engine classes and algorithms."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.quality import DataQualityAssessor


@pytest.fixture
def sample_quality_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates synthetic training and testing data for tests."""
    # Contains 1 duplicate TransactionID and a negative Amt
    train_dict = {
        "TransactionID": [
            1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11
        ],
        "isFraud": [
            0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0
        ],
        "TransactionAmt": [
            100.0, 50.0, -10.0, 20.0, 30.0, 30.0,
            40.0, 50.0, 60.0, 70.0, 80.0, 90.0
        ],
        "ProductCD": [
            "W", "W", "H", "W", "W", "W", "W", "W", "W", "W", "W", "W"
        ],
        "card1": [12345] * 12,
        "card2": [
            10.0, 10.0, 10.0, 10.0, 10.0, 15.0,
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0
        ],
        "dist1": [
            1.0, 2.0, np.nan, 4.0, 5.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0, 11.0
        ],
        "dist2": [
            np.inf, 2.0, 3.0, 4.0, 5.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0, 11.0
        ],
        "P_emaildomain": [
            "gmail.com", "nan", "yahoo.com", "gmail.com", "gmail.com",
            "gmail.com", "gmail.com", "gmail.com", "gmail.com",
            "gmail.com", "gmail.com", "gmail.com"
        ],
        "C1": [
            1.0, 2.0, 100.0, 3.0, 4.5, 5.0,
            6.0, 7.0, 8.0, 9.0, 11.0, 12.0
        ],
    }

    test_dict = {
        "TransactionID": [7, 8, 9, 10, 11],
        "TransactionAmt": [150.0, 60.0, 110.0, 25.0, 35.0],
        "ProductCD": ["W", "W", "H", "W", "W"],
        "card1": [12345, 12345, 12345, 12345, 12345],
        "card2": [10.0, 10.0, 10.0, 10.0, 10.0],
        "dist1": [1.0, 2.0, 3.0, np.nan, 5.0],
        "dist2": [1.0, 2.0, 3.0, 4.0, 5.0],
        "P_emaildomain": [
            "gmail.com", "gmail.com", "yahoo.com", "gmail.com", "gmail.com"
        ],
        "C1": [1.0, 2.0, 3.0, 4.0, 5.0],
    }

    df_train = pd.DataFrame(train_dict)
    df_test = pd.DataFrame(test_dict)
    return df_train, df_test


def test_quality_assessor_missingness(
        sample_quality_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Verifies missing values audit counts and percentages."""
    df_train, df_test = sample_quality_data
    assessor = DataQualityAssessor(df_train, df_test)

    df_missing, _ = assessor.audit_missingness(tmp_path)
    assert not df_missing.empty
    assert (tmp_path / "missing_summary.csv").exists()
    assert (tmp_path / "missing_summary.json").exists()

    # Check dist1 missing percent in train (1 out of 12 is missing ~ 8.33%)
    dist_row = df_missing[df_missing["column"] == "dist1"].iloc[0]
    assert np.isclose(dist_row["missing_pct_train"], 8.3333, atol=1e-2)


def test_quality_assessor_duplicates(
        sample_quality_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Verifies duplicate checking logic and identifier unique audits."""
    df_train, df_test = sample_quality_data
    assessor = DataQualityAssessor(df_train, df_test)

    df_dup = assessor.audit_duplicates(tmp_path)
    assert (tmp_path / "duplicate_report.csv").exists()

    # Check duplicate TransactionID (1 dup)
    metric_str = "Duplicate TransactionID (Train)"
    dup_id_val = df_dup[
        df_dup["Metric"] == metric_str
    ]["Value"].iloc[0]
    assert dup_id_val == 1


def test_quality_assessor_constants(
        sample_quality_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Verifies constant and near-constant feature recognition logic."""
    df_train, df_test = sample_quality_data
    assessor = DataQualityAssessor(df_train, df_test)

    # Constant
    df_const = assessor.detect_constant_features(tmp_path)
    assert "card1" in df_const["column"].tolist()

    # Near Constant
    df_near = assessor.detect_near_constant_features(tmp_path, threshold=0.8)
    assert "card2" in df_near["column"].tolist()


def test_quality_assessor_anomalies(
        sample_quality_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Verifies negative numbers, corrupted null strings, and infinite val check."""
    df_train, df_test = sample_quality_data
    assessor = DataQualityAssessor(df_train, df_test)

    # Invalid values (negative TransactionAmt and 'nan' email string)
    df_invalid = assessor.detect_invalid_values(tmp_path)
    assert "TransactionAmt" in df_invalid["column"].tolist()
    assert "P_emaildomain" in df_invalid["column"].tolist()

    # Infinite
    df_inf = assessor.detect_infinite_values(tmp_path)
    assert "dist2" in df_inf["column"].tolist()
    assert df_inf[df_inf["column"] == "dist2"]["infinite_count"].iloc[0] == 1


def test_quality_assessor_outliers(
        sample_quality_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Verifies outlier detection calculations."""
    df_train, df_test = sample_quality_data
    assessor = DataQualityAssessor(df_train, df_test)

    df_out = assessor.assess_outliers(tmp_path)
    assert "C1" in df_out["column"].tolist()
    # C1 has outlier 100.0 which exceeds 1.5 * IQR
    c1_row = df_out[df_out["column"] == "C1"].iloc[0]
    assert c1_row["outliers_iqr_count"] >= 1


def test_quality_assessor_consistency(
        sample_quality_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Verifies schema consistency, type checks, and target checks."""
    df_train, df_test = sample_quality_data
    assessor = DataQualityAssessor(df_train, df_test)

    reports = assessor.validate_consistency(tmp_path)
    assert reports["target_checks"]["is_target_in_train"] is True
    assert reports["target_checks"]["is_target_in_test"] is False


def test_quality_assessor_score(
        sample_quality_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Verifies score weights and boundary criteria range (0 to 100)."""
    df_train, df_test = sample_quality_data
    assessor = DataQualityAssessor(df_train, df_test)

    scoring_metrics = {
        "missing_pct_train": 5.0,
        "duplicate_trans_pct_train": 2.0,
        "invalid_count": 1,
        "constant_pct": 10.0,
        "near_constant_pct": 10.0,
        "outlier_pct": 5.0,
        "schema_mismatches": 0,
        "type_mismatches": 0,
    }

    summary = assessor.compute_quality_score(tmp_path, scoring_metrics)
    score = summary["overall_data_quality_score"]
    assert 0.0 <= score <= 100.0
    assert isinstance(summary["rating"], str)
