"""Unit tests for the DatasetProfiler and visualization metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.profiling import DatasetProfiler, classify_column_sdd


def test_classify_column_sdd() -> None:
    """Verifies classification maps to SDD Part 3.2 rules."""
    assert classify_column_sdd("isFraud") == "Target"
    assert classify_column_sdd("TransactionID") == "Identifier"
    assert classify_column_sdd("TransactionDT") == "Time"
    assert classify_column_sdd("card1") == "Card"
    assert classify_column_sdd("addr2") == "Address"
    assert classify_column_sdd("dist1") == "Distance"
    assert classify_column_sdd("P_emaildomain") == "Email"
    assert classify_column_sdd("DeviceType") == "Device"
    assert classify_column_sdd("id_01") == "Identity"
    assert classify_column_sdd("C12") == "Count"
    assert classify_column_sdd("D5") == "Delta"
    assert classify_column_sdd("M3") == "Match"
    assert classify_column_sdd("V120") == "Anonymous"
    assert classify_column_sdd("TransactionAmt") == "Transaction"


@pytest.fixture
def mock_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates small mock train and test sets with missing/duplicate structures."""
    train_data = {
        "TransactionID": [1, 2, 3, 4, 5],
        "isFraud": [0, 1, 0, 0, 0],
        "TransactionDT": [86400, 86450, 86500, 86600, 86700],
        "TransactionAmt": [100.0, 250.0, 50.0, 15.0, 100.0],
        "ProductCD": ["W", "H", "W", "W", "W"],
        "card1": [123, 456, 123, 789, 123],
        "dist1": [10.0, np.nan, 5.0, np.nan, 10.0],
        "P_emaildomain": ["gmail.com", "yahoo.com", "gmail.com", np.nan, "gmail.com"],
        "V1": [1.0, 1.0, 1.0, 1.0, 1.0],  # Constant
        "id_01": [-5.0, np.nan, -20.0, np.nan, np.nan],
        "DeviceType": ["desktop", "mobile", np.nan, np.nan, np.nan],
    }

    test_data = {
        "TransactionID": [6, 7, 8],
        "TransactionDT": [96400, 96500, 96600],
        "TransactionAmt": [200.0, 15.0, 30.0],
        "ProductCD": ["W", "W", "H"],
        "card1": [123, 123, 999],
        "dist1": [np.nan, 2.0, np.nan],
        "P_emaildomain": ["gmail.com", "gmail.com", "outlook.com"],
        "V1": [1.0, 1.0, 1.0],
        "id_01": [np.nan, -10.0, np.nan],
        "DeviceType": [np.nan, "mobile", np.nan],
    }

    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)

    # Convert object columns to categoricals
    for col in ["ProductCD", "P_emaildomain", "DeviceType"]:
        df_train[col] = df_train[col].astype("category")
        df_test[col] = df_test[col].astype("category")

    return df_train, df_test


def test_profiler_reports(
    mock_datasets: tuple[pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """Verifies all inventory, cardinality, stats, and completeness outputs."""
    df_train, df_test = mock_datasets
    profiler = DatasetProfiler(df_train, df_test, target_col="isFraud")

    # 1. Inventory & shape
    inventory = profiler.profile_inventory()
    assert inventory["train_rows"] == 5
    assert inventory["test_rows"] == 3
    assert inventory["fraud_samples"] == 1
    assert inventory["fraud_pct"] == 20.0
    # rows having id_01 or DeviceType populated
    assert inventory["train_identity_count"] == 3

    # 2. Memory checks
    _col_mem, mem_summary = profiler.profile_memory(tmp_path)
    assert (tmp_path / "memory_usage.png").exists()
    assert (tmp_path / "memory_profile.csv").exists()
    assert (tmp_path / "memory_summary.json").exists()
    assert mem_summary["total_mem_train_bytes"] > 0

    # 3. Cardinality checks
    df_card = profiler.analyze_cardinality(tmp_path)
    assert (tmp_path / "cardinality_histogram.png").exists()
    assert (tmp_path / "cardinality_report.csv").exists()
    assert (tmp_path / "high_cardinality_features.csv").exists()

    # V1 has unique count 1 -> Constant
    v1_card = df_card[df_card["column"] == "V1"]["classification"].values[0]
    assert v1_card == "Constant"

    # Card1 has unique count 3 -> Low Cardinality
    card1_card = df_card[df_card["column"] == "card1"]["classification"].values[0]
    assert card1_card == "Low Cardinality"

    # 4. Statistical properties
    df_num, df_cat = profiler.profile_statistics(tmp_path)
    assert (tmp_path / "numerical_statistics.csv").exists()
    assert (tmp_path / "categorical_statistics.csv").exists()

    # ProductCD is categorical
    assert "ProductCD" in df_cat["column"].tolist()
    # TransactionAmt is numerical
    assert "TransactionAmt" in df_num["column"].tolist()

    # Mean of TransactionAmt
    mean_val = df_num[df_num["column"] == "TransactionAmt"]["mean"].values[0]
    assert mean_val == 103.0

    # 5. Completeness checks
    df_comp, _comp_summary = profiler.profile_completeness(tmp_path)
    assert (tmp_path / "completeness_families.png").exists()
    assert (tmp_path / "completeness_report.csv").exists()
    assert (tmp_path / "completeness_summary.json").exists()

    # Identity completeness count (id_01 has 2 populated values out of 5 = 40%)
    id01_comp = df_comp[df_comp["column"] == "id_01"]["completeness_pct"].values[0]
    assert id01_comp == 40.0

    # 6. Recommendations check
    recs = profiler.generate_recommendations(df_card, df_comp)
    assert len(recs) > 0
    # Verifies recommendation categories
    categories = [r["category"] for r in recs]
    assert "Potential Removal" in categories
