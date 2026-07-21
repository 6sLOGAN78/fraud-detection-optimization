"""Verification tests for the Data Engineering Pipeline components."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import IngestionError, validate_file
from src.data.schema import generate_schema, validate_schema
from src.monitoring.drift import calculate_ks_test, calculate_psi
from src.preprocessing.leakage import detect_target_leakage
from src.preprocessing.memory import optimize_memory
from src.preprocessing.merge import merge_datasets
from src.preprocessing.quality import drop_constant_columns, handle_infinite_values


def test_ingestion_validation(tmp_path: Path) -> None:
    """Tests delimiter and path availability checks in data ingestion."""
    dummy_file = tmp_path / "train.csv"

    # Non-existent file
    with pytest.raises(IngestionError):
        validate_file(dummy_file)

    # Invalid delimiter
    dummy_file.write_text("col1;col2\n1;2", encoding="utf-8")
    with pytest.raises(IngestionError):
        validate_file(dummy_file)

    # Valid delimiter
    dummy_file.write_text("col1,col2\n1,2", encoding="utf-8")
    validate_file(dummy_file, expected_cols=2)


def test_schema_mismatch() -> None:
    """Verifies that schema validator logs unregistered types or mismatch."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})
    schema = generate_schema(df)

    # Perfect check
    errors = validate_schema(df, schema)
    assert len(errors) == 0

    # Type mismatch check
    mismatch_df = pd.DataFrame({"col1": ["X", "Y", "Z"], "col2": ["A", "B", "C"]})
    errors = validate_schema(mismatch_df, schema)
    assert len(errors) > 0


def test_memory_optimization() -> None:
    """Verifies numeric downcasting shrinks memory footprint safely."""
    df = pd.DataFrame(
        {
            "int_col": [1, 5, 10, 100],  # Fits in int8
            "float_col": [1.5, 2.5, 3.5, 4.5],  # Fits in float32
        }
    )
    opt_df, rep = optimize_memory(df)
    assert opt_df["int_col"].dtype == np.int8
    assert opt_df["float_col"].dtype == np.float32
    assert rep["reduction_pct"] > 0


def test_left_merge() -> None:
    """Verifies Transaction + Identity join and has_identity indicator."""
    tx_df = pd.DataFrame({"TransactionID": [1, 2, 3], "Amt": [10.0, 20.0, 30.0]})
    id_df = pd.DataFrame({"TransactionID": [1, 3], "Device": ["PC", "Mobile"]})

    merged = merge_datasets(tx_df, id_df)
    assert len(merged) == 3
    assert list(merged["has_identity"]) == [1, 0, 1]
    assert list(merged["Device"].fillna("Missing")) == ["PC", "Missing", "Mobile"]


def test_quality_and_inf_treatment() -> None:
    """Tests that infinite values are cast to NaN and constants are dropped."""
    df = pd.DataFrame(
        {
            "inf_col": [1.0, np.inf, 3.0],
            "const_col": ["A", "A", "A"],
            "normal_col": [1, 2, 3],
        }
    )
    df = handle_infinite_values(df)
    assert df["inf_col"].isna().sum() == 1

    df, dropped = drop_constant_columns(df)
    assert "const_col" not in df.columns
    assert "const_col" in dropped


def test_drift_metrics() -> None:
    """Tests calculations of PSI and KS values under distinct distributions."""
    train_vals = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    test_vals = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    psi = calculate_psi(train_vals, test_vals)
    # Identical features have ~0 PSI
    assert psi < 0.1

    # KS check
    ks_stat, p_val = calculate_ks_test(train_vals, test_vals)
    assert ks_stat == 0.0
    assert p_val == 1.0


def test_leakage_detection() -> None:
    """Verifies that high correlation targets are flagged as leakage."""
    df = pd.DataFrame(
        {
            "isFraud": [0, 0, 1, 1],
            "leak_col": [0.0, 0.0, 1.0, 1.0],  # Perfect correlation
            "safe_col": [0.2, 0.5, 0.1, 0.9],
        }
    )
    leaked = detect_target_leakage(df, target_col="isFraud", threshold=0.95)
    assert "leak_col" in leaked
    assert "safe_col" not in leaked
