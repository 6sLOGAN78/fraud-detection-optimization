"""Unit tests for Missing Features components."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.missing import (
    VectorizedMissingEngine,
    AutomaticMissingDiscoveryEngine,
    MissingPatternBuilder,
    MissingValidationGate,
    MissingRegistry,
)


def test_vectorized_missing_engine() -> None:
    engine = VectorizedMissingEngine()
    
    df = pd.DataFrame({
        "col_a": [1.0, np.nan, 3.0, 4.0],
        "col_b": [2.0, np.nan, np.nan, 8.0],
    })
    
    # 1. Indicators
    df_ind = engine.compute_indicators(df, ["col_a", "col_b"])
    assert df_ind["col_a_isna"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert df_ind["col_b_isna"].tolist() == [0.0, 1.0, 1.0, 0.0]
    
    # 2. Row stats
    df_stats = engine.compute_row_stats(df, ["col_a", "col_b"])
    assert df_stats["missing_count"].tolist() == [0.0, 2.0, 1.0, 0.0]
    assert df_stats["missing_ratio"].tolist() == [0.0, 1.0, 0.5, 0.0]
    assert df_stats["completeness_score"].tolist() == [1.0, 0.0, 0.5, 1.0]


def test_missing_pattern_builder() -> None:
    builder = MissingPatternBuilder()
    
    df = pd.DataFrame({
        "id_01": [1.0, np.nan, 3.0, np.nan],
        "id_02": [2.0, 4.0, np.nan, np.nan],
    })
    
    hashes = builder.build_pattern_hashes(df, ["id_01", "id_02"])
    
    # Row 0: no missing -> 0
    # Row 1: id_01 is missing -> 1 * 2^0 + 0 * 2^1 = 1
    # Row 2: id_02 is missing -> 0 * 2^0 + 1 * 2^1 = 2
    # Row 3: both missing -> 1 * 2^0 + 1 * 2^1 = 3
    assert hashes.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_automatic_missing_discovery_engine() -> None:
    df = pd.DataFrame({
        "col_valid": [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2.0],  # 80% missingness (valid)
        "col_mostly_present": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, np.nan],            # 10% missingness (valid)
        "col_too_present": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],                  # 0% missingness (invalid)
    })
    
    engine = AutomaticMissingDiscoveryEngine(min_missing_ratio=0.05, max_missing_ratio=0.95)
    cols = engine.discover_missing_columns(df)
    
    assert "col_valid" in cols
    assert "col_mostly_present" in cols
    assert "col_too_present" not in cols


def test_missing_validation_gate() -> None:
    df = pd.DataFrame({
        "missing_count": [1.0, 2.0, 1.0],
        "constant_col": [1.0, 1.0, 1.0],
    })
    
    gate = MissingValidationGate()
    report = gate.validate(df)
    
    assert report["constant_columns_count"] == 1
    assert report["status"] == "PASS"  # validation outputs PASS unless there is infs, nans, or duplicates in headers


def test_missing_registry(tmp_path: Path) -> None:
    reg = MissingRegistry()
    reg.register("col_a_isna", "col_a", "binary_indicator")
    
    manifest_path, csv_path = reg.save_catalog(tmp_path)
    assert manifest_path.exists()
    assert csv_path.exists()
    
    with open(manifest_path) as f:
        meta = json.load(f)
    assert meta["registry"][0]["feature_name"] == "col_a_isna"
