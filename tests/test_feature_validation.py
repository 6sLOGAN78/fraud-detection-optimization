"""Unit tests for Feature Validation components."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.validation import (
    SchemaValidator,
    MissingValueValidator,
    StatisticalValidator,
    FeatureValidationPipeline,
)


def test_schema_validator() -> None:
    validator = SchemaValidator()
    
    df = pd.DataFrame({
        "TransactionID": [100, 101, 102],
        "feature_a": [1.2, 3.4, 5.6],
        "feature_b": ["legit", "fraud", "legit"],
    })
    
    expected_types = {
        "TransactionID": "numeric",
        "feature_a": "numeric",
        "feature_b": "categorical",
        "feature_c": "numeric",  # missing
    }
    
    report = validator.validate(df, expected_types)
    assert report["missing_columns"] == ["feature_c"]
    assert report["dtype_mismatches_count"] == 0
    assert report["status"] == "FAIL"


def test_missing_value_validator() -> None:
    validator = MissingValueValidator()
    
    df = pd.DataFrame({
        "col_a": [1.0, np.nan, np.nan, np.nan],  # 75% missing
        "col_b": [2.0, np.inf, 4.0, 5.0],      # contains Inf
    })
    
    thresholds = {"col_a": 0.50}
    report = validator.validate(df, thresholds)
    
    assert "col_a" in report["high_missingness"]
    assert "col_b" in report["infinite_columns"]
    assert report["status"] == "WARN"


def test_statistical_validator() -> None:
    validator = StatisticalValidator()
    
    df = pd.DataFrame({
        "col_a": [1.0, 1.0, 1.0, 1.0],               # constant
        "col_b": [1.0, 2.0, 100.0, np.nan],         # out of bounds (bounds: 1.0 to 10.0)
        "col_c": [1.0, 1.1, 1.2, 10.0],             # outlier check
    })
    
    range_bounds = {"col_b": (1.0, 10.0)}
    report = validator.validate(df, range_bounds, outlier_sigma=1.2)
    
    assert "col_a" in report["constant_columns"]
    assert "col_b" in report["out_of_bounds"]
    assert "col_c" in report["outliers"]
    assert report["status"] == "WARN"


def test_validation_pipeline(tmp_path: Path) -> None:
    pipeline = FeatureValidationPipeline()
    
    df = pd.DataFrame({
        "TransactionID": [100, 101, 102],
        "missing_ratio": [0.1, 0.2, 0.15],
    })
    
    expected_types = {"TransactionID": "numeric", "missing_ratio": "numeric"}
    range_bounds = {"missing_ratio": (0.0, 1.0)}
    
    report = pipeline.run_validation(df, expected_types, range_bounds)
    assert report["overall_status"] == "PASS"
    
    report_file = tmp_path / "report.json"
    pipeline.save_report(report, report_file)
    assert report_file.exists()
    
    with open(report_file) as f:
        meta = json.load(f)
    assert meta["overall_status"] == "PASS"
