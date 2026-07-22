"""Unit tests for Feature Validation components, drift metrics, target leakage, Pearson/Spearman redundancies, and surrogate models."""

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
    DriftDetector,
    LeakageDetector,
    CorrelationValidator,
    ImportanceValidator,
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
        "feature_c": "numeric",
    }
    
    report = validator.validate(df, expected_types)
    assert report["missing_columns"] == ["feature_c"]
    assert report["dtype_mismatches_count"] == 0
    assert report["status"] == "FAIL"


def test_missing_value_validator() -> None:
    validator = MissingValueValidator()
    
    df = pd.DataFrame({
        "col_a": [1.0, np.nan, np.nan, np.nan],
        "col_b": [2.0, np.inf, 4.0, 5.0],
    })
    
    thresholds = {"col_a": 0.50}
    report = validator.validate(df, thresholds)
    
    assert "col_a" in report["high_missingness"]
    assert "col_b" in report["infinite_columns"]
    assert report["status"] == "WARN"


def test_statistical_validator() -> None:
    validator = StatisticalValidator()
    
    df = pd.DataFrame({
        "col_a": [1.0, 1.0, 1.0, 1.0],
        "col_b": [1.0, 2.0, 100.0, np.nan],
        "col_c": [1.0, 1.1, 1.2, 10.0],
    })
    
    range_bounds = {"col_b": (1.0, 10.0)}
    report = validator.validate(df, range_bounds, outlier_sigma=1.2)
    
    assert "col_a" in report["constant_columns"]
    assert "col_b" in report["out_of_bounds"]
    assert "col_c" in report["outliers"]
    assert report["status"] == "WARN"


def test_drift_detector() -> None:
    detector = DriftDetector()
    
    # Large matching datasets to avoid bins artifact
    df_train = pd.DataFrame({"col_a": np.random.normal(0, 1, 1000)})
    df_test_same = pd.DataFrame({"col_a": np.random.normal(0, 1, 1000)})
    df_test_shift = pd.DataFrame({"col_a": np.random.normal(3, 1, 1000)})
    
    report_same = detector.validate_drift(df_train, df_test_same, threshold=0.20)
    report_shift = detector.validate_drift(df_train, df_test_shift, threshold=0.20)
    
    assert report_same["status"] == "PASS"
    assert report_shift["status"] == "WARN"
    assert "col_a" in report_shift["drifted_columns"]


def test_leakage_detector() -> None:
    detector = LeakageDetector()
    
    df = pd.DataFrame({
        "col_perfect": [0, 1, 0, 1, 0, 1, 0, 1],
        "col_random": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    })
    target = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    
    report = detector.validate_leakage(df, target, threshold=0.90)
    assert "col_perfect" in report["leakage_columns"]
    assert "col_random" not in report["leakage_columns"]


def test_correlation_validator() -> None:
    validator = CorrelationValidator()
    
    df = pd.DataFrame({
        "col_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "col_b": [2.0, 4.0, 6.0, 8.0, 10.0],
        "col_c": [1.0, 5.0, 2.0, 8.0, 3.0],
    })
    
    report = validator.validate_redundancy(df, threshold=0.90)
    assert "col_a__vs__col_b" in report["pairs"]
    assert "col_a__vs__col_c" not in report["pairs"]


def test_importance_validator() -> None:
    validator = ImportanceValidator()
    
    df = pd.DataFrame({
        "col_important": [0, 1, 0, 1, 0, 1, 0, 1],
        "col_noise": [9, 2, 8, 1, 5, 7, 3, 6],
    })
    target = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    
    report = validator.validate_importance(df, target, threshold=0.05)
    assert "col_noise" in report["zero_importance_features"]


def test_validation_pipeline_comprehensive(tmp_path: Path) -> None:
    pipeline = FeatureValidationPipeline()
    
    df_train = pd.DataFrame({
        "TransactionID": [100, 101, 102, 103, 104],
        "missing_ratio": [0.1, 0.2, 0.15, 0.25, 0.3],
    })
    df_test = pd.DataFrame({
        "TransactionID": [200, 201, 202, 203, 204],
        "missing_ratio": [0.1, 0.2, 0.15, 0.25, 0.3],
    })
    target = pd.Series([0, 0, 1, 0, 0])
    
    expected_types = {"TransactionID": "numeric", "missing_ratio": "numeric"}
    range_bounds = {"missing_ratio": (0.0, 1.0)}
    
    report = pipeline.run_validation(
        df_test,
        expected_types=expected_types,
        range_bounds=range_bounds,
        df_ref=df_train,
        target=target,
    )
    
    assert report["overall_status"] == "PASS"
    assert report["drift_validation"]["status"] == "PASS"
    assert report["leakage_validation"]["status"] == "PASS"
    assert report["correlation_validation"]["status"] == "PASS"
    assert report["importance_validation"]["status"] == "PASS"
