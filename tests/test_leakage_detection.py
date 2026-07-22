"""Unit tests for DataLeakageDetector — Part 3.16."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.leakage import DataLeakageDetector


@pytest.fixture()
def mock_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates mock train/test dataframes representing leakage risks."""
    rng = np.random.default_rng(42)
    n = 200

    # Train Feature Distribution
    x_trn = rng.normal(50, 10, n)
    cat_trn = rng.choice(["A", "B"], n, p=[0.7, 0.3])
    # Introduce direct target leakage feature
    chargeback_trn = rng.choice([0.0, 1.0], n, p=[0.9, 0.1])
    is_fraud = (chargeback_trn == 1.0).astype(int)

    # Test Feature Distribution (introducing overlaps)
    x_tst = rng.normal(52, 10, n)
    cat_tst = rng.choice(["A", "B"], n, p=[0.6, 0.4])
    chargeback_tst = rng.choice([0.0, 1.0], n, p=[0.9, 0.1])

    df_train = pd.DataFrame({
        "TransactionID": range(n),
        "feat_x": x_trn,
        "cat_feat": cat_trn,
        "chargeback_status": chargeback_trn,
        "isFraud": is_fraud,
    })
    df_test = pd.DataFrame({
        "TransactionID": range(n, n + n),
        "feat_x": x_tst,
        "cat_feat": cat_tst,
        "chargeback_status": chargeback_tst,
    })

    return df_train, df_test


@pytest.fixture()
def detector(mock_datasets: tuple[pd.DataFrame, pd.DataFrame]) -> DataLeakageDetector:
    return DataLeakageDetector(df_train=mock_datasets[0], df_test=mock_datasets[1])


def test_analyze_leakage_prep_inventory(detector: DataLeakageDetector, tmp_path: Path) -> None:
    df = detector.analyze_leakage_prep_inventory(tmp_path)
    assert not df.empty
    assert (tmp_path / "leakage_feature_inventory.csv").exists()
    assert (tmp_path / "leakage_metadata.json").exists()


def test_analyze_high_target_correlation(detector: DataLeakageDetector, tmp_path: Path) -> None:
    df = detector.analyze_high_target_correlation(tmp_path)
    assert not df.empty
    assert (tmp_path / "high_target_correlation.csv").exists()
    assert (tmp_path / "plots" / "high_correlation_plot.png").exists()


def test_analyze_target_leakage(detector: DataLeakageDetector, tmp_path: Path) -> None:
    corr_df = detector.analyze_high_target_correlation(tmp_path)
    df = detector.analyze_target_leakage(tmp_path, corr_df)
    assert not df.empty
    assert (tmp_path / "target_leakage_analysis.csv").exists()
    # Check if chargeback_status was flagged with high/critical risk due to correlation/name keyword
    leak_rows = df[df["feature_name"] == "chargeback_status"]
    assert not leak_rows.empty
    assert leak_rows.iloc[0]["leakage_severity"] in {"HIGH", "CRITICAL"}


def test_analyze_future_leakage(detector: DataLeakageDetector, tmp_path: Path) -> None:
    df = detector.analyze_future_leakage(tmp_path)
    assert not df.empty
    assert (tmp_path / "future_leakage_analysis.csv").exists()


def test_analyze_duplicate_leakage(detector: DataLeakageDetector, tmp_path: Path) -> None:
    df = detector.analyze_duplicate_leakage(tmp_path)
    assert not df.empty
    assert (tmp_path / "duplicate_leakage.csv").exists()


def test_analyze_contamination(detector: DataLeakageDetector, tmp_path: Path) -> None:
    df = detector.analyze_contamination(tmp_path)
    assert not df.empty
    assert (tmp_path / "train_test_contamination.csv").exists()


def test_analyze_pipeline_leakage(detector: DataLeakageDetector, tmp_path: Path) -> None:
    df_eng, df_enc, df_agg, df_trans = detector.analyze_pipeline_leakage(tmp_path)
    assert (tmp_path / "feature_engineering_leakage.csv").exists()
    assert (tmp_path / "encoding_leakage.csv").exists()
    assert (tmp_path / "aggregation_leakage.csv").exists()
    assert (tmp_path / "transformation_leakage.csv").exists()


def test_assess_leakage_risk(detector: DataLeakageDetector, tmp_path: Path) -> None:
    corr_df = detector.analyze_high_target_correlation(tmp_path)
    target_leak_df = detector.analyze_target_leakage(tmp_path, corr_df)
    contamination_df = detector.analyze_contamination(tmp_path)
    dup_df = detector.analyze_duplicate_leakage(tmp_path)

    df_risk = detector.assess_leakage_risk(
        tmp_path,
        corr_df,
        target_leak_df,
        contamination_df,
        dup_df,
    )
    assert not df_risk.empty
    assert (tmp_path / "leakage_severity_report.csv").exists()


def test_analyze_all(detector: DataLeakageDetector, tmp_path: Path) -> None:
    detector.analyze_all(tmp_path)
    assert (tmp_path / "leakage_feature_inventory.csv").exists()
    assert (tmp_path / "leakage_metadata.json").exists()
    assert (tmp_path / "high_target_correlation.csv").exists()
    assert (tmp_path / "target_leakage_analysis.csv").exists()
    assert (tmp_path / "future_leakage_analysis.csv").exists()
    assert (tmp_path / "duplicate_leakage.csv").exists()
    assert (tmp_path / "train_test_contamination.csv").exists()
    assert (tmp_path / "feature_engineering_leakage.csv").exists()
    assert (tmp_path / "encoding_leakage.csv").exists()
    assert (tmp_path / "aggregation_leakage.csv").exists()
    assert (tmp_path / "transformation_leakage.csv").exists()
    assert (tmp_path / "leakage_severity_report.csv").exists()
    assert (tmp_path / "leakage_report.html").exists()
    assert (tmp_path / "plots" / "high_correlation_plot.png").exists()
