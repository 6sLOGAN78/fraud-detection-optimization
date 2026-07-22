"""Unit tests for DriftAnalyzer — Part 3.15."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.drift import DriftAnalyzer


@pytest.fixture()
def mock_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates mock train/test dataframes representing drift behaviors."""
    rng = np.random.default_rng(42)
    n = 200

    # Train Feature Distribution
    x_trn = rng.normal(50, 10, n)
    cat_trn = rng.choice(["A", "B"], n, p=[0.7, 0.3])

    # Test Feature Distribution (introducing drift for x and cat)
    x_tst = rng.normal(58, 12, n)
    cat_tst = rng.choice(["A", "B", "C"], n, p=[0.2, 0.4, 0.4])

    df_train = pd.DataFrame({
        "TransactionID": range(n),
        "feat_x": x_trn,
        "cat_feat": cat_trn,
    })
    df_test = pd.DataFrame({
        "TransactionID": range(n, n + n),
        "feat_x": x_tst,
        "cat_feat": cat_tst,
    })

    return df_train, df_test


@pytest.fixture()
def analyzer(mock_datasets: tuple[pd.DataFrame, pd.DataFrame]) -> DriftAnalyzer:
    return DriftAnalyzer(df_train=mock_datasets[0], df_test=mock_datasets[1])


def test_analyze_drift_inventory(analyzer: DriftAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_drift_inventory(tmp_path)
    assert not df.empty
    assert (tmp_path / "drift_feature_inventory.csv").exists()
    assert (tmp_path / "drift_metadata.json").exists()


def test_analyze_train_test_stats_drift(analyzer: DriftAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_train_test_stats_drift(tmp_path)
    assert not df.empty
    assert (tmp_path / "train_test_drift.csv").exists()


def test_compute_numerical_psi(analyzer: DriftAnalyzer) -> None:
    rng = np.random.default_rng(42)
    # Target and baseline match exactly -> PSI should be close to 0
    t_v = rng.normal(100, 10, 500)
    p_val = analyzer.compute_numerical_psi(t_v, t_v)
    assert abs(p_val) < 0.05

    # Target drifted significantly -> PSI should be high
    t_drifted = rng.normal(115, 10, 500)
    p_val_drifted = analyzer.compute_numerical_psi(t_v, t_drifted)
    assert p_val_drifted > 0.25


def test_compute_categorical_psi(analyzer: DriftAnalyzer) -> None:
    t_v = pd.Series(["A"] * 70 + ["B"] * 30)
    # Match -> PSI should be near 0
    p_val = analyzer.compute_categorical_psi(t_v, t_v)
    assert p_val < 0.05

    # Drifted -> PSI should be high
    t_dr = pd.Series(["A"] * 20 + ["B"] * 80)
    p_val_d = analyzer.compute_categorical_psi(t_v, t_dr)
    assert p_val_d > 0.25


def test_analyze_psi(analyzer: DriftAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_psi(tmp_path)
    assert not df.empty
    assert (tmp_path / "psi_analysis_summary.csv").exists()


def test_analyze_ks_drift(analyzer: DriftAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_ks_drift(tmp_path)
    assert not df.empty
    assert (tmp_path / "ks_drift_analysis.csv").exists()
    assert (tmp_path / "plots" / "ks_drift_shifts.png").exists()


def test_analyze_distribution_drift(analyzer: DriftAnalyzer, tmp_path: Path) -> None:
    psi_df = analyzer.analyze_psi(tmp_path)
    ks_df = analyzer.analyze_ks_drift(tmp_path)
    df = analyzer.analyze_distribution_drift(tmp_path, psi_df, ks_df)
    assert not df.empty
    assert (tmp_path / "distribution_drift_analysis.csv").exists()
    assert (tmp_path / "plots" / "categorical_drift_shifts.png").exists()


def test_analyze_feature_stability(analyzer: DriftAnalyzer, tmp_path: Path) -> None:
    psi_df = analyzer.analyze_psi(tmp_path)
    ks_df = analyzer.analyze_ks_drift(tmp_path)
    df = analyzer.analyze_feature_stability(tmp_path, psi_df, ks_df)
    assert not df.empty
    assert (tmp_path / "feature_stability_analysis.csv").exists()


def test_classify_drift_severity(analyzer: DriftAnalyzer, tmp_path: Path) -> None:
    psi_df = analyzer.analyze_psi(tmp_path)
    ks_df = analyzer.analyze_ks_drift(tmp_path)
    stab_df = analyzer.analyze_feature_stability(tmp_path, psi_df, ks_df)
    df = analyzer.classify_drift_severity(tmp_path, stab_df)
    assert not df.empty
    assert (tmp_path / "drift_severity_report.csv").exists()


def test_analyze_drift_root_cause(analyzer: DriftAnalyzer, tmp_path: Path) -> None:
    psi_df = analyzer.analyze_psi(tmp_path)
    ks_df = analyzer.analyze_ks_drift(tmp_path)
    stab_df = analyzer.analyze_feature_stability(tmp_path, psi_df, ks_df)
    sev_df = analyzer.classify_drift_severity(tmp_path, stab_df)
    df = analyzer.analyze_drift_root_cause(tmp_path, sev_df)
    assert not df.empty
    assert (tmp_path / "drift_root_cause_analysis.csv").exists()


def test_analyze_all(analyzer: DriftAnalyzer, tmp_path: Path) -> None:
    analyzer.analyze_all(tmp_path)
    assert (tmp_path / "drift_feature_inventory.csv").exists()
    assert (tmp_path / "drift_metadata.json").exists()
    assert (tmp_path / "train_test_drift.csv").exists()
    assert (tmp_path / "psi_analysis_summary.csv").exists()
    assert (tmp_path / "ks_drift_analysis.csv").exists()
    assert (tmp_path / "distribution_drift_analysis.csv").exists()
    assert (tmp_path / "feature_stability_analysis.csv").exists()
    assert (tmp_path / "drift_severity_report.csv").exists()
    assert (tmp_path / "drift_root_cause_analysis.csv").exists()
    assert (tmp_path / "drift_report.html").exists()
    assert (tmp_path / "plots" / "ks_drift_shifts.png").exists()
    assert (tmp_path / "plots" / "categorical_drift_shifts.png").exists()
