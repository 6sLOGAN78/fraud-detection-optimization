"""Unit tests for TimeSeriesFeatureAnalyzer — Part 3.10."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.timeseries import TimeSeriesFeatureAnalyzer


@pytest.fixture()
def sample_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates minimal train/test dataframes with TransactionDT features."""
    rng = np.random.default_rng(42)
    n = 200

    # Start DT: 1 day in seconds = 86400. Span = 20 days.
    train_dt = np.linspace(86400, 86400 + 20 * 86400, n)
    test_dt = np.linspace(86400 + 21 * 86400, 86400 + 30 * 86400, 100)

    df_train = pd.DataFrame({
        "TransactionID": range(n),
        "TransactionDT": train_dt,
        "TransactionAmt": rng.uniform(5, 500, n),
        "isFraud": rng.choice([0, 1], n, p=[0.95, 0.05]),
    })

    df_test = pd.DataFrame({
        "TransactionID": range(n, n + 100),
        "TransactionDT": test_dt,
        "TransactionAmt": rng.uniform(5, 500, 100),
    })

    return df_train, df_test


@pytest.fixture()
def analyzer(sample_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> TimeSeriesFeatureAnalyzer:
    return TimeSeriesFeatureAnalyzer(sample_dfs[0], sample_dfs[1], target_col="isFraud")


# ---------------------------------------------------------------------------
# Core Analyzer Tests
# ---------------------------------------------------------------------------

def test_analyze_transaction_dt(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_transaction_dt(tmp_path)
    assert not df.empty
    assert (tmp_path / "transactiondt_analysis.csv").exists()
    assert (tmp_path / "transactiondt_summary.json").exists()


def test_analyze_hourly(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_hourly(tmp_path)
    assert not df.empty
    assert len(df) <= 24
    assert (tmp_path / "hourly_analysis.csv").exists()


def test_analyze_daily(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_daily(tmp_path)
    assert not df.empty
    assert (tmp_path / "daily_analysis.csv").exists()


def test_analyze_weekly(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_weekly(tmp_path)
    assert not df.empty
    assert (tmp_path / "weekly_analysis.csv").exists()


def test_analyze_monthly(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_monthly(tmp_path)
    assert not df.empty
    assert (tmp_path / "monthly_analysis.csv").exists()


def test_analyze_fraud_trends(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_fraud_trends(tmp_path)
    assert not df.empty
    assert (tmp_path / "fraud_trend_analysis.csv").exists()


def test_analyze_seasonality(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_seasonality(tmp_path)
    assert not df.empty
    assert (tmp_path / "seasonality_analysis.csv").exists()


def test_analyze_temporal_drift(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_temporal_drift(tmp_path)
    assert not df.empty
    assert (tmp_path / "temporal_drift_analysis.csv").exists()
    assert (tmp_path / "temporal_drift_report.json").exists()

    with (tmp_path / "temporal_drift_report.json").open("r") as f:
        rep = json.load(f)
        assert "psi_total" in rep
        assert "drift_status" in rep


def test_detect_temporal_anomalies(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.detect_temporal_anomalies(tmp_path)
    assert not df.empty
    assert (tmp_path / "temporal_anomalies.csv").exists()


def test_generate_time_feature_recommendations(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.generate_time_feature_recommendations(tmp_path)
    assert not df.empty
    assert (tmp_path / "time_feature_recommendations.csv").exists()


def test_analyze_all(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    analyzer.analyze_all(tmp_path)
    assert (tmp_path / "timeseries_analysis.json").exists()
    assert (tmp_path / "timeseries_analysis_report.html").exists()
    assert (tmp_path / "plots" / "transaction_timeline.png").exists()
    assert (tmp_path / "plots" / "hourly_distribution.png").exists()
    assert (tmp_path / "plots" / "weekly_trend.png").exists()
    assert (tmp_path / "plots" / "drift_analysis.png").exists()
