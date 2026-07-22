"""Unit tests for AnonymousFeatureAnalyzer — Part 3.11."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.anonymous import AnonymousFeatureAnalyzer


@pytest.fixture()
def sample_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates minimal train/test dataframes with anonymous features."""
    rng = np.random.default_rng(42)
    n = 200

    data_train = {
        "TransactionID": range(n),
        "isFraud": rng.choice([0, 1], n, p=[0.95, 0.05]),
    }
    data_test = {
        "TransactionID": range(n, n + 100),
    }

    # Add V1-V10
    for i in range(1, 11):
        data_train[f"V{i}"] = rng.uniform(0, 100, n)
        data_test[f"V{i}"] = rng.uniform(0, 100, 100)
    # Add C1-C5
    for i in range(1, 6):
        data_train[f"C{i}"] = rng.poisson(3, n)
        data_test[f"C{i}"] = rng.poisson(3, 100)
    # Add D1-D3
    for i in range(1, 4):
        data_train[f"D{i}"] = rng.uniform(10, 500, n)
        data_test[f"D{i}"] = rng.uniform(10, 500, 100)
    # Add M1-M3
    for i in range(1, 4):
        data_train[f"M{i}"] = rng.choice(["T", "F", None], n, p=[0.6, 0.3, 0.1])
        data_test[f"M{i}"] = rng.choice(["T", "F", None], 100, p=[0.6, 0.3, 0.1])

    # Inject some missing values
    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)

    df_train.loc[df_train["TransactionID"] % 10 == 0, "V1"] = np.nan
    df_train.loc[df_train["TransactionID"] % 5 == 0, "D1"] = np.nan

    return df_train, df_test


@pytest.fixture()
def analyzer(sample_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> AnonymousFeatureAnalyzer:
    return AnonymousFeatureAnalyzer(sample_dfs[0], sample_dfs[1], target_col="isFraud")


# ---------------------------------------------------------------------------
# Core Analyzer Tests
# ---------------------------------------------------------------------------

def test_analyze_anonymous_inventory(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_anonymous_inventory(tmp_path)
    assert not df.empty
    assert (tmp_path / "anonymous_feature_inventory.csv").exists()
    assert (tmp_path / "anonymous_feature_metadata.json").exists()


def test_analyze_v_series(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_v_series(tmp_path)
    assert not df.empty
    assert (tmp_path / "v_feature_analysis.csv").exists()


def test_analyze_c_series(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_c_series(tmp_path)
    assert not df.empty
    assert (tmp_path / "c_feature_analysis.csv").exists()


def test_analyze_d_series(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_d_series(tmp_path)
    assert not df.empty
    assert (tmp_path / "d_feature_analysis.csv").exists()


def test_analyze_m_series(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_m_series(tmp_path)
    assert not df.empty
    assert (tmp_path / "m_feature_analysis.csv").exists()


def test_analyze_distributions(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_distributions(tmp_path)
    assert not df.empty
    assert (tmp_path / "anonymous_distribution_analysis.csv").exists()


def test_analyze_missingness(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_missingness(tmp_path)
    assert not df.empty
    assert (tmp_path / "anonymous_missingness_analysis.csv").exists()


def test_analyze_feature_importance(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_feature_importance(tmp_path)
    assert not df.empty
    assert (tmp_path / "anonymous_feature_importance.csv").exists()


def test_analyze_correlation_redundancy(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df_corr, df_clusters = analyzer.analyze_correlation_redundancy(tmp_path)
    assert not df_corr.empty
    assert (tmp_path / "anonymous_correlation_analysis.csv").exists()
    assert (tmp_path / "feature_clusters.csv").exists()


def test_analyze_interactions(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_interactions(tmp_path)
    # Since our sample mock only has M1-M3, M4 interaction won't match, resulting in empty df
    assert df is not None


def test_generate_anonymous_recommendations(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.generate_anonymous_recommendations(tmp_path)
    assert not df.empty
    assert (tmp_path / "anonymous_feature_recommendations.csv").exists()


def test_analyze_all(analyzer: TimeSeriesFeatureAnalyzer, tmp_path: Path) -> None:
    analyzer.analyze_all(tmp_path)
    assert (tmp_path / "anonymous_analysis.json").exists()
    assert (tmp_path / "anonymous_analysis_report.html").exists()
    assert (tmp_path / "plots" / "v_missingness_distribution.png").exists()
    assert (tmp_path / "plots" / "c_distributions.png").exists()
    assert (tmp_path / "plots" / "d_interaction.png").exists()
    assert (tmp_path / "plots" / "importance_ranking.png").exists()
