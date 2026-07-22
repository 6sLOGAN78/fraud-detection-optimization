"""Unit tests for CorrelationAnalyzer — Part 3.12."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.correlation import CorrelationAnalyzer


@pytest.fixture()
def sample_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates mock train/test dataframes with correlated features."""
    rng = np.random.default_rng(42)
    n = 200

    # Base feature
    x1 = rng.uniform(0, 100, n)
    x2 = rng.uniform(0, 100, 100)

    # Correlated features
    y1 = x1 + rng.normal(0, 1, n)  # Pearson/Spearman near 1.0
    y2 = x2 + rng.normal(0, 1, 100)

    # Highly redundant target pair
    z1 = x1 * 2 + rng.normal(0, 0.1, n)
    z2 = x2 * 2 + rng.normal(0, 0.1, 100)

    # Some target variable (binary)
    target_train = rng.choice([0, 1], n, p=[0.90, 0.10])

    data_train = {
        "TransactionID": range(n),
        "isFraud": target_train,
        "feat_x": x1,
        "feat_y": y1,
        "feat_z": z1,
    }
    data_test = {
        "TransactionID": range(n, n + 100),
        "feat_x": x2,
        "feat_y": y2,
        "feat_z": z2,
    }

    # Add a categorical column to test filtering
    data_train["cat_feat"] = rng.choice(["A", "B"], n)
    data_test["cat_feat"] = rng.choice(["A", "B"], 100)

    return pd.DataFrame(data_train), pd.DataFrame(data_test)


@pytest.fixture()
def analyzer(sample_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> CorrelationAnalyzer:
    return CorrelationAnalyzer(df_train=sample_dfs[0], df_test=sample_dfs[1], target_col="isFraud")


def test_analyze_correlation_inventory(analyzer: CorrelationAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_correlation_inventory(tmp_path)
    assert not df.empty
    assert (tmp_path / "correlation_feature_inventory.csv").exists()
    assert (tmp_path / "correlation_metadata.json").exists()


def test_compute_correlations(analyzer: CorrelationAnalyzer, tmp_path: Path) -> None:
    p_mat, s_mat, k_mat = analyzer.compute_correlations(tmp_path)
    assert not p_mat.empty
    assert not s_mat.empty
    assert not k_mat.empty
    assert (tmp_path / "pearson_correlation.csv").exists()
    assert (tmp_path / "spearman_correlation.csv").exists()
    assert (tmp_path / "kendall_correlation.csv").exists()


def test_analyze_cluster_map(analyzer: CorrelationAnalyzer, tmp_path: Path) -> None:
    p_mat, _, _ = analyzer.compute_correlations(tmp_path)
    (tmp_path / "plots").mkdir(parents=True, exist_ok=True)
    df_clusters = analyzer.analyze_cluster_map(tmp_path, p_mat)
    assert not df_clusters.empty
    assert (tmp_path / "feature_clusters.csv").exists()
    assert (tmp_path / "plots" / "dendrogram.png").exists()


def test_analyze_network(analyzer: CorrelationAnalyzer, tmp_path: Path) -> None:
    p_mat, _, _ = analyzer.compute_correlations(tmp_path)
    df_nodes = analyzer.analyze_network(tmp_path, p_mat)
    assert not df_nodes.empty
    assert (tmp_path / "correlation_network.csv").exists()
    assert (tmp_path / "network_nodes.csv").exists()


def test_analyze_target_correlation(analyzer: CorrelationAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_target_correlation(tmp_path)
    assert not df.empty
    assert (tmp_path / "target_correlation.csv").exists()


def test_correlation_pruning(analyzer: CorrelationAnalyzer, tmp_path: Path) -> None:
    p_mat, _, _ = analyzer.compute_correlations(tmp_path)
    df_target = analyzer.analyze_target_correlation(tmp_path)
    
    df_pruni, df_keep, df_drop = analyzer.correlation_pruning(tmp_path, p_mat, df_target)
    assert not df_pruni.empty
    assert not df_keep.empty
    assert not df_drop.empty
    
    assert (tmp_path / "correlation_pruning.csv").exists()
    assert (tmp_path / "retained_features.csv").exists()
    assert (tmp_path / "removed_features.csv").exists()


def test_analyze_all(analyzer: CorrelationAnalyzer, tmp_path: Path) -> None:
    analyzer.analyze_all(tmp_path)
    assert (tmp_path / "correlation_feature_inventory.csv").exists()
    assert (tmp_path / "correlation_metadata.json").exists()
    assert (tmp_path / "pearson_correlation.csv").exists()
    assert (tmp_path / "spearman_correlation.csv").exists()
    assert (tmp_path / "correlation_matrix.csv").exists()
    assert (tmp_path / "feature_clusters.csv").exists()
    assert (tmp_path / "correlation_network.csv").exists()
    assert (tmp_path / "target_correlation.csv").exists()
    assert (tmp_path / "correlation_pruning.csv").exists()
    assert (tmp_path / "correlation_analysis_report.html").exists()
    assert (tmp_path / "plots" / "pearson_heatmap.png").exists()
    assert (tmp_path / "plots" / "spearman_heatmap.png").exists()
    assert (tmp_path / "plots" / "dendrogram.png").exists()
    assert (tmp_path / "plots" / "target_mutual_info.png").exists()
