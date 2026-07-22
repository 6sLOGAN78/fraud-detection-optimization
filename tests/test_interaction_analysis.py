"""Unit tests for FeatureInteractionAnalyzer — Part 3.13."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.interaction import FeatureInteractionAnalyzer


@pytest.fixture()
def sample_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates mock train/test dataframes with interacting features."""
    rng = np.random.default_rng(42)
    n = 200

    # Base features
    x1 = rng.uniform(0, 100, n)
    x2 = rng.uniform(0, 100, 100)

    y1 = rng.uniform(0, 100, n)
    y2 = rng.uniform(0, 100, 100)

    # Some target variable (binary) depending on interaction:
    # high fraud iff both x1 and y1 are high
    target_train = np.where((x1 > 50) & (y1 > 50), 1, 0)
    # Add minor noise
    noise = rng.choice([0, 1], n, p=[0.95, 0.05])
    target_train = np.bitwise_xor(target_train, noise)

    data_train = {
        "TransactionID": range(n),
        "isFraud": target_train,
        "feat_x": x1,
        "feat_y": y1,
    }
    data_test = {
        "TransactionID": range(n, n + 100),
        "feat_x": x2,
        "feat_y": y2,
    }

    # Add a categorical column to test filtering
    data_train["cat_feat"] = rng.choice(["A", "B"], n)
    data_test["cat_feat"] = rng.choice(["A", "B"], 100)

    return pd.DataFrame(data_train), pd.DataFrame(data_test)


@pytest.fixture()
def analyzer(sample_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> FeatureInteractionAnalyzer:
    return FeatureInteractionAnalyzer(df_train=sample_dfs[0], df_test=sample_dfs[1], target_col="isFraud")


def test_analyze_interaction_inventory(analyzer: FeatureInteractionAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_interaction_inventory(tmp_path)
    assert not df.empty
    assert (tmp_path / "interaction_feature_inventory.csv").exists()
    assert (tmp_path / "interaction_metadata.json").exists()


def test_analyze_pairplots(analyzer: FeatureInteractionAnalyzer, tmp_path: Path) -> None:
    (tmp_path / "plots").mkdir(parents=True, exist_ok=True)
    df = analyzer.analyze_pairplots(tmp_path)
    assert not df.empty
    assert (tmp_path / "pairplot_analysis.csv").exists()
    assert (tmp_path / "plots" / "pairplot_scatter.png").exists()


def test_analyze_cross_features(analyzer: FeatureInteractionAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_cross_features(tmp_path)
    assert not df.empty
    assert (tmp_path / "cross_feature_analysis.csv").exists()


def test_analyze_fraud_interactions(analyzer: FeatureInteractionAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_fraud_interactions(tmp_path)
    assert (tmp_path / "fraud_interactions.csv").exists()


def test_analyze_interaction_strength(analyzer: FeatureInteractionAnalyzer, tmp_path: Path) -> None:
    (tmp_path / "plots").mkdir(parents=True, exist_ok=True)
    df = analyzer.analyze_interaction_strength(tmp_path)
    assert not df.empty
    assert (tmp_path / "interaction_strength.csv").exists()
    assert (tmp_path / "plots" / "interaction_heatmap.png").exists()


def test_analyze_higher_order_interactions(analyzer: FeatureInteractionAnalyzer, tmp_path: Path) -> None:
    # Add dummy 3rd feature to test triplet combinations
    rng = np.random.default_rng(42)
    analyzer.df_sample_large["feat_z"] = rng.uniform(0, 100, len(analyzer.df_sample_large))
    analyzer.features.append("feat_z")
    
    df = analyzer.analyze_higher_order_interactions(tmp_path)
    assert not df.empty
    assert (tmp_path / "higher_order_interactions.csv").exists()


def test_analyze_interaction_clustering(analyzer: FeatureInteractionAnalyzer, tmp_path: Path) -> None:
    (tmp_path / "plots").mkdir(parents=True, exist_ok=True)
    strength_df = analyzer.analyze_interaction_strength(tmp_path)
    df = analyzer.analyze_interaction_clustering(tmp_path, strength_df)
    assert not df.empty
    assert (tmp_path / "interaction_clusters.csv").exists()
    assert (tmp_path / "plots" / "interaction_dendrogram.png").exists()


def test_recommend_interactions(analyzer: FeatureInteractionAnalyzer, tmp_path: Path) -> None:
    strength_df = analyzer.analyze_interaction_strength(tmp_path)
    df = analyzer.recommend_interactions(tmp_path, strength_df)
    assert not df.empty
    assert (tmp_path / "interaction_feature_recommendations.csv").exists()


def test_analyze_interaction_stability(analyzer: FeatureInteractionAnalyzer, tmp_path: Path) -> None:
    df, report = analyzer.analyze_interaction_stability(tmp_path)
    assert not df.empty
    assert (tmp_path / "interaction_stability.csv").exists()
    assert (tmp_path / "interaction_drift_report.json").exists()


def test_screen_production_interactions(analyzer: FeatureInteractionAnalyzer, tmp_path: Path) -> None:
    stability_df, _ = analyzer.analyze_interaction_stability(tmp_path)
    df = analyzer.screen_production_interactions(tmp_path, stability_df)
    assert not df.empty
    assert (tmp_path / "production_interaction_screening.csv").exists()


def test_analyze_all(analyzer: FeatureInteractionAnalyzer, tmp_path: Path) -> None:
    analyzer.analyze_all(tmp_path)
    assert (tmp_path / "interaction_feature_inventory.csv").exists()
    assert (tmp_path / "interaction_metadata.json").exists()
    assert (tmp_path / "pairplot_analysis.csv").exists()
    assert (tmp_path / "cross_feature_analysis.csv").exists()
    assert (tmp_path / "fraud_interactions.csv").exists()
    assert (tmp_path / "interaction_strength.csv").exists()
    assert (tmp_path / "higher_order_interactions.csv").exists()
    assert (tmp_path / "interaction_clusters.csv").exists()
    assert (tmp_path / "interaction_feature_recommendations.csv").exists()
    assert (tmp_path / "interaction_stability.csv").exists()
    assert (tmp_path / "interaction_drift_report.json").exists()
    assert (tmp_path / "production_interaction_screening.csv").exists()
    assert (tmp_path / "interaction_analysis_report.html").exists()
    assert (tmp_path / "plots" / "pairplot_scatter.png").exists()
    assert (tmp_path / "plots" / "interaction_heatmap.png").exists()
    assert (tmp_path / "plots" / "interaction_dendrogram.png").exists()
