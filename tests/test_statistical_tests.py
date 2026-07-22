"""Unit tests for StatisticalTestsAnalyzer — Part 3.14."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.statistical_tests import StatisticalTestsAnalyzer


@pytest.fixture()
def sample_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates mock train/test dataframes for statistical testing."""
    rng = np.random.default_rng(42)
    n = 200

    # Two distinct normal distributions for numerical feature x to ensure KS/MW tests are significant
    x_legit = rng.normal(50, 10, 150)
    x_fraud = rng.normal(65, 10, 50)
    x = np.concatenate([x_legit, x_fraud])

    # Distinct categories for categorical feature cat to ensure Chi-Square is significant
    cat_legit = rng.choice(["A", "B"], 150, p=[0.8, 0.2])
    cat_fraud = rng.choice(["A", "B"], 50, p=[0.2, 0.8])
    cat = np.concatenate([cat_legit, cat_fraud])

    target = np.array([0] * 150 + [1] * 50)

    data_train = {
        "TransactionID": range(n),
        "isFraud": target,
        "feat_x": x,
        "cat_feat": cat,
    }
    data_test = {
        "TransactionID": range(n, n + 100),
        "feat_x": rng.normal(55, 10, 100),
        "cat_feat": rng.choice(["A", "B"], 100),
    }

    return pd.DataFrame(data_train), pd.DataFrame(data_test)


@pytest.fixture()
def analyzer(sample_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> StatisticalTestsAnalyzer:
    return StatisticalTestsAnalyzer(df_train=sample_dfs[0], df_test=sample_dfs[1], target_col="isFraud")


def test_analyze_statistical_inventory(analyzer: StatisticalTestsAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_statistical_inventory(tmp_path)
    assert not df.empty
    assert (tmp_path / "statistical_feature_inventory.csv").exists()
    assert (tmp_path / "statistical_metadata.json").exists()


def test_analyze_ks_test(analyzer: StatisticalTestsAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_ks_test(tmp_path)
    assert not df.empty
    assert (tmp_path / "ks_results.csv").exists()
    assert (tmp_path / "plots" / "ks_distribution_shifts.png").exists()


def test_analyze_chi_square_test(analyzer: StatisticalTestsAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_chi_square_test(tmp_path)
    assert not df.empty
    assert (tmp_path / "chi_square_results.csv").exists()


def test_analyze_mann_whitney_test(analyzer: StatisticalTestsAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_mann_whitney_test(tmp_path)
    assert not df.empty
    assert (tmp_path / "mann_whitney_results.csv").exists()


def test_analyze_anova(analyzer: StatisticalTestsAnalyzer, tmp_path: Path) -> None:
    # ANOVA tests variation of TransactionAmt across groups
    # Mocking TransactionAmt inside sample data to ensure it runs
    rng = np.random.default_rng(42)
    analyzer.df_sample_large["TransactionAmt"] = rng.normal(100, 15, len(analyzer.df_sample_large))
    
    df = analyzer.analyze_anova(tmp_path)
    assert not df.empty
    assert (tmp_path / "anova_results.csv").exists()


def test_analyze_mutual_information(analyzer: StatisticalTestsAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_mutual_information(tmp_path)
    assert not df.empty
    assert (tmp_path / "mutual_information_results.csv").exists()
    assert (tmp_path / "plots" / "mi_relevance_plot.png").exists()


def test_analyze_multiple_testing_correction(analyzer: StatisticalTestsAnalyzer, tmp_path: Path) -> None:
    pvals_mock = pd.DataFrame({
        "feature_name": ["feat_x", "cat_feat"],
        "raw_p_value": [0.0001, 0.45],
        "test_type": ["KS (Numerical)", "Chi-Square (Categorical)"],
    })
    df = analyzer.analyze_multiple_testing_correction(tmp_path, pvals_mock)
    assert not df.empty
    assert (tmp_path / "multiple_testing_correction.csv").exists()


def test_analyze_effect_size(analyzer: StatisticalTestsAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_effect_size(tmp_path)
    assert not df.empty
    assert (tmp_path / "effect_size_analysis.csv").exists()


def test_rank_statistical_significance(analyzer: StatisticalTestsAnalyzer, tmp_path: Path) -> None:
    corrected_mock = pd.DataFrame({
        "feature_name": ["feat_x", "cat_feat"],
        "raw_p_value": [0.0001, 0.45],
        "fdr_p_value": [0.0002, 0.45],
        "test_type": ["KS (Numerical)", "Chi-Square (Categorical)"],
    })
    mi_mock = pd.DataFrame({
        "feature_name": ["feat_x", "cat_feat"],
        "mutual_information_score": [0.08, 0.001],
    })
    ks_mock = pd.DataFrame({
        "feature_name": ["feat_x"],
        "ks_statistic": [0.35],
    })
    
    df = analyzer.rank_statistical_significance(tmp_path, corrected_mock, mi_mock, ks_mock)
    assert not df.empty
    assert (tmp_path / "statistical_significance_ranking.csv").exists()
    assert (tmp_path / "statistical_feature_recommendations.csv").exists()


def test_analyze_all(analyzer: StatisticalTestsAnalyzer, tmp_path: Path) -> None:
    # Pre-add TransactionAmt to avoid any KeyError/NaN inside code during run
    rng = np.random.default_rng(42)
    analyzer.df_sample_large["TransactionAmt"] = rng.normal(100, 15, len(analyzer.df_sample_large))

    analyzer.analyze_all(tmp_path)
    assert (tmp_path / "statistical_feature_inventory.csv").exists()
    assert (tmp_path / "statistical_metadata.json").exists()
    assert (tmp_path / "ks_results.csv").exists()
    assert (tmp_path / "chi_square_results.csv").exists()
    assert (tmp_path / "mann_whitney_results.csv").exists()
    assert (tmp_path / "anova_results.csv").exists()
    assert (tmp_path / "mutual_information_results.csv").exists()
    assert (tmp_path / "multiple_testing_correction.csv").exists()
    assert (tmp_path / "effect_size_analysis.csv").exists()
    assert (tmp_path / "statistical_significance_ranking.csv").exists()
    assert (tmp_path / "statistical_feature_recommendations.csv").exists()
    assert (tmp_path / "statistical_tests_report.html").exists()
    assert (tmp_path / "plots" / "ks_distribution_shifts.png").exists()
    assert (tmp_path / "plots" / "mi_relevance_plot.png").exists()
