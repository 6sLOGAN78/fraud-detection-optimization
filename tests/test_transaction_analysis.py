"""Unit tests for TransactionFeatureAnalyzer — Part 3.8."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.transaction import (
    TransactionFeatureAnalyzer,
    _amt_bin,
    _dist_bin,
)


@pytest.fixture()
def sample_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates minimal train/test dataframes with transaction features."""
    rng = np.random.default_rng(42)
    n = 200

    df_train = pd.DataFrame({
        "TransactionID": range(n),
        "TransactionDT": rng.integers(86400, 15000000, n),
        "TransactionAmt": rng.exponential(100, n),
        "ProductCD": rng.choice(["W", "H", "C", "S", "R"], n),
        "card1": rng.integers(1000, 20000, n),
        "card2": rng.integers(100, 600, n),
        "card3": rng.integers(100, 200, n),
        "card4": rng.choice(["visa", "mastercard", "american express", "discover"], n),
        "card5": rng.integers(100, 250, n),
        "card6": rng.choice(["debit", "credit"], n),
        "addr1": rng.integers(100, 500, n),
        "addr2": rng.integers(100, 150, n),
        "dist1": rng.exponential(50, n),
        "dist2": rng.exponential(100, n),
        "isFraud": rng.choice([0, 1], n, p=[0.96, 0.04]),
    })

    # Add some null values
    df_train.loc[rng.choice(n, 10, replace=False), "dist1"] = np.nan
    df_train.loc[rng.choice(n, 10, replace=False), "dist2"] = np.nan

    df_test = pd.DataFrame({
        "TransactionID": range(n, n + 100),
        "TransactionDT": rng.integers(86400, 15000000, 100),
        "TransactionAmt": rng.exponential(100, 100),
        "ProductCD": rng.choice(["W", "H", "C"], 100),
        "card1": rng.integers(1000, 20000, 100),
        "card2": rng.integers(100, 600, 100),
        "card3": rng.integers(100, 200, 100),
        "card4": rng.choice(["visa", "mastercard"], 100),
        "card5": rng.integers(100, 250, 100),
        "card6": rng.choice(["debit", "credit"], 100),
        "addr1": rng.integers(100, 500, 100),
        "addr2": rng.integers(100, 150, 100),
        "dist1": rng.exponential(50, 100),
        "dist2": rng.exponential(100, 100),
    })

    return df_train, df_test


@pytest.fixture()
def analyzer(sample_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> TransactionFeatureAnalyzer:
    return TransactionFeatureAnalyzer(sample_dfs[0], sample_dfs[1], target_col="isFraud")


# ---------------------------------------------------------------------------
# Binning / Helper Tests
# ---------------------------------------------------------------------------

def test_amt_binning() -> None:
    s = pd.Series([10, 50, 250, 1000, 5000])
    binned = _amt_bin(s)
    assert binned.tolist() == ["Very Low", "Low", "Medium", "High", "Very High"]


def test_dist_binning() -> None:
    s = pd.Series([1.0, 10.0, 100.0, 500.0, np.nan])
    binned = _dist_bin(s)
    # verify it returns categorical categories
    assert binned.dropna().nunique() <= 4


# ---------------------------------------------------------------------------
# Analyzer Tests
# ---------------------------------------------------------------------------

def test_analyze_transaction_amount(analyzer: TransactionFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_transaction_amount(tmp_path)
    assert not df.empty
    assert (tmp_path / "transaction_amount_analysis.csv").exists()
    assert (tmp_path / "transaction_amount_stats.json").exists()


def test_analyze_productcd(analyzer: TransactionFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_productcd(tmp_path)
    assert not df.empty
    assert "product" in df.columns
    assert "fraud_rate" in df.columns
    assert (tmp_path / "productcd_analysis.csv").exists()


def test_analyze_card_features(analyzer: TransactionFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_card_features(tmp_path)
    assert not df.empty
    assert "feature" in df.columns
    assert "n_unique" in df.columns
    assert (tmp_path / "card_feature_analysis.csv").exists()


def test_analyze_address_features(analyzer: TransactionFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_address_features(tmp_path)
    assert not df.empty
    assert "feature" in df.columns
    assert "n_unique" in df.columns
    assert (tmp_path / "address_feature_analysis.csv").exists()


def test_analyze_distance_features(analyzer: TransactionFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_distance_features(tmp_path)
    assert not df.empty
    assert "dist_bin" in df.columns
    assert (tmp_path / "distance_feature_analysis.csv").exists()


def test_analyze_transaction_timing(analyzer: TransactionFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_transaction_timing(tmp_path)
    assert not df.empty
    assert "hour" in df.columns
    assert "fraud_rate" in df.columns
    assert (tmp_path / "transaction_time_analysis.csv").exists()


def test_analyze_cross_features(analyzer: TransactionFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_cross_features(tmp_path)
    assert not df.empty
    assert "interaction" in df.columns
    assert (tmp_path / "transaction_feature_interactions.csv").exists()


def test_analyze_risk_profiles(analyzer: TransactionFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_risk_profiles(tmp_path)
    assert not df.empty
    assert "transaction_count" in df.columns
    assert (tmp_path / "transaction_risk_profiles.csv").exists()


def test_generate_feature_engineering_recommendations(analyzer: TransactionFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.generate_feature_engineering_recommendations(tmp_path)
    assert not df.empty
    assert "engineered_feature" in df.columns
    assert (tmp_path / "transaction_feature_recommendations.csv").exists()


def test_analyze_all(analyzer: TransactionFeatureAnalyzer, tmp_path: Path) -> None:
    analyzer.analyze_all(tmp_path)
    assert (tmp_path / "transaction_analysis.json").exists()
    assert (tmp_path / "transaction_analysis_report.html").exists()
    assert (tmp_path / "plots" / "transaction_amt_hist.png").exists()
    assert (tmp_path / "plots" / "productcd_fraud_rate.png").exists()
    assert (tmp_path / "plots" / "hourly_volume.png").exists()
    assert (tmp_path / "plots" / "hourly_fraud_rate.png").exists()
