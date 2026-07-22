"""Unit tests for IdentityFeatureAnalyzer — Part 3.9."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.identity import (
    IdentityFeatureAnalyzer,
    _parse_browser,
    _parse_os,
)


@pytest.fixture()
def sample_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates minimal train/test dataframes with identity features."""
    rng = np.random.default_rng(42)
    n = 200

    df_train = pd.DataFrame({
        "TransactionID": range(n),
        "id_01": rng.normal(0, 10, n),
        "id_02": rng.exponential(100, n),
        "id_12": rng.choice(["Found", "NotFound"], n),
        "id_30": rng.choice(["Windows 10", "iOS 11.3.0", "Mac OS X 10_13_4", "Android 7.0", None], n),
        "id_31": rng.choice(["chrome 63.0", "mobile safari 11.0", "ie 11.0 for desktop", "firefox 62.0", None], n),
        "DeviceType": rng.choice(["desktop", "mobile", None], n),
        "DeviceInfo": rng.choice(["Windows", "iOS Device", "SAMSUNG", None], n),
        "isFraud": rng.choice([0, 1], n, p=[0.96, 0.04]),
    })

    df_test = pd.DataFrame({
        "TransactionID": range(n, n + 100),
        "id_01": rng.normal(0, 10, 100),
        "id_02": rng.exponential(100, 100),
        "id_12": rng.choice(["Found", "NotFound"], 100),
        "id_30": rng.choice(["Windows 10", "iOS 11.3.0", None], 100),
        "id_31": rng.choice(["chrome 63.0", "mobile safari 11.0", None], 100),
        "DeviceType": rng.choice(["desktop", "mobile", None], 100),
        "DeviceInfo": rng.choice(["Windows", "iOS Device", None], 100),
    })

    return df_train, df_test


@pytest.fixture()
def analyzer(sample_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> IdentityFeatureAnalyzer:
    return IdentityFeatureAnalyzer(sample_dfs[0], sample_dfs[1], target_col="isFraud")


# ---------------------------------------------------------------------------
# Helper Parsing Tests
# ---------------------------------------------------------------------------

def test_parse_browser() -> None:
    assert _parse_browser("chrome 63.0") == "Chrome"
    assert _parse_browser("mobile safari 11.0") == "Safari"
    assert _parse_browser("ie 11.0") == "IE"
    assert _parse_browser("edge") == "Edge"
    assert _parse_browser("firefox 60") == "Firefox"
    assert _parse_browser("opera") == "Opera"
    assert _parse_browser("some user agent") == "Other"
    assert _parse_browser(None) == "Unknown"


def test_parse_os() -> None:
    assert _parse_os("Windows 10") == "Windows"
    assert _parse_os("Mac OS X 10_13_4") == "macOS"
    assert _parse_os("iOS 11.3.0") == "iOS"
    assert _parse_os("Android 7.0") == "Android"
    assert _parse_os("Linux 4.19") == "Linux"
    assert _parse_os("some custom OS") == "Other"
    assert _parse_os(None) == "Unknown"


# ---------------------------------------------------------------------------
# Core Analyzer Tests
# ---------------------------------------------------------------------------

def test_analyze_identity_inventory(analyzer: IdentityFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_identity_inventory(tmp_path)
    assert not df.empty
    assert (tmp_path / "identity_feature_inventory.csv").exists()
    assert (tmp_path / "identity_metadata.json").exists()


def test_analyze_id_features(analyzer: IdentityFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_id_features(tmp_path)
    assert not df.empty
    assert (tmp_path / "identity_feature_analysis.csv").exists()


def test_analyze_device_type(analyzer: IdentityFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_device_type(tmp_path)
    assert not df.empty
    assert (tmp_path / "device_type_analysis.csv").exists()


def test_analyze_device_info(analyzer: IdentityFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_device_info(tmp_path)
    assert not df.empty
    assert (tmp_path / "device_info_analysis.csv").exists()


def test_analyze_browsers(analyzer: IdentityFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_browsers(tmp_path)
    assert not df.empty
    assert (tmp_path / "browser_analysis.csv").exists()


def test_analyze_operating_systems(analyzer: IdentityFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_operating_systems(tmp_path)
    assert not df.empty
    assert (tmp_path / "os_analysis.csv").exists()


def test_analyze_identity_availability(analyzer: IdentityFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_identity_availability(tmp_path)
    assert not df.empty
    assert (tmp_path / "identity_availability.csv").exists()


def test_analyze_missingness_fraud(analyzer: IdentityFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_missingness_fraud(tmp_path)
    assert not df.empty
    assert (tmp_path / "identity_missing_analysis.csv").exists()


def test_analyze_interactions(analyzer: IdentityFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_interactions(tmp_path)
    assert not df.empty
    assert (tmp_path / "identity_interactions.csv").exists()


def test_analyze_risk_profiles(analyzer: IdentityFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.analyze_risk_profiles(tmp_path)
    assert not df.empty
    assert (tmp_path / "identity_risk_profiles.csv").exists()


def test_generate_feature_engineering_recommendations(analyzer: IdentityFeatureAnalyzer, tmp_path: Path) -> None:
    df = analyzer.generate_feature_engineering_recommendations(tmp_path)
    assert not df.empty
    assert (tmp_path / "identity_feature_recommendations.csv").exists()


def test_analyze_all(analyzer: IdentityFeatureAnalyzer, tmp_path: Path) -> None:
    analyzer.analyze_all(tmp_path)
    assert (tmp_path / "identity_analysis.json").exists()
    assert (tmp_path / "identity_analysis_report.html").exists()
    assert (tmp_path / "plots" / "devicetype_distribution.png").exists()
    assert (tmp_path / "plots" / "browser_family_distribution.png").exists()
    assert (tmp_path / "plots" / "os_family_distribution.png").exists()
    assert (tmp_path / "plots" / "availability_vs_fraud.png").exists()
