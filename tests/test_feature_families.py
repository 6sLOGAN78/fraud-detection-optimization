"""Unit tests for Feature Families engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.families import (
    TransactionFamilyBuilder,
    IdentityFamilyBuilder,
    TimeFamilyBuilder,
    AmountFamilyBuilder,
    EmailFamilyBuilder,
    DeviceFamilyBuilder,
    AddressFamilyBuilder,
    CardFamilyBuilder,
    FeatureFamilyTracker,
    FeatureFamilyIntegrator,
)


@pytest.fixture()
def mock_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 20
    df = pd.DataFrame({
        "TransactionID": range(n),
        "TransactionDT": rng.integers(100, 100000, n),
        "TransactionAmt": rng.uniform(5.5, 400.0, n),
        "ProductCD": rng.choice(["W", "H", "C"], n),
        "DeviceType": rng.choice(["mobile", "desktop", None], n),
        "DeviceInfo": rng.choice(["Windows", "iOS Device", "Android phone"], n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "corporate.com"], n),
        "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", "corporate.com"], n),
        "addr1": rng.choice([101.0, 202.0, np.nan], n),
        "addr2": rng.choice([87.0, 87.0, np.nan], n),
        "card1": rng.choice([1000, 2000, 3000], n),
        "card4": rng.choice(["visa", "mastercard", "discover"], n),
        "id_01": rng.choice([0.0, np.nan], n),
        "id_02": rng.choice([120.0, np.nan], n),
    })
    return df


def test_transaction_family(mock_df: pd.DataFrame) -> None:
    builder = TransactionFamilyBuilder()
    res = builder.transform(mock_df)
    assert "TransactionID" in res.columns
    assert "ProductCD_count" in res.columns
    assert "transaction_density_index" in res.columns


def test_identity_family(mock_df: pd.DataFrame) -> None:
    builder = IdentityFamilyBuilder()
    res = builder.transform(mock_df)
    assert "TransactionID" in res.columns
    assert "identity_completeness_score" in res.columns
    assert "is_mobile_device" in res.columns


def test_time_family(mock_df: pd.DataFrame) -> None:
    builder = TimeFamilyBuilder()
    res = builder.transform(mock_df)
    assert "TransactionID" in res.columns
    assert "transaction_hour" in res.columns
    assert "transaction_day" in res.columns
    assert "is_business_hour" in res.columns


def test_amount_family(mock_df: pd.DataFrame) -> None:
    builder = AmountFamilyBuilder()
    res = builder.transform(mock_df)
    assert "TransactionID" in res.columns
    assert "log_TransactionAmt" in res.columns
    assert "is_high_value_amt" in res.columns
    assert "fractional_amt" in res.columns


def test_email_family(mock_df: pd.DataFrame) -> None:
    builder = EmailFamilyBuilder()
    res = builder.transform(mock_df)
    assert "TransactionID" in res.columns
    assert "is_domain_match" in res.columns
    assert "p_is_free_domain" in res.columns
    assert "r_is_free_domain" in res.columns


def test_device_family(mock_df: pd.DataFrame) -> None:
    builder = DeviceFamilyBuilder()
    res = builder.transform(mock_df)
    assert "TransactionID" in res.columns
    assert "device_is_windows" in res.columns
    assert "device_is_ios_apple" in res.columns
    assert "device_is_android" in res.columns


def test_address_family(mock_df: pd.DataFrame) -> None:
    builder = AddressFamilyBuilder()
    res = builder.transform(mock_df)
    assert "TransactionID" in res.columns
    assert "is_addr1_missing" in res.columns
    assert "addr1_frequency" in res.columns


def test_card_family(mock_df: pd.DataFrame) -> None:
    builder = CardFamilyBuilder()
    res = builder.transform(mock_df)
    assert "TransactionID" in res.columns
    assert "card1_frequency" in res.columns
    assert "card_is_visa" in res.columns
    assert "card_is_mastercard" in res.columns


def test_tracker_and_integrator(tmp_path: Path, mock_df: pd.DataFrame) -> None:
    tracker = FeatureFamilyTracker()
    integrator = FeatureFamilyIntegrator(tmp_path / "store")

    # Builders dict
    builders = {
        "transaction": TransactionFamilyBuilder(),
        "identity": IdentityFamilyBuilder(),
    }
    
    families_dfs = {}
    for name, b in builders.items():
        df_fam = b.transform(mock_df)
        families_dfs[name] = df_fam
        tracker.record_family(name, [c for c in df_fam.columns if c != "TransactionID"], ["TransactionID"])

    # Test integration
    unified_path = integrator.integrate(families_dfs, "train", "v1")
    assert unified_path.exists()
    
    df_unified = pd.read_parquet(unified_path)
    assert "ProductCD_count" in df_unified.columns
    assert "identity_completeness_score" in df_unified.columns
    assert df_unified.shape[0] == len(mock_df)

    # Test tracker metadata saving
    meta_json, meta_csv = tracker.save_metadata(tmp_path / "store" / "v1")
    assert meta_json.exists()
    assert meta_csv.exists()

    with open(meta_json) as f:
        catalog = json.load(f)
    assert len(catalog) == 2
