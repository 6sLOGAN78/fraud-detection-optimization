"""Unit tests for Feature Encoding components."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.encoding import (
    EncodingStrategySelector,
    VectorizedLabelEncoder,
    VectorizedFrequencyEncoder,
    VectorizedCountEncoder,
    LeakageSafeTargetEncoder,
    VectorizedOneHotEncoder,
    EncodingValidationGate,
    EncoderRegistry,
)


@pytest.fixture()
def mock_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "TransactionID": range(n),
        "isFraud": rng.choice([0, 1], n, p=[0.9, 0.1]),
        "ProductCD": rng.choice(["W", "H", "C"], n),
        "card4": rng.choice(["visa", "mastercard"], n),
        "DeviceType": rng.choice(["mobile", "desktop", "nan"], n),
        "high_card": [f"user_{idx}" for idx in range(n)],
        "binary_feat": rng.choice(["yes", "no"], n),
    })
    return df


def test_strategy_selector(mock_df: pd.DataFrame) -> None:
    selector = EncodingStrategySelector()
    strategies = selector.select_strategy(mock_df, ["ProductCD", "card4", "DeviceType", "high_card", "binary_feat"])
    
    assert strategies["binary_feat"] == "Label"
    assert strategies["ProductCD"] == "OneHot"
    assert strategies["high_card"] == "Target"


def test_label_encoder() -> None:
    s = pd.Series(["yes", "no", "yes", "maybe", np.nan])
    enc = VectorizedLabelEncoder()
    enc.fit(s)
    
    encoded = enc.transform(s)
    assert encoded.iloc[0] == encoded.iloc[2]
    assert encoded.iloc[4] == -1


def test_frequency_encoder() -> None:
    s = pd.Series(["apple", "apple", "banana", "cherry"])
    enc = VectorizedFrequencyEncoder()
    enc.fit(s)
    
    encoded = enc.transform(s)
    assert encoded.iloc[0] == 0.5
    assert encoded.iloc[2] == 0.25
    assert encoded.iloc[3] == 0.25


def test_count_encoder() -> None:
    s = pd.Series(["apple", "apple", "banana", "cherry"])
    enc = VectorizedCountEncoder()
    enc.fit(s)
    
    encoded_counts = enc.transform(s, mode="count")
    assert encoded_counts.iloc[0] == 2
    assert encoded_counts.iloc[2] == 1
    
    encoded_log = enc.transform(s, mode="log_count")
    assert np.allclose(encoded_log.iloc[0], np.log1p(2))
    
    encoded_pct = enc.transform(s, mode="percentile")
    assert encoded_pct.iloc[0] > encoded_pct.iloc[2]


def test_target_encoder(mock_df: pd.DataFrame) -> None:
    enc = LeakageSafeTargetEncoder(min_samples_leaf=2.0, smoothing=1.0, n_folds=3)
    
    # Check out-of-fold fit_transform
    train_encoded = enc.fit_transform(mock_df["high_card"], mock_df["isFraud"])
    assert train_encoded.shape[0] == len(mock_df)
    assert not train_encoded.isnull().any()
    
    # Check test transform
    test_encoded = enc.transform(mock_df["high_card"])
    assert test_encoded.shape[0] == len(mock_df)
    assert not test_encoded.isnull().any()


def test_one_hot_encoder() -> None:
    s = pd.Series(["apple"] * 20 + ["banana"] * 10 + ["cherry"] * 1, name="s") # total 31
    enc = VectorizedOneHotEncoder(threshold=0.1)
    enc.fit(s)
    
    encoded = enc.transform(s)
    # apple (>10%), banana (>10%) should be individual cols, cherry (<10%) in unknown
    assert "s_apple" in encoded.columns
    assert "s_banana" in encoded.columns
    assert "s_unknown" in encoded.columns
    
    # cherry should be marked unknown
    assert encoded.loc[30, "s_unknown"] == 1
    assert encoded.loc[0, "s_unknown"] == 0


def test_validation_gate() -> None:
    df = pd.DataFrame({
        "col_a": [1, 2, np.nan],
        "col_b": [3, 4, 5],
        "col_c": [1.0, 1.0, 1.0],
    })
    
    gate = EncodingValidationGate()
    report = gate.validate(df)
    
    assert report["nan_columns_count"] == 1
    assert report["nan_columns"] == ["col_a"]
    assert report["constant_columns_count"] == 1
    assert report["constant_columns"] == ["col_c"]
    assert report["status"] == "WARN"


def test_registry(tmp_path: Path) -> None:
    reg = EncoderRegistry()
    enc_label = VectorizedLabelEncoder()
    reg.register("feat_a", "Label", enc_label, cardinality=5)
    
    # Save bundle
    bundle_path, manifest_path = reg.save_bundle(tmp_path)
    assert bundle_path.exists()
    assert manifest_path.exists()
    
    with open(manifest_path) as f:
        meta = json.load(f)
    assert meta["version"] == "v1.0"
    assert meta["registry"][0]["feature_name"] == "feat_a"
    
    with open(bundle_path, "rb") as f:
        loaded_encs = pickle.load(f)
    assert "feat_a" in loaded_encs
