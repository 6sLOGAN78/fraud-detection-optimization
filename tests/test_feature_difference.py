"""Unit tests for Difference Features components."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.differences import (
    VectorizedDifferenceEngine,
    AutomaticDifferenceDiscoveryEngine,
    DifferenceValidationGate,
    DifferenceRegistry,
)


def test_vectorized_difference_engine() -> None:
    engine = VectorizedDifferenceEngine(default_val=0.0)
    
    num = pd.Series([10.0, 0.0, np.nan, 5.0])
    den = pd.Series([2.0, 0.0, 5.0, np.nan])
    
    res = engine.compute_difference(num, den)
    
    # 10.0 - 2.0 = 8.0
    assert res.iloc[0] == pytest.approx(8.0)
    # 0.0 - 0.0 = 0.0
    assert res.iloc[1] == pytest.approx(0.0)
    # NaN - 5.0 -> 0.0 - 5.0 = -5.0
    assert res.iloc[2] == pytest.approx(-5.0)
    # 5.0 - NaN -> 5.0 - 0.0 = 5.0
    assert res.iloc[3] == pytest.approx(5.0)


def test_automatic_difference_discovery_engine() -> None:
    cols = [
        "TransactionAmt",
        "dist1",
        "card1_TransactionAmt_mean",
        "addr1_dist1_min",
        "ProductCD_TransactionAmt_median",
        "card1_TransactionAmt_roll_mean",
        "some_unrelated_column",
    ]
    
    engine = AutomaticDifferenceDiscoveryEngine(target_numerators=["TransactionAmt", "dist1"])
    pairings = engine.discover_pairings(cols)
    
    feat_names = [p[0] for p in pairings]
    assert "card1_TransactionAmt_mean_diff" in feat_names
    assert "addr1_dist1_min_diff" in feat_names
    assert "ProductCD_TransactionAmt_median_diff" in feat_names
    assert "card1_TransactionAmt_roll_mean_diff" in feat_names
    assert len(pairings) == 4


def test_difference_validation_gate() -> None:
    df = pd.DataFrame({
        "diff_a": [1.2, 0.8, np.nan],
        "diff_b": [0.0, 0.0, 0.0],
    })
    
    gate = DifferenceValidationGate()
    report = gate.validate(df)
    
    assert report["nan_columns_count"] == 1
    assert report["constant_columns_count"] == 1
    assert report["status"] == "WARN"


def test_difference_registry(tmp_path: Path) -> None:
    reg = DifferenceRegistry()
    reg.register("feat_diff", "TransactionAmt", "card1_TransactionAmt_mean")
    
    manifest_path, csv_path = reg.save_catalog(tmp_path)
    assert manifest_path.exists()
    assert csv_path.exists()
    
    with open(manifest_path) as f:
        meta = json.load(f)
    assert meta["registry"][0]["feature_name"] == "feat_diff"
