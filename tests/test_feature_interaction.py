"""Unit tests for Interaction Features components."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.interactions import (
    VectorizedInteractionEngine,
    AutomaticInteractionDiscoveryEngine,
    FeatureExplosionController,
    InteractionValidationGate,
    InteractionRegistry,
)


def test_vectorized_interaction_engine() -> None:
    engine = VectorizedInteractionEngine(default_val=0.0)
    
    a = pd.Series([10.0, 5.0, np.nan, 2.0])
    b = pd.Series([2.0, 0.0, 3.0, np.nan])
    
    # 1. Multiplication
    res_mult = engine.compute_interaction(a, b, "multiplication")
    assert res_mult.iloc[0] == pytest.approx(20.0)
    assert res_mult.iloc[1] == pytest.approx(0.0)
    # NaN -> 0 * 3 = 0.0
    assert res_mult.iloc[2] == pytest.approx(0.0)
    # NaN -> 2.0 * 0.0 = 0.0
    assert res_mult.iloc[3] == pytest.approx(0.0)
    
    # 2. Safety Division
    res_div = engine.compute_interaction(a, b, "division")
    # 10.0 / 2.0 = 5.0
    assert res_div.iloc[0] == pytest.approx(5.0)
    # 5.0 / 0.0 -> NaN -> default_val = 0.0
    assert res_div.iloc[1] == pytest.approx(0.0)


def test_automatic_interaction_discovery_engine() -> None:
    cols = [
        "TransactionAmt",
        "dist1",
        "card1_TransactionAmt_mean",
        "addr1_dist1_roll_mean",
        "some_unrelated_column",
    ]
    
    engine = AutomaticInteractionDiscoveryEngine(target_cols=["TransactionAmt", "dist1"])
    pairings = engine.discover_pairings(cols)
    
    feat_names = [p[0] for p in pairings]
    assert "TransactionAmt_x_card1_TransactionAmt_mean" in feat_names
    assert "TransactionAmt_div_card1_TransactionAmt_mean" in feat_names
    assert "dist1_x_addr1_dist1_roll_mean" in feat_names
    assert "dist1_div_addr1_dist1_roll_mean" in feat_names
    assert len(pairings) == 4


def test_feature_explosion_controller() -> None:
    df = pd.DataFrame({
        "TransactionID": [100, 101, 102],
        "inter_a": [1.0, 1.0, 1.0],      # zero variance
        "inter_b": [2.5, 3.8, 1.1],      # non-zero variance
    })
    
    explorer = FeatureExplosionController(variance_threshold=0.01)
    filtered = explorer.filter_features(df)
    
    assert "TransactionID" in filtered
    assert "inter_b" in filtered
    assert "inter_a" not in filtered


def test_interaction_validation_gate() -> None:
    df = pd.DataFrame({
        "inter_v1": [1.2, 0.8, np.nan],
        "inter_v2": [0.0, 0.0, 0.0],
    })
    
    gate = InteractionValidationGate()
    report = gate.validate(df)
    
    assert report["nan_columns_count"] == 1
    assert report["constant_columns_count"] == 1
    assert report["status"] == "WARN"


def test_interaction_registry(tmp_path: Path) -> None:
    reg = InteractionRegistry()
    reg.register("feat_inter", "TransactionAmt", "card1_TransactionAmt_mean", "multiplication")
    
    manifest_path, csv_path = reg.save_catalog(tmp_path)
    assert manifest_path.exists()
    assert csv_path.exists()
    
    with open(manifest_path) as f:
        meta = json.load(f)
    assert meta["registry"][0]["feature_name"] == "feat_inter"
