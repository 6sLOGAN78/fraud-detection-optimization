"""Unit tests for Feature Aggregation components."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.aggregations import (
    AggregationGroupBuilder,
    VectorizedAggregationEngine,
    RollingAggregationEngine,
    AggregationValidationGate,
    AggregationRegistry,
)


@pytest.fixture()
def mock_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 10
    df = pd.DataFrame({
        "TransactionID": range(1001, 1001 + n),
        "card1": [101, 101, 102, 101, 102, 103, 101, 102, 103, 101],
        "dist1": [10.0, np.nan, 20.0, 15.0, 25.0, np.nan, 12.0, 22.0, 5.0, 18.0],
        "TransactionAmt": [100.0, 150.0, 200.0, 120.0, 250.0, 300.0, 110.0, 210.0, 310.0, 130.0],
    })
    return df


def test_group_builder() -> None:
    builder = AggregationGroupBuilder()
    keys = builder.get_group_keys()
    assert "card1" in keys
    assert "card2" in keys
    assert "P_emaildomain" in keys


def test_vectorized_aggregation_engine(mock_df: pd.DataFrame) -> None:
    # We aggregate dist1 grouped by card1
    # Train = first 8 rows, Test = last 2 rows
    df_train = mock_df.iloc[:8]
    df_test = mock_df.iloc[8:]
    
    engine = VectorizedAggregationEngine(group_col="card1", agg_col="dist1")
    engine.fit(df_train)
    
    # Test values
    mean_val = engine.transform(df_test, stat_type="mean")
    assert len(mean_val) == 2
    # card1 = 103 on row 8, card1 = 101 on row 9
    # check if not NaN
    assert not mean_val.isnull().any()


def test_rolling_aggregation_engine(mock_df: pd.DataFrame) -> None:
    engine = RollingAggregationEngine(group_col="card1", agg_col="TransactionAmt", window_size=2)
    res = engine.compute_rolling(mock_df)
    
    assert res.shape[0] == len(mock_df)
    assert f"card1_TransactionAmt_exp_mean" in res.columns
    assert f"card1_TransactionAmt_exp_cnt" in res.columns
    assert f"card1_TransactionAmt_roll_mean" in res.columns

    # Row 0: card1=101, val=100. Previous: None. Exp cnt=0, exp mean=0.0
    assert res.loc[0, "card1_TransactionAmt_exp_cnt"] == 0
    
    # Row 1: card1=101, val=150. Previous: row 0 (100). Exp cnt=1, exp mean=100.0
    assert res.loc[1, "card1_TransactionAmt_exp_cnt"] == 1
    assert res.loc[1, "card1_TransactionAmt_exp_mean"] == 100.0

    # Row 3: card1=101, val=120. Previous: row 0, 1 (100, 150). Exp cnt=2, exp mean=125.0
    assert res.loc[3, "card1_TransactionAmt_exp_cnt"] == 2
    assert res.loc[3, "card1_TransactionAmt_exp_mean"] == 125.0


def test_validation_gate() -> None:
    df = pd.DataFrame({
        "col_a": [1, 2, np.nan],
        "col_b": [3, 4, 5],
        "col_c": [1.0, 1.0, 1.0],
    })
    
    gate = AggregationValidationGate()
    report = gate.validate(df)
    
    assert report["nan_columns_count"] == 1
    assert report["nan_columns"] == ["col_a"]
    assert report["constant_columns_count"] == 1
    assert report["constant_columns"] == ["col_c"]
    assert report["status"] == "WARN"


def test_registry(tmp_path: Path) -> None:
    reg = AggregationRegistry()
    reg.register("feat_a", "mean", "TransactionAmt", "card1")
    
    manifest_path, csv_path = reg.save_catalog(tmp_path)
    assert manifest_path.exists()
    assert csv_path.exists()
    
    with open(manifest_path) as f:
        meta = json.load(f)
    assert meta["version"] == "v1.0"
    assert meta["registry"][0]["feature_name"] == "feat_a"
    
    df_cat = pd.read_csv(csv_path)
    assert len(df_cat) == 1
    assert df_cat.loc[0, "feature_name"] == "feat_a"
