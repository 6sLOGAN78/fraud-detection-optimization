"""Unit tests for FeatureEngineeringPipeline — Part 4.1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.pipeline import FeatureEngineeringPipeline


@pytest.fixture()
def mock_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates mock train/test datasets."""
    rng = np.random.default_rng(42)
    n = 100

    df_train = pd.DataFrame({
        "TransactionID": range(n),
        "TransactionDT": rng.integers(0, 86400 * 2, n),
        "TransactionAmt": rng.uniform(1.0, 500.0, n),
        "C1": rng.integers(0, 10, n),
        "ProductCD": rng.choice(["W", "H", "C", "M"], n),
    })
    df_test = pd.DataFrame({
        "TransactionID": range(n, n + n),
        "TransactionDT": rng.integers(86400 * 2, 86400 * 4, n),
        "TransactionAmt": rng.uniform(1.0, 500.0, n),
        "C1": rng.integers(0, 10, n),
        "ProductCD": rng.choice(["W", "H", "C", "M"], n),
    })

    return df_train, df_test


@pytest.fixture()
def pipeline(tmp_path: Path) -> FeatureEngineeringPipeline:
    pip = FeatureEngineeringPipeline(threshold=0.95)
    # Override store output directory to use tmp_path
    pip.store.store_dir = tmp_path / "feature_store"
    pip.store.store_dir.mkdir(parents=True, exist_ok=True)
    return pip


def test_input_validation(pipeline: FeatureEngineeringPipeline) -> None:
    # Test empty dataframe behavior
    with pytest.raises(ValueError, match="Input dataframe is empty"):
        pipeline.validator.validate_inputs(pd.DataFrame())

    # Test missing TransactionID behavior
    with pytest.raises(KeyError, match="TransactionID column is required"):
        pipeline.validator.validate_inputs(pd.DataFrame({"A": [1, 2]}))


def test_fit_transform_successful(pipeline: FeatureEngineeringPipeline, mock_datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_train, df_test = mock_datasets
    
    trn_out, tst_out = pipeline.fit_transform(df_train, df_test, version="v_test")

    # Assert new features are created in train
    assert "hour_of_day" in trn_out.columns
    assert "log_TransactionAmt" in trn_out.columns
    assert "amt_ratio_C1" in trn_out.columns
    assert "ProductCD_freq" in trn_out.columns

    # Assert new features are created in test
    assert "hour_of_day" in tst_out.columns
    assert "log_TransactionAmt" in tst_out.columns
    assert "amt_ratio_C1" in tst_out.columns
    assert "ProductCD_freq" in tst_out.columns

    # Check store structure
    store_dir = pipeline.store.store_dir / "v_test"
    assert (store_dir / "train_features.parquet").exists()
    assert (store_dir / "test_features.parquet").exists()
    assert (store_dir / "manifest.json").exists()


def test_feature_manifest_catalog(pipeline: FeatureEngineeringPipeline, mock_datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_train, df_test = mock_datasets
    
    pipeline.fit_transform(df_train, df_test, version="v_test")
    
    manifest_path = pipeline.store.store_dir / "v_test" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["version"] == "v_test"
    catalog = manifest["features_catalog"]
    
    # Check that registered features are cataloged
    feature_names = [f["name"] for f in catalog]
    assert "hour_of_day" in feature_names
    assert "log_TransactionAmt" in feature_names
    assert "amt_ratio_C1" in feature_names
    assert "ProductCD_freq" in feature_names
