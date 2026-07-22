"""Unit tests for the Enterprise Feature Store Foundation, testing OfflineStore, OnlineStore, FeatureRegistry, FeatureCatalog, and FeatureStoreClient APIs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.store import (
    FeatureViewMetadata,
    FeatureRegistry,
    FeatureCatalog,
    OfflineStore,
    OnlineStore,
    FeatureStoreClient,
)


def test_feature_registry_and_catalog(tmp_path: Path) -> None:
    registry_file = tmp_path / "registry.json"
    registry = FeatureRegistry(registry_file)
    catalog = FeatureCatalog(registry)
    
    meta_train = FeatureViewMetadata(
        name="missing_train",
        entity_id="TransactionID",
        features=["missing_ratio", "completeness_score"],
        source_path="/path/to/train.parquet",
    )
    
    registry.register_view(meta_train)
    
    # Reload and test state persistence
    registry_reloaded = FeatureRegistry(registry_file)
    assert "missing_train" in registry_reloaded.views
    
    # Catalog checks
    views = catalog.list_views()
    assert len(views) == 1
    assert views[0]["name"] == "missing_train"
    
    # Search checks
    search_res = catalog.search_features("completeness")
    assert len(search_res) == 1
    assert search_res[0]["name"] == "missing_train"


def test_offline_store(tmp_path: Path) -> None:
    offline_dir = tmp_path / "offline"
    store = OfflineStore(offline_dir)
    
    df_features = pd.DataFrame({
        "TransactionID": [100, 101, 102],
        "missing_ratio": [0.1, 0.2, 0.3],
    })
    
    store.save_features("missingness", "v1", df_features)
    
    entity_df = pd.DataFrame({
        "TransactionID": [101, 102, 103],
        "amount": [10.5, 99.0, 4.0],
    })
    
    merged = store.get_historical_features(entity_df, [("missingness", "v1")], entity_id="TransactionID")
    
    # TransactionID 101, 102 should join successfully, 103 should have NaN
    assert merged.shape[0] == 3
    assert merged.loc[merged["TransactionID"] == 101, "missing_ratio"].values[0] == 0.2
    assert merged.loc[merged["TransactionID"] == 102, "missing_ratio"].values[0] == 0.3
    assert np.isnan(merged.loc[merged["TransactionID"] == 103, "missing_ratio"].values[0])


def test_online_store(tmp_path: Path) -> None:
    db_path = tmp_path / "online.db"
    store = OnlineStore(db_path)
    
    df_features = pd.DataFrame({
        "TransactionID": [10, 11, 12],
        "missing_ratio": [0.05, 0.08, 0.02],
        "pattern": ["A", "B", "C"],
    })
    
    store.write_features("missingness", df_features, entity_id="TransactionID")
    
    # Read online features
    results = store.get_online_features(
        entity_keys=[11, 12, 13],
        view_name="missingness",
        feature_names=["missing_ratio", "pattern"],
        entity_id="TransactionID",
    )
    
    assert len(results) == 3
    assert results[0]["TransactionID"] == 11
    assert results[0]["pattern"] == "B"
    assert results[0]["missing_ratio"] == 0.08
    
    assert results[1]["TransactionID"] == 12
    assert results[1]["pattern"] == "C"
    assert results[1]["missing_ratio"] == 0.02
    
    # Non-existent key should return None placeholders
    assert results[2]["TransactionID"] == 13
    assert results[2]["pattern"] is None


def test_feature_store_client_workflow(tmp_path: Path) -> None:
    registry_file = tmp_path / "registry.json"
    offline_dir = tmp_path / "offline"
    online_db = tmp_path / "online.db"
    
    client = FeatureStoreClient(registry_file, offline_dir, online_db)
    
    df_features = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "missing_ratio": [0.1, 0.2, 0.15],
        "completeness_score": [0.9, 0.8, 0.85],
    })
    
    client.register_feature_view(
        name="missingness",
        entity_id="TransactionID",
        features=["missing_ratio", "completeness_score"],
        source_path=str(registry_file),
    )
    
    # Unified Ingest
    client.ingest("missingness", df_features)
    
    # Historical retrieval
    entity_df = pd.DataFrame({"TransactionID": [2, 3]})
    hist = client.get_historical_features(entity_df, [("missingness", "v1")])
    assert hist.columns.tolist() == ["TransactionID", "missing_ratio", "completeness_score"]
    assert hist["completeness_score"].tolist() == [0.8, 0.85]
    
    # Online retrieval
    online_res = client.get_online_features([2, 3], "missingness", ["missing_ratio"])
    assert len(online_res) == 2
    assert online_res[0]["missing_ratio"] == 0.2
    assert online_res[1]["missing_ratio"] == 0.15
