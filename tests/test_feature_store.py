"""Unit tests for the Enterprise Feature Store operations, testing Security RBAC, FeatureLineage, database serving latency, and PSI drift monitoring."""

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
    FeatureLineage,
    AccessController,
)


def test_rbac_access_controller() -> None:
    rbac = AccessController()
    
    # Authenticates valid tokens
    assert rbac.authenticate_token("ADMIN_TOKEN_999") == "ADMIN"
    assert rbac.authenticate_token("RW_TOKEN_888") == "READ_WRITE"
    assert rbac.authenticate_token("RO_TOKEN_777") == "READ_ONLY"
    
    # Throws PermissionError on invalid tokens
    with pytest.raises(PermissionError, match="Invalid or missing API security token"):
        rbac.authenticate_token("INVALID_TOKEN")
        
    # Check authorization roles
    rbac.authorize("ADMIN_TOKEN_999", ["READ_ONLY"])
    rbac.authorize("RW_TOKEN_888", ["ADMIN", "READ_WRITE"])
    
    with pytest.raises(PermissionError, match="lacks permission"):
        rbac.authorize("RO_TOKEN_777", ["ADMIN", "READ_WRITE"])


def test_feature_registry_with_lineage(tmp_path: Path) -> None:
    registry_file = tmp_path / "registry.json"
    registry = FeatureRegistry(registry_file)
    
    lineage = FeatureLineage(
        source_dataset="/raw/train.parquet",
        pipeline_stage="feature_missing",
        transformation_type="vectorized",
        description="computes column missing count",
    )
    
    meta_train = FeatureViewMetadata(
        name="missing_train",
        entity_id="TransactionID",
        features=["missing_ratio", "completeness_score"],
        source_path="/path/to/train.parquet",
        owner="Data Team",
        tags=["missingness"],
        description="Contains transaction missingness metrics",
        lineage=lineage,
    )
    
    registry.register_view(meta_train)
    
    registry_reloaded = FeatureRegistry(registry_file)
    view = registry_reloaded.get_view("missing_train")
    assert view is not None
    assert view.lineage is not None
    assert view.lineage.pipeline_stage == "feature_missing"


def test_feature_store_observability_and_monitoring(tmp_path: Path) -> None:
    registry_file = tmp_path / "registry.json"
    offline_dir = tmp_path / "offline"
    online_db = tmp_path / "online.db"
    
    client = FeatureStoreClient(registry_file, offline_dir, online_db)
    admin_token = "ADMIN_TOKEN_999"
    
    # 1. Prepare simple mock datasets
    df_train = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "missing_ratio": [0.1, 0.2, 0.15],
    })
    
    df_test = pd.DataFrame({
        "TransactionID": [4, 5, 6],
        "missing_ratio": [0.12, 0.22, 0.14],  # minor shift
    })
    
    client.register_feature_view(
        name="missingness_train_features",
        entity_id="TransactionID",
        features=["missing_ratio"],
        source_path="mock_tr",
        token=admin_token,
    )
    
    client.register_feature_view(
        name="missingness_test_features",
        entity_id="TransactionID",
        features=["missing_ratio"],
        source_path="mock_te",
        token=admin_token,
    )
    
    # Ingest both
    client.ingest("missingness_train_features", df_train, token=admin_token)
    client.ingest("missingness_test_features", df_test, token=admin_token)
    
    # 2. Assert PSI calculations
    psi = client.monitor.calculate_psi(df_train["missing_ratio"], df_test["missing_ratio"], bins=5)
    assert isinstance(psi, float)
    assert psi >= 0.0
    
    # 3. Assert online latency benchmark
    avg_latency = client.monitor.measure_online_latency(
        client.online_store, "missingness_test_features", [4, 5], ["missing_ratio"]
    )
    assert isinstance(avg_latency, float)
    
    # 4. Assert general monitoring report matches structure
    report = client.monitor.get_monitoring_report(client, [4, 5])
    assert "timestamp" in report
    assert "online_store" in report
    assert "offline_store" in report
    assert "monitored_views" in report
    assert len(report["monitored_views"]) == 2
