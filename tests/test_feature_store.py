"""Unit tests for the Enterprise Feature Store Operations, testing Security RBAC, FeatureLineage details, and Serving logic."""

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
        
    with pytest.raises(PermissionError, match="Invalid or missing API security token"):
        rbac.authenticate_token(None)
        
    # Test authorization roles
    rbac.authorize("ADMIN_TOKEN_999", ["READ_ONLY"])  # Admin can access anything
    rbac.authorize("RW_TOKEN_888", ["ADMIN", "READ_WRITE"])  # Success
    
    with pytest.raises(PermissionError, match="lacks permission"):
        rbac.authorize("RO_TOKEN_777", ["ADMIN", "READ_WRITE"])


def test_feature_registry_with_lineage(tmp_path: Path) -> None:
    registry_file = tmp_path / "registry.json"
    registry = FeatureRegistry(registry_file)
    catalog = FeatureCatalog(registry)
    
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
    
    # Reload and test details
    registry_reloaded = FeatureRegistry(registry_file)
    view = registry_reloaded.get_view("missing_train")
    assert view is not None
    assert view.owner == "Data Team"
    assert view.lineage is not None
    assert view.lineage.pipeline_stage == "feature_missing"


def test_feature_store_client_rbac_workflow(tmp_path: Path) -> None:
    registry_file = tmp_path / "registry.json"
    offline_dir = tmp_path / "offline"
    online_db = tmp_path / "online.db"
    
    client = FeatureStoreClient(registry_file, offline_dir, online_db)
    
    df_features = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "missing_ratio": [0.1, 0.2, 0.15],
        "completeness_score": [0.9, 0.8, 0.85],
    })
    
    lineage = FeatureLineage(
        source_dataset="/path/to/src",
        pipeline_stage="missingness_stage",
    )
    
    # 1. Register view fails without token
    with pytest.raises(PermissionError):
        client.register_feature_view(
            name="missingness",
            entity_id="TransactionID",
            features=["missing_ratio", "completeness_score"],
            source_path="src_path",
            lineage=lineage,
            token=None,
        )
        
    # Register view successfully with ADMIN token
    client.register_feature_view(
        name="missingness",
        entity_id="TransactionID",
        features=["missing_ratio", "completeness_score"],
        source_path="src_path",
        lineage=lineage,
        token="ADMIN_TOKEN_999",
    )
    
    # 2. Ingest fails with RO token or invalid token
    with pytest.raises(PermissionError):
        client.ingest("missingness", df_features, token="RO_TOKEN_777")
        
    # Ingest succeeds with RW token
    client.ingest("missingness", df_features, token="RW_TOKEN_888")
    
    # 3. Read features fails with invalid token
    with pytest.raises(PermissionError):
        client.get_online_features([2, 3], "missingness", ["missing_ratio"], token="BAD_TOKEN")
        
    # Read features succeeds with RO token
    online_res = client.get_online_features([2, 3], "missingness", ["missing_ratio"], token="RO_TOKEN_777")
    assert len(online_res) == 2
    assert online_res[0]["missing_ratio"] == 0.2
    assert online_res[1]["missing_ratio"] == 0.15
