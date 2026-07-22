"""Unit tests verifying Feature Store Governance Engine, evaluating null tolerance ceilings, variance checks, and PSI threshold states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.store import FeatureViewMetadata, FeatureRegistry, FeatureLineage
from src.data.governance import FeatureGovernanceEngine


def test_feature_governance_checker(tmp_path: Path) -> None:
    registry_file = tmp_path / "registry.json"
    registry = FeatureRegistry(registry_file)
    
    # 1. Register a fake feature view
    view = FeatureViewMetadata(
        name="test_view",
        entity_id="TransactionID",
        features=["feature_ok", "feature_null_heavy", "feature_zero_var", "feature_drifted"],
        source_path="/path/to/source",
        token="ADMIN_TOKEN_999",
    )
    registry.register_view(view)
    registry.save()
    
    # Generate 100 samples to ensure stable PSI calculations
    rng = np.random.RandomState(42)
    
    # feature_ok: similar distribution (normal, mean=0, std=1)
    train_ok = rng.normal(loc=0.0, scale=1.0, size=100)
    test_ok = rng.normal(loc=0.05, scale=0.98, size=100)
    
    # feature_null_heavy: many NaNs in test/train
    train_null = [None] * 40 + list(rng.normal(0, 1, 60))
    test_null = [None] * 45 + list(rng.normal(0, 1, 55))
    
    # feature_zero_var: constant value
    train_zero = [5.0] * 100
    test_zero = [5.0] * 100
    
    # feature_drifted: significant shift in mean
    train_drift = rng.normal(loc=0.0, scale=1.0, size=100)
    test_drift = rng.normal(loc=2.0, scale=1.0, size=100)  # highly drifted
    
    df_train = pd.DataFrame({
        "TransactionID": list(range(100)),
        "feature_ok": train_ok,
        "feature_null_heavy": train_null,
        "feature_zero_var": train_zero,
        "feature_drifted": train_drift,
    })
    
    df_test = pd.DataFrame({
        "TransactionID": list(range(100, 200)),
        "feature_ok": test_ok,
        "feature_null_heavy": test_null,
        "feature_zero_var": test_zero,
        "feature_drifted": test_drift,
    })
    
    engine = FeatureGovernanceEngine(registry_file)
    report = engine.audit_features(
        df_train=df_train,
        df_test=df_test,
        null_threshold=0.20,  # nulls (40% and 45%) exceed this limit
        drift_threshold=0.35,  # 0.35 threshold accommodates minor sample variance
        token="ADMIN_TOKEN_999"
    )
    
    # Verify report status and view logs
    assert report["audited_views_count"] == 1
    assert report["status"] == "WARNING"
    
    view_log = report["view_logs"][0]
    assert view_log["view_name"] == "test_view"
    assert view_log["status"] == "FAIL"
    
    features_report = {f["feature_name"]: f for f in view_log["features_audited"]}
    
    # feature_ok passes all
    assert features_report["feature_ok"]["checks"]["missingness"] == "PASS"
    assert features_report["feature_ok"]["checks"]["variance"] == "PASS"
    assert features_report["feature_ok"]["checks"]["drift"] == "PASS"
    
    # feature_null_heavy fails missingness
    assert features_report["feature_null_heavy"]["checks"]["missingness"] == "FAIL"
    
    # feature_zero_var fails variance
    assert features_report["feature_zero_var"]["checks"]["variance"] == "FAIL"
    
    # feature_drifted fails drift
    assert features_report["feature_drifted"]["checks"]["drift"] == "FAIL"
    
    # Verify the registry was updated to tag the view as CRITICAL
    registry_reloaded = FeatureRegistry(registry_file)
    updated_view = registry_reloaded.get_view("test_view")
    assert updated_view is not None
    assert "CRITICAL" in updated_view.tags
