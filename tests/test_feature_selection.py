"""Unit tests verifying individual feature selector filters and pipeline orchestrator outputs."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.feature_selection.selectors import NullSelector, VarianceSelector, CorrelationSelector, ImportanceSelector, MutualInformationSelector
from src.feature_selection.pipeline import FeatureSelectionPipeline


def test_individual_selectors() -> None:
    # Set up mock dataframe where feat_ok is uncorrelated
    df = pd.DataFrame({
        "feat_ok": [3.0, 1.0, 4.5, 2.0, 5.0],        # uncorrelated with feat_corr
        "feat_null": [None, None, 1.0, None, 2.0],  # 60% nulls
        "feat_zero": [2.5, 2.5, 2.5, 2.5, 2.5],      # zero variance
        "feat_corr1": [1.0, 2.0, 3.0, 4.0, 5.0],     # collinear pair
        "feat_corr2": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    y = pd.Series([0, 1, 0, 1, 0])

    # 1. Test NullSelector
    ns = NullSelector(threshold=0.50)
    transformed_ns = ns.fit_transform(df)
    assert "feat_null" not in transformed_ns.columns
    assert "feat_ok" in transformed_ns.columns

    # 2. Test VarianceSelector
    vs = VarianceSelector(threshold=0.0)
    transformed_vs = vs.fit_transform(df)
    assert "feat_zero" not in transformed_vs.columns
    assert "feat_ok" in transformed_vs.columns

    # 3. Test CorrelationSelector
    cs = CorrelationSelector(threshold=0.95)
    transformed_cs = cs.fit_transform(df)
    cols = list(transformed_cs.columns)
    
    # Exactly one of the pair should survive
    assert ("feat_corr1" in cols and "feat_corr2" not in cols) or ("feat_corr2" in cols and "feat_corr1" not in cols)


def test_importance_selector() -> None:
    # Set up mock dataframe with predictive feature and noise feature
    rng = np.random.RandomState(42)
    feat_pred = rng.normal(0, 1, 100)
    y = pd.Series((feat_pred > 0.05).astype(int))
    
    # Predictor tracks y very well, noise is pure random
    df = pd.DataFrame({
        "feat_pred": feat_pred + y * 0.5,
        "feat_noise": rng.normal(0, 10, 100),
    })

    # Test ImportanceSelector
    im = ImportanceSelector(threshold=0.20, random_state=42)
    transformed_im = im.fit_transform(df, y)
    
    assert "feat_pred" in transformed_im.columns
    # Noise should have very low normalized importance and be dropped
    assert "feat_noise" not in transformed_im.columns


def test_mutual_information_selector() -> None:
    # Set up mock dataframe with predictive feature and noise feature
    rng = np.random.RandomState(42)
    feat_pred = rng.normal(0, 1, 100)
    y = pd.Series((feat_pred > 0.05).astype(int))
    
    # Predictor tracks y very well, noise is pure random
    df = pd.DataFrame({
        "feat_pred": feat_pred + y * 0.5,
        "feat_noise": rng.normal(0, 10, 100),
    })

    # Test MutualInformationSelector
    mi = MutualInformationSelector(threshold=0.20, random_state=42)
    transformed_mi = mi.fit_transform(df, y)
    
    assert "feat_pred" in transformed_mi.columns
    assert "feat_noise" not in transformed_mi.columns


def test_selection_pipeline() -> None:
    # Pipeline integration check
    df = pd.DataFrame({
        "feat_ok": [3.0, 1.0, 4.5, 2.0, 5.0],
        "feat_null": [None, None, 1.0, None, 2.0],  # 60% nulls
        "feat_zero": [2.5, 2.5, 2.5, 2.5, 2.5],      # zero variance
    })
    y = pd.Series([0, 1, 0, 1, 0])

    null_sel = NullSelector(threshold=0.50)
    var_sel = VarianceSelector(threshold=0.0)
    
    pipeline = FeatureSelectionPipeline([null_sel, var_sel])
    transformed = pipeline.fit_transform(df, y)
    
    assert list(transformed.columns) == ["feat_ok"]
    summary = pipeline.get_summary_report()
    assert summary["total_initial_features"] == 3
    assert summary["total_final_features"] == 1
    assert "feat_null" in summary["dropped_by_selector"]["NullSelector"]
    assert "feat_zero" in summary["dropped_by_selector"]["VarianceSelector"]
