"""Unit tests verifying individual feature selector filters and pipeline orchestrator outputs."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.feature_selection.selectors import NullSelector, VarianceSelector, CorrelationSelector, ImportanceSelector, MutualInformationSelector, SHAPSelector, PermutationImportanceSelector, RFESelector, SequentialSelector, BorutaSelector, SimulatedAnnealingSelector, FeatureStabilitySelector, FeatureSelectionValidator
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


def test_shap_selector() -> None:
    # Set up mock dataframe with predictive feature and noise feature
    rng = np.random.RandomState(42)
    feat_pred = rng.normal(0, 1, 100)
    y = pd.Series((feat_pred > 0.05).astype(int))
    
    # Predictor tracks y very well, noise is pure random
    df = pd.DataFrame({
        "feat_pred": feat_pred + y * 0.5,
        "feat_noise": rng.normal(0, 10, 100),
    })

    # Test SHAPSelector
    sh = SHAPSelector(threshold=0.20, random_state=42)
    transformed_sh = sh.fit_transform(df, y)
    
    assert "feat_pred" in transformed_sh.columns
    assert "feat_noise" not in transformed_sh.columns


def test_permutation_importance_selector() -> None:
    # Set up mock dataframe with predictive feature and noise feature
    rng = np.random.RandomState(42)
    feat_pred = rng.normal(0, 1, 100)
    y = pd.Series((feat_pred > 0.05).astype(int))
    
    # Predictor tracks y very well, noise is pure random
    df = pd.DataFrame({
        "feat_pred": feat_pred + y * 0.5,
        "feat_noise": rng.normal(0, 10, 100),
    })

    # Test PermutationImportanceSelector
    pi = PermutationImportanceSelector(threshold=0.20, random_state=42)
    transformed_pi = pi.fit_transform(df, y)
    
    assert "feat_pred" in transformed_pi.columns
    assert "feat_noise" not in transformed_pi.columns


def test_rfe_selector() -> None:
    # Set up mock dataframe with predictive feature and noise feature
    rng = np.random.RandomState(42)
    feat_pred = rng.normal(0, 1, 100)
    y = pd.Series((feat_pred > 0.05).astype(int))
    
    # Predictor tracks y very well, noise is pure random
    df = pd.DataFrame({
        "feat_pred": feat_pred + y * 0.5,
        "feat_noise": rng.normal(0, 10, 100),
    })

    # Test RFESelector (threshold=0.60 ensures rank 2 score 0.50 gets dropped)
    rfe_s = RFESelector(threshold=0.60, random_state=42)
    transformed_rfe = rfe_s.fit_transform(df, y)
    
    assert "feat_pred" in transformed_rfe.columns
    assert "feat_noise" not in transformed_rfe.columns


def test_sequential_selector() -> None:
    # Set up mock dataframe with predictive feature and noise feature
    rng = np.random.RandomState(42)
    feat_pred = rng.normal(0, 1, 100)
    y = pd.Series((feat_pred > 0.05).astype(int))
    
    # Predictor tracks y very well, noise is pure random
    df = pd.DataFrame({
        "feat_pred": feat_pred + y * 0.5,
        "feat_noise": rng.normal(0, 10, 100),
    })

    # Test SequentialSelector (n_features_to_select=1 SFS correctly selects feat_pred)
    sfs = SequentialSelector(n_features_to_select=1, random_state=42)
    transformed_sfs = sfs.fit_transform(df, y)
    
    assert "feat_pred" in transformed_sfs.columns
    assert "feat_noise" not in transformed_sfs.columns


def test_boruta_selector() -> None:
    # Set up mock dataframe with predictive feature and noise feature
    rng = np.random.RandomState(42)
    feat_pred = rng.normal(0, 1, 100)
    y = pd.Series((feat_pred > 0.05).astype(int))
    
    # Predictor tracks y very well, noise is pure random
    df = pd.DataFrame({
        "feat_pred": feat_pred + y * 0.5,
        "feat_noise": rng.normal(0, 10, 100),
    })

    # Test BorutaSelector (threshold=0.50 ensures noise feature is dropped)
    boruta_s = BorutaSelector(threshold=0.50, n_iterations=5, random_state=42)
    transformed_boruta = boruta_s.fit_transform(df, y)
    
    assert "feat_pred" in transformed_boruta.columns
    assert "feat_noise" not in transformed_boruta.columns


def test_simulated_annealing_selector() -> None:
    # Set up mock dataframe with predictive feature and noise feature
    rng = np.random.RandomState(42)
    feat_pred = rng.normal(0, 1, 100)
    y = pd.Series((feat_pred > 0.05).astype(int))
    
    # Predictor tracks y very well, noise is pure random
    df = pd.DataFrame({
        "feat_pred": feat_pred + y * 0.5,
        "feat_noise": rng.normal(0, 10, 100),
    })

    # Test SimulatedAnnealingSelector
    sa_s = SimulatedAnnealingSelector(threshold=0.05, n_iterations=10, random_state=42)
    transformed_sa = sa_s.fit_transform(df, y)
    
    assert "feat_pred" in transformed_sa.columns
    assert "feat_noise" not in transformed_sa.columns


def test_feature_stability_selector() -> None:
    # Set up mock dataframe with predictive feature and noise feature
    rng = np.random.RandomState(42)
    feat_pred = rng.normal(0, 1, 100)
    y = pd.Series((feat_pred > 0.05).astype(int))
    
    # Predictor tracks y very well, noise is pure random
    df = pd.DataFrame({
        "feat_pred": feat_pred + y * 0.5,
        "feat_noise": rng.normal(0, 10, 100),
    })

    # Test FeatureStabilitySelector (threshold=10.0 ensures noise feature is dropped)
    stability_s = FeatureStabilitySelector(threshold=10.0, n_bootstraps=5, random_state=42)
    transformed_stability = stability_s.fit_transform(df, y)
    
    assert "feat_pred" in transformed_stability.columns
    assert "feat_noise" not in transformed_stability.columns


def test_feature_selection_validator() -> None:
    # 1. Empty features check
    df_empty = pd.DataFrame()
    validator = FeatureSelectionValidator()
    with pytest.raises(ValueError, match="Quality Gate Failed: Selected features list lies empty"):
        validator.fit(df_empty)

    # 2. Duplicate columns check
    df_dup = pd.DataFrame([[1, 2]], columns=["col1", "col1"])
    with pytest.raises(ValueError, match="Quality Gate Failed: DataFrame contains duplicate columns"):
        validator.fit(df_dup)

    # 3. Variance check
    df_no_var = pd.DataFrame({"feat_stable": [1.0, 2.0, 3.0], "feat_no_var": [0.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="Quality Gate Failed: Feature 'feat_no_var' has near-zero/NaN variance"):
        validator.fit(df_no_var)

    # 4. Collinearity check
    df_coll = pd.DataFrame({
        "col1": [1.0, 2.0, 3.0],
        "col2": [2.0, 4.0, 6.0]  # Perfectly collinear with col1
    })
    with pytest.raises(ValueError, match="Quality Gate Failed: High collinearity detected"):
        validator.fit(df_coll)

    # 5. Target alignment check
    df_no_align = pd.DataFrame({
        "feat_stable": [1.0, 0.0, 1.0, 0.0],
        "feat_noise": [-1.0, 1.0, 1.0, -1.0]
    })
    y = pd.Series([1, 0, 1, 0])
    with pytest.raises(ValueError, match="Quality Gate Failed: Feature 'feat_noise' has negligible target MI/corr alignment"):
        validator.fit(df_no_align, y)

    # 6. Valid pass check
    df_valid = pd.DataFrame({
        "feat_pred": [1.0, 0.0, 1.0, 0.0],
        "feat_pred2": [0.1, 1.2, 0.9, 0.2]
    })
    validator.fit(df_valid, y)  # Should pass without error


def test_feature_registry_generation() -> None:
    import json
    from pathlib import Path
    registry_path = Path("data/feature_store_engineered/v1/feature_registry.json")
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
        assert "version" in registry
        assert "timestamp" in registry
        assert "access_control" in registry
        assert "features" in registry
        assert "lifecycle_management" in registry
        assert registry["version"] == "v1.0"


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
