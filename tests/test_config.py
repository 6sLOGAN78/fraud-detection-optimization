"""Unit tests validating ConfigurationManager loading and type assertions."""

import pytest
from src.config.config import ConfigurationManager, ProjectConfig


def test_configuration_manager() -> None:
    manager = ConfigurationManager()
    config = manager.get_config()
    
    assert isinstance(config, ProjectConfig)
    assert config.project_name == "ieee-cis-fraud-detection"
    assert config.seed == 42
    assert config.data.target_col == "isFraud"
    assert config.data.selection_threshold == 0.05
    assert config.eda.sample_size == 50000
    assert config.training.max_samples == 50000
    assert config.training.decision_threshold == 0.05

