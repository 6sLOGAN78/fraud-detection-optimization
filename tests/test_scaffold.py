"""Basic scaffold verification tests."""

from pathlib import Path

from src.utils.logging import setup_logger


def test_logger_setup() -> None:
    """Test logger configuration returns a valid Logger."""
    logger = setup_logger("test_scaffold_logger")
    assert logger is not None
    assert logger.name == "test_scaffold_logger"


def test_folders_exist() -> None:
    """Verify that expected target directories were created."""
    expected_dirs = [
        "configs/data",
        "configs/model",
        "configs/optimization",
        "src/config",
        "src/data",
        "src/preprocessing",
        "src/eda",
        "src/feature_engineering",
        "src/feature_selection",
        "src/models",
        "src/optimization",
        "src/evaluation",
        "src/explainability",
        "src/deployment",
        "src/monitoring",
        "src/visualization",
        "src/pipelines",
        "src/utils",
    ]
    for d in expected_dirs:
        assert Path(d).is_dir(), f"Directory {d} should exist"
