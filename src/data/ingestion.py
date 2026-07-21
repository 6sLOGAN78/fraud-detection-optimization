"""Data Ingestion module for loading and validating raw IEEE-CIS CSV files."""

from pathlib import Path

import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger("data_ingestion")


class IngestionError(Exception):
    """Raised when data ingestion or validation checks fail."""

    pass


def validate_file(file_path: Path, expected_cols: int | None = None) -> None:
    """Validates raw CSV file path, accessibility, delimiter and shape aspects.

    Args:
        file_path: Path to the target CSV.
        expected_cols: Optional number of columns expected.

    Raises:
        IngestionError: If file validation checks fail.
    """
    if not file_path.exists():
        raise IngestionError(f"Target file {file_path} does not exist.")

    if not file_path.is_file():
        raise IngestionError(f"Target path {file_path} is not a file.")

    # UTF-8 encoding scan & delimiter check on first line
    try:
        with Path(file_path).open(encoding="utf-8") as f:
            first_line = f.readline()
    except UnicodeDecodeError as e:
        raise IngestionError(f"File {file_path} is not valid UTF-8: {e}") from e

    if "," not in first_line:
        # Check standard commas for csv
        raise IngestionError(
            f"File {file_path} does not appear to use comma as delimiter."
        )

    # Quick column dimensions check using sample header parsing
    try:
        header_df = pd.read_csv(file_path, nrows=5)
    except Exception as e:
        raise IngestionError(f"Failed to read CSV header from {file_path}: {e}") from e

    cols_count = len(header_df.columns)
    if expected_cols and cols_count != expected_cols:
        raise IngestionError(
            f"File {file_path} has {cols_count} columns; expected {expected_cols}."
        )


def load_dataset(file_path: Path, expected_cols: int | None = None) -> pd.DataFrame:
    """Loads a raw CSV from the specified path using pyarrow backend.

    Args:
        file_path: Path to the target CSV.
        expected_cols: Expected columns count.

    Returns:
        Loaded pandas DataFrame.
    """
    logger.info("Starting validation for: %s", file_path)
    validate_file(file_path, expected_cols)

    logger.info("Loading dataset: %s", file_path)
    try:
        df = pd.read_csv(file_path, engine="pyarrow")
        logger.info(
            "Successfully loaded %s columns and %s rows.",
            len(df.columns),
            len(df),
        )
        return df
    except Exception as e:
        raise IngestionError(f"Failed to load dataset {file_path}: {e}") from e
