"""Schema Validation module to extract, store, and validate data properties."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger("schema_validation")


def generate_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Generates structural schemas representing data features.

    Args:
        df: Input DataFrame.

    Returns:
        Structured schema dictionary mapping column properties.
    """
    logger.info("Generating structural schema from DataFrame.")
    schema: dict[str, Any] = {}

    for col in df.columns:
        series = df[col]
        cardinality = int(series.nunique(dropna=True))
        missing_pct = float((series.isna().sum() / len(df)) * 100)

        # Decide column type family
        if pd.api.types.is_numeric_dtype(series.dtype):
            family = "numeric"
            raw_min = series.min()
            raw_max = series.max()
            col_min = float(raw_min) if pd.notna(raw_min) else None
            col_max = float(raw_max) if pd.notna(raw_max) else None
        else:
            family = "categorical"
            col_min, col_max = None, None

        schema[col] = {
            "dtype": str(series.dtype),
            "nullable": bool(series.isna().any()),
            "cardinality": cardinality,
            "missing_pct": missing_pct,
            "family": family,
            "min": col_min,
            "max": col_max,
        }

    return schema


def save_schema(schema: dict[str, Any], output_path: Path) -> None:
    """Saves a schema dictionary to the specified JSON file.

    Args:
        schema: Schema dictionary.
        output_path: Destination JSON filepath.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=4)
    logger.info("Saved schema to %s", output_path)


def load_schema(schema_path: Path) -> dict[str, Any]:
    """Loads a schema dictionary from a JSON file.

    Args:
        schema_path: Path to the JSON schema file.

    Returns:
        Loaded schema dictionary.
    """
    with Path(schema_path).open(encoding="utf-8") as f:
        return json.load(f)


def validate_schema(df: pd.DataFrame, schema: dict[str, Any]) -> list[str]:
    """Compares DataFrame features against a reference schema config.

    Args:
        df: Input DataFrame.
        schema: Reference schema dictionary.

    Returns:
        List of schema mismatch description strings.
    """
    logger.info("Comparing DataFrame against baseline schema rules.")
    errors = []

    # Check for missing required columns
    for expected_col in schema:
        if expected_col not in df.columns:
            errors.append(f"Missing expected column: '{expected_col}'")

    # Check for type mismatch on matching columns
    for col in df.columns:
        if col not in schema:
            # Tolerable, but log a caution
            logger.warning("Unregistered column found in dataset: '%s'", col)
            continue

        ref_info = schema[col]
        col_dtype = str(df[col].dtype)

        # Basic type comparison (e.g. check numeric compatibility)
        actual_is_numeric = pd.api.types.is_numeric_dtype(df[col].dtype)
        expected_is_numeric = ref_info["family"] == "numeric"

        if actual_is_numeric != expected_is_numeric:
            errors.append(
                f"Column '{col}' family mismatch. Expected numeric: "
                f"{expected_is_numeric}, Actual type: {col_dtype}"
            )

    return errors
