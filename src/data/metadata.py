"""Metadata generation and column classification module for IEEE-CIS datasets."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger("metadata_generation")


def classify_column(col: str, dtype: str) -> str:
    """Classifies a column name into its IEEE-CIS domain feature family.

    Args:
        col: Column name.
        dtype: Data type string.

    Returns:
        Feature family name.
    """
    col_lower = col.lower()

    if col == "isFraud":
        return "Target"
    if col == "TransactionID":
        return "Identifier"
    if col == "TransactionDT":
        return "Time Delta (D)"
    if col == "TransactionAmt":
        return "Numeric"
    if col == "ProductCD":
        return "Categorical"
    if col == "has_identity":
        return "Boolean"

    # Card columns (card1 - card6)
    if col.startswith("card"):
        return "Card"

    # Address columns (addr1, addr2)
    if col.startswith("addr"):
        return "Address"

    # Email domains
    if "emaildomain" in col_lower:
        return "Email"

    # count features (C1 - C14)
    if col.startswith("C") and col[1:].isdigit():
        return "Count (C)"

    # time deltas (D1 - D15)
    if col.startswith("D") and col[1:].isdigit():
        return "Time Delta (D)"

    # match features (M1 - M9)
    if col.startswith("M") and col[1:].isdigit():
        return "Categorical"

    # anonymous V-features (V1 - V339)
    if col.startswith("V") and col[1:].isdigit():
        return "Anonymous (V)"

    # identity features (id_01 - id_38)
    if col.startswith("id_") and col[3:].split("_")[0].isdigit():
        return "Identity"

    # Device features
    if col in ["DeviceInfo", "DeviceType"]:
        return "Device"

    # Fallback classifications
    if "int" in dtype or "float" in dtype:
        return "Numeric"
    return "Categorical"


def generate_metadata_reports(
    df: pd.DataFrame,
    dict_path: Path | None = None,
    groups_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Generates the feature dictionary and classifies columns.

    Args:
        df: Input DataFrame.
        dict_path: Optional JSON path to export feature dictionary.
        groups_path: Optional JSON path to export column groups.

    Returns:
        A tuple of (Feature Dictionary, Column Groups mapping).
    """
    logger.info("Generating feature dictionary and column classifications.")

    feature_dict: dict[str, Any] = {}
    column_groups: dict[str, list[str]] = {}

    n_rows = len(df)

    for col in df.columns:
        dtype_str = str(df[col].dtype)
        family = classify_column(col, dtype_str)

        # Compute basics
        missing_pct = float((df[col].isna().sum() / n_rows) * 100)
        cardinality = int(df[col].nunique(dropna=True))

        source = (
            "Identity File" if family in ["Identity", "Device"] else "Transaction File"
        )
        if col == "has_identity":
            source = "Engineered"

        feature_dict[col] = {
            "family": family,
            "dtype": dtype_str,
            "missing_pct": missing_pct,
            "cardinality": cardinality,
            "source": source,
        }

        column_groups.setdefault(family, []).append(col)

    if dict_path:
        dict_path.parent.mkdir(parents=True, exist_ok=True)
        with Path(dict_path).open("w", encoding="utf-8") as f:
            json.dump(feature_dict, f, indent=4)
        logger.info("Saved feature dictionary to %s", dict_path)

    if groups_path:
        groups_path.parent.mkdir(parents=True, exist_ok=True)
        with Path(groups_path).open("w", encoding="utf-8") as f:
            json.dump(column_groups, f, indent=4)
        logger.info("Saved column groups map to %s", groups_path)

    return feature_dict, column_groups
