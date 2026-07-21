"""Feature Store registration and versioned dataset management module."""

import re
from pathlib import Path

import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger("feature_store")


def register_features_to_store(
    df: pd.DataFrame, column_groups: dict[str, list[str]], store_dir: Path
) -> None:
    """Saves column families as independent compressed parquets in the Feature Store.

    Args:
        df: Input DataFrame containing the columns.
        column_groups: Mappings of family names to lists of columns.
        store_dir: Base directory for the Feature Store.
    """
    logger.info("Registering features into Feature Store at %s", store_dir)

    # Ensure TransactionID is preserved in every subset to allow joins
    id_col = "TransactionID"
    if id_col not in df.columns:
        raise ValueError(f"Required identifier '{id_col}' missing from DataFrame.")

    for family, cols in column_groups.items():
        # Keep identifier column in each slice
        cols_to_save = [c for c in cols if c in df.columns]
        if id_col in df.columns and id_col not in cols_to_save:
            cols_to_save.insert(0, id_col)

        if len(cols_to_save) <= 1:
            # Only transaction ID or empty
            continue

        # Map domain family names to folder subdirectories
        folder_name = family.lower().replace(" ", "_").replace("(", "").replace(")", "")
        family_dir = store_dir / folder_name
        family_dir.mkdir(parents=True, exist_ok=True)

        target_file = family_dir / f"{folder_name}.parquet"
        logger.info("Saving family '%s' to %s", family, target_file)

        subset_df = df[cols_to_save]
        subset_df.to_parquet(target_file, compression="snappy", index=False)


def save_processed_dataset(df: pd.DataFrame, processed_dir: Path) -> Path:
    """Saves the final clean dataset with an auto-incrementing version.

    Matches the format: processed_v001.parquet, processed_v002.parquet, etc.

    Args:
        df: Input DataFrame to save.
        processed_dir: Parent output folder.

    Returns:
        The Path to the versioned output file.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Scan directory for versioned parquets to select the next count
    existing_files = list(processed_dir.glob("processed_v*.parquet"))
    max_version = 0

    for file in existing_files:
        match = re.search(r"processed_v(\d+)\.parquet", file.name)
        if match:
            max_version = max(max_version, int(match.group(1)))

    next_version = max_version + 1
    output_filename = f"processed_v{next_version:03d}.parquet"
    target_path = processed_dir / output_filename

    logger.info("Saving versioned processed dataset to %s", target_path)
    df.to_parquet(target_path, compression="snappy", index=False)

    return target_path
