"""Dataset Merge module to join transaction and identity tables."""

import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger("dataset_merge")


def merge_datasets(
    transaction_df: pd.DataFrame, identity_df: pd.DataFrame
) -> pd.DataFrame:
    """Joins transaction and identity tables using a left join on TransactionID.

    Appends the 'has_identity' boolean flag.

    Args:
        transaction_df: Transaction DataFrame.
        identity_df: Identity DataFrame.

    Returns:
        Merged DataFrame.
    """
    logger.info("Merging transaction and identity datasets.")

    # Validate that TransactionID merges correctly
    if "TransactionID" not in transaction_df.columns:
        raise ValueError("TransactionID column not found in transactions.")
    if "TransactionID" not in identity_df.columns:
        raise ValueError("TransactionID column not found in identity data.")

    # Perform left join
    merged_df = pd.merge(transaction_df, identity_df, on="TransactionID", how="left")

    # Add 'has_identity' boolean feature
    identity_ids = set(identity_df["TransactionID"])
    merged_df["has_identity"] = (
        merged_df["TransactionID"].isin(identity_ids).astype(int)
    )

    logger.info(
        "Successfully merged: %d columns and %d rows. 'has_identity' added.",
        len(merged_df.columns),
        len(merged_df),
    )

    return merged_df
