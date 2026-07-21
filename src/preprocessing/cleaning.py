"""Categorical and numerical columns cleaning and normalization module."""

import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger("data_cleaning")


def normalize_device_os(val: str) -> str:
    """Normalizes string casing and typos in OS or Device features.

    Args:
        val: Raw feature value string.

    Returns:
        Normalized value.
    """
    if not isinstance(val, str):
        return val

    # Clean double spaces and pad whitespaces
    val = " ".join(val.strip().split())
    val_lower = val.lower()

    # Browser Casing normalizations
    if "chrome" in val_lower:
        return "Chrome"
    if "safari" in val_lower:
        return "Safari"
    if "firefox" in val_lower:
        return "Firefox"
    if "opera" in val_lower:
        return "Opera"
    if "edge" in val_lower:
        return "Edge"
    if "ie" in val_lower or "internet explorer" in val_lower:
        return "IE"

    # OS Casing normalizations
    if "windows" in val_lower:
        return "Windows"
    if "ios" in val_lower or "iphone" in val_lower or "ipad" in val_lower:
        return "iOS"
    if "android" in val_lower:
        return "Android"
    if "mac" in val_lower:
        return "MacOS"
    if "linux" in val_lower:
        return "Linux"

    return val


def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trims whitespace, normalizes string casing, and replaces NaNs with 'Missing'.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with cleaned categorical columns.
    """
    logger.info("Starting categorical column cleaning.")
    df = df.copy()

    # Locate categorical or object columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        # Save categorical type info if existed
        is_categorical = isinstance(df[col].dtype, pd.CategoricalDtype)

        # Work on string representations
        series = df[col].astype(str)

        # Replace standard nan strings
        series = series.replace({"nan": "Missing", "None": "Missing", "": "Missing"})

        # Trim spaces & Normalize casing
        series = series.apply(normalize_device_os)

        # Final fillna representation
        series = series.fillna("Missing")

        if is_categorical:
            df[col] = series.astype("category")
        else:
            df[col] = series

    logger.info("Completed cleaning for %d categorical columns.", len(cat_cols))
    return df


def impute_numerical_columns(df: pd.DataFrame, strategy: str = "none") -> pd.DataFrame:
    """Optionally imputes missing values in numerical columns.

    Args:
        df: Input DataFrame.
        strategy: Imputation approach ('median', 'mean', or 'none').

    Returns:
        DataFrame with optionally imputed columns.
    """
    if strategy == "none":
        logger.info("Imputation skipped (trees natively support NaNs).")
        return df

    logger.info("Imputing numerical columns using strategy: %s", strategy)
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns

    for col in num_cols:
        if df[col].isna().any():
            if strategy == "median":
                fill_val = df[col].median()
            elif strategy == "mean":
                fill_val = df[col].mean()
            else:
                raise ValueError(f"Unknown imputation strategy: {strategy}")

            df[col] = df[col].fillna(fill_val)

    logger.info("Completed numerical imputation.")
    return df
