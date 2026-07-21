"""Data Quality checks on missingness, duplicates, constants, and anomalies."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger("data_quality")


def handle_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces infinite (inf, -inf) values with standard NaN.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with converted infinites.
    """
    logger.info("Checking for infinite values.")
    num_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if num_inf > 0:
        logger.info("Found %d infinite values; converting to NaN.", num_inf)
        df = df.replace([np.inf, -np.inf], np.nan)
    return df


def drop_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Identifies and drops columns with a single unique value.

    Args:
        df: Input DataFrame.

    Returns:
        A tuple of (modified DataFrame, list of dropped column names).
    """
    logger.info("Scanning for constant columns.")
    dropped_cols = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            dropped_cols.append(col)

    if dropped_cols:
        logger.info("Dropping constant columns: %s", dropped_cols)
        df = df.drop(columns=dropped_cols)
    return df, dropped_cols


def scan_near_constant_columns(df: pd.DataFrame, threshold: float = 0.999) -> list[str]:
    """Identifies columns where a single value dominates above the threshold percentage.

    Args:
        df: Input DataFrame.
        threshold: config ratio cutoff (e.g., 0.999 means 99.9% same value).

    Returns:
        List of near-constant feature names.
    """
    logger.info("Scanning for near-constant columns (threshold=%s).", threshold)
    near_constants = []
    n_rows = len(df)
    for col in df.columns:
        counts = df[col].value_counts(dropna=True)
        if len(counts) > 0:
            major_freq = counts.iloc[0] / n_rows
            if major_freq >= threshold:
                near_constants.append(col)
    if near_constants:
        logger.info("Found near-constant columns: %s", near_constants)
    return near_constants


def detect_duplicate_columns(df: pd.DataFrame, sample_size: int = 10000) -> list[str]:
    """Scans and detects perfectly identical duplicate columns.

    Uses a fast hashed approach on a sample or entire frame to avoid O(N^2).

    Args:
        df: Input DataFrame.
        sample_size: Number of rows to sample for fast initial check.

    Returns:
        List of duplicate columns to drop.
    """
    logger.info("Scanning for duplicate columns.")
    # Quick preliminary check via sample transposing & hashing
    df_sample = df.head(sample_size) if len(df) > sample_size else df

    # We need to fill na to hash consistently
    df_filled = df_sample.astype(str)
    hashes = df_filled.apply(lambda col: hash(frozenset(col.items())))

    # Cluster identical hashes
    hash_groups: dict[int, list[str]] = {}
    for col, h in hashes.items():
        hash_groups.setdefault(h, []).append(str(col))

    cand_duplicates = [cols for cols in hash_groups.values() if len(cols) > 1]
    confirmed_dups = []

    for group in cand_duplicates:
        lead_col = group[0]
        for follower in group[1:]:
            # Verify full mismatch
            if df[lead_col].equals(df[follower]):
                confirmed_dups.append(follower)

    if confirmed_dups:
        logger.info("Targeted duplicate columns for drop: %s", confirmed_dups)
    return confirmed_dups


def run_quality_checks(
    df: pd.DataFrame,
    near_const_threshold: float = 0.999,
    report_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Runs a complete quality audit, modifying and cleaning features.

    Args:
        df: Input DataFrame.
        near_const_threshold: Near constant cutoff.
        report_path: Output JSON path for reports.

    Returns:
        A tuple of (Modified DataFrame, quality metrics report).
    """
    results: dict[str, Any] = {}

    # 1. Total Rows & duplicates
    n_rows = len(df)
    dup_rows_count = int(df.duplicated().sum())
    results["row_count"] = n_rows
    results["duplicate_rows_count"] = dup_rows_count
    results["duplicate_rows_pct"] = float((dup_rows_count / n_rows) * 100)

    # 2. Infinite Values
    df = handle_infinite_values(df)

    # 3. Constant columns
    df, dropped_constants = drop_constant_columns(df)
    results["dropped_constant_columns"] = dropped_constants

    # 4. Near-constant columns
    near_constants = scan_near_constant_columns(df, threshold=near_const_threshold)
    results["near_constant_columns"] = near_constants

    # 5. Duplicate columns
    dup_cols = detect_duplicate_columns(df)
    if dup_cols:
        df = df.drop(columns=dup_cols)
    results["dropped_duplicate_columns"] = dup_cols

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with Path(report_path).open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info("Saved quality checks report to %s", report_path)

    return df, results
