"""Memory Optimization module to shrink numerical types and categories."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger("memory_optimization")


def optimize_memory(
    df: pd.DataFrame, report_path: Path | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Minimizes DataFrame memory usage by downcasting numeric types.

    Args:
        df: Input DataFrame.
        report_path: Optional destination JSON path to write memory changes.

    Returns:
        Tuple of (optimized DataFrame, memory metrics report).
    """
    orig_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info("Initializing memory optimization (Original: %.2f MB)", orig_memory)

    col_reports = {}

    for col in df.columns:
        col_type = df[col].dtype
        orig_col_mem = df[col].memory_usage(deep=True) / (1024 * 1024)

        # Skip already optimized/category types
        if isinstance(col_type, pd.CategoricalDtype):
            continue

        if pd.api.types.is_integer_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)

        elif pd.api.types.is_float_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            # Downcasting to float32 is safe, float16 can fail on precision limits
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

        else:
            # Check string cardinality for category conversion
            # Convert only if cardinality is low (< 50% unique values)
            if df[col].nunique() < (len(df) * 0.5):
                df[col] = df[col].astype("category")

        opt_col_mem = df[col].memory_usage(deep=True) / (1024 * 1024)
        col_reports[col] = {
            "original_type": str(col_type),
            "optimized_type": str(df[col].dtype),
            "original_mem_mb": orig_col_mem,
            "optimized_mem_mb": opt_col_mem,
        }

    opt_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    reduction = ((orig_memory - opt_memory) / orig_memory) * 100
    logger.info(
        "Memory optimization complete: %.2f MB -> %.2f MB (%.2f%% reduction)",
        orig_memory,
        opt_memory,
        reduction,
    )

    report = {
        "original_memory_mb": orig_memory,
        "optimized_memory_mb": opt_memory,
        "reduction_pct": reduction,
        "columns": col_reports,
    }

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with Path(report_path).open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        logger.info("Saved memory optimization report to %s", report_path)

    return df, report
