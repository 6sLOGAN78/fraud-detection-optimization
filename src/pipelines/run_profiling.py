"""Pipeline execution script for Dataset Profiling and visualization outputs."""

import re
from pathlib import Path

import pandas as pd

from src.eda.profiling import (
    DatasetProfiler,
    generate_feature_dictionary_reports,
    write_html_report,
    write_json_report,
    write_markdown_summary,
)
from src.utils.logging import setup_logger

logger = setup_logger("run_profiling")


def find_latest_processed_train(processed_dir: Path) -> Path:
    """Finds the latest versioned train file under data/processed/."""
    files = list(processed_dir.glob("processed_v*.parquet"))
    if not files:
        raise FileNotFoundError(f"No processed train files found in {processed_dir}")

    max_version = -1
    latest_file = None
    for f in files:
        match = re.search(r"processed_v(\d+)\.parquet", f.name)
        if match:
            v = int(match.group(1))
            if v > max_version:
                max_version = v
                latest_file = f

    if latest_file is None:
        raise FileNotFoundError(
            "Could not find matching processed train dataset versions."
        )
    return latest_file


def main() -> None:
    """Orchestrates dataset profiling diagnostics and compiles reports."""
    logger.info("Starting Dataset Profiling pipeline execution.")

    processed_dir = Path("data/processed")
    reports_dir = Path("reports/eda")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load latest processed train dataset
    train_path = find_latest_processed_train(processed_dir)
    logger.info("Loading processed train dataset from %s", train_path)
    df_train = pd.read_parquet(train_path)

    # 2. Load merged test dataset
    test_path = Path("data/interim/test_merged.parquet")
    if not test_path.exists():
        logger.warning("%s not found, falling back to opt test transaction.", test_path)
        test_path = Path("data/interim/test_transaction_opt.parquet")

    logger.info("Loading test dataset from %s", test_path)
    df_test = pd.read_parquet(test_path)

    # 3. Create profiler
    logger.info("Analyzing datasets properties...")
    profiler = DatasetProfiler(df_train, df_test)

    # Execute checks
    inventory = profiler.profile_inventory()
    _col_mem, mem_summary = profiler.profile_memory(reports_dir)
    df_card = profiler.analyze_cardinality(reports_dir)
    _df_num, _df_cat = profiler.profile_statistics(reports_dir)
    df_comp, comp_summary = profiler.profile_completeness(reports_dir)

    # Recommendations
    logger.info("Compiling analytic recommendations...")
    recs = profiler.generate_recommendations(df_card, df_comp)

    # Feature Dictionary
    logger.info("Writing Feature Dictionary indices...")
    feat_dict = generate_feature_dictionary_reports(df_card, df_comp, recs, reports_dir)

    # Reports
    logger.info("Compiling Markdown summary, JSON stats, and HTML dashboard...")
    write_markdown_summary(
        inventory,
        mem_summary,
        comp_summary,
        recs,
        reports_dir / "dataset_summary.md",
    )
    write_html_report(
        inventory,
        mem_summary,
        comp_summary,
        recs,
        feat_dict,
        reports_dir / "dataset_profile.html",
    )
    write_json_report(
        inventory,
        mem_summary,
        comp_summary,
        recs,
        reports_dir / "dataset_profile.json",
    )

    logger.info(
        "Dataset Profiling completed successfully. Output reports in %s",
        reports_dir,
    )


if __name__ == "__main__":
    main()
