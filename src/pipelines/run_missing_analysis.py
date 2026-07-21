"""Pipeline orchestration script for running Missing Value Analysis."""

import json
import logging
from pathlib import Path

import pandas as pd

from src.eda.missing import MissingValueAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Orchestrates missing value percentage, correlations, and HTML reports."""
    logger.info("Initializing Missing Value Analysis stage...")

    # Load merged data outputs
    train_path = Path("data/interim/train_merged.parquet")
    test_path = Path("data/interim/test_merged.parquet")

    if not train_path.exists():
        raise FileNotFoundError(
            f"Required train dataset {train_path} does not exist."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"Required test dataset {test_path} does not exist."
        )

    logger.info("Loading training dataset from %s", train_path)
    df_train = pd.read_parquet(train_path)

    logger.info("Loading testing dataset from %s", test_path)
    df_test = pd.read_parquet(test_path)

    report_dir = Path("reports/eda/missing")
    report_dir.mkdir(parents=True, exist_ok=True)

    analyzer = MissingValueAnalyzer(df_train, df_test)

    logger.info("Calculating missingness percentages...")
    df_pct = analyzer.analyze_missing_percentages(report_dir)

    logger.info("Generating mapping heatmaps...")
    analyzer.generate_missing_heatmaps(report_dir)

    logger.info("Calculating missingness pairwise correlations...")
    analyzer.analyze_missing_correlations(report_dir)

    logger.info("Analyzing missingness against target fraud rates...")
    df_fraud = analyzer.analyze_missing_vs_target(report_dir)

    logger.info("Clustering row-wise missingness patterns...")
    analyzer.analyze_missing_patterns(report_dir)

    logger.info("Grouping metrics by 11 feature families...")
    df_fam = analyzer.analyze_family_missingness(report_dir)

    logger.info("Conducting train vs test missingness divergence checks...")
    df_comp = analyzer.compare_train_test_missingness(report_dir)

    logger.info("Formulating missingness handling strategy recommendations...")
    df_recs = analyzer.generate_recommendations(report_dir, df_pct)

    logger.info("Compiling responsive HTML report dashboard...")
    with (report_dir / "missing_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)

    analyzer.compile_html_report(
        report_dir=report_dir,
        summary=summary,
        df_pct=df_pct,
        df_fraud=df_fraud,
        df_fam=df_fam,
        df_comp=df_comp,
        df_recs=df_recs,
    )

    logger.info("Missing Value Analysis completed successfully.")


if __name__ == "__main__":
    main()
