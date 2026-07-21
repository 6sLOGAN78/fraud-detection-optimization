"""Pipeline orchestration script for running Data Quality Assessment."""

import logging
from pathlib import Path

import pandas as pd

from src.eda.quality import DataQualityAssessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Executes the data quality assessment orchestration pipeline step."""
    logger.info("Initializing Data Quality Assessment stage...")

    # 1. Load merged datasets
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

    # 2. Run Quality Assessor
    report_dir = Path("reports/eda/quality")
    report_dir.mkdir(parents=True, exist_ok=True)

    assessor = DataQualityAssessor(df_train, df_test)

    logger.info("Auditing missing values...")
    missing_df, missing_summary = assessor.audit_missingness(report_dir)

    logger.info("Auditing duplicate rows, TransactionIDs, columns...")
    dup_df = assessor.audit_duplicates(report_dir)

    logger.info("Detecting constant features...")
    const_df = assessor.detect_constant_features(report_dir)

    logger.info("Detecting near-constant features (dominant >= 99%)...")
    near_const_df = assessor.detect_near_constant_features(report_dir)

    logger.info("Detecting invalid ranges or placeholders...")
    invalid_df = assessor.detect_invalid_values(report_dir)

    logger.info("Detecting infinite floats...")
    infinite_df = assessor.detect_infinite_values(report_dir)

    logger.info("Assessing outliers via IQR and Z-scores...")
    outlier_df = assessor.assess_outliers(report_dir)

    logger.info("Validating schema consistency...")
    consistency_dict = assessor.validate_consistency(report_dir)

    # 3. Compile recommendations
    recs = []

    # Recommendations on missing values
    high_missing = missing_df[
        missing_df["missing_pct_train"] > 50.0
    ]["column"].tolist()
    if high_missing:
        recs.append({
            "category": "High Missingness",
            "target": ", ".join(high_missing[:5]) + (
                "..." if len(high_missing) > 5 else ""
            ),
            "recommendation": (
                "Features have over 50% missing data. "
                "Investigate imputation or drop suitability."
            ),
        })

    # Recommendations on duplicates
    dup_rows_tr = int(
        dup_df[
            dup_df["Metric"] == "Exact Duplicate Rows (Train)"
        ]["Value"].iloc[0]
    )
    if dup_rows_tr > 0:
        recs.append({
            "category": "Duplicates",
            "target": "df_train rows",
            "recommendation": (
                f"Found {dup_rows_tr} exact duplicate rows in train. "
                "Deduplicate before training."
            ),
        })

    # Recommendations on constant features
    if not const_df.empty:
        const_cols = const_df["column"].tolist()
        recs.append({
            "category": "Constant Features",
            "target": ", ".join(const_cols[:5]) + (
                "..." if len(const_cols) > 5 else ""
            ),
            "recommendation": (
                f"Found {len(const_cols)} constant features with "
                "zero variance. Drop before training."
            ),
        })

    # Recommendations on near-constant features
    if not near_const_df.empty:
        near_cols = near_const_df["column"].tolist()
        recs.append({
            "category": "Near-Constant Features",
            "target": ", ".join(near_cols[:5]) + (
                "..." if len(near_cols) > 5 else ""
            ),
            "recommendation": (
                f"Found {len(near_cols)} features with >= 99% dominant val. "
                "Review predictive power."
            ),
        })

    # Recommendations on invalid values
    if not invalid_df.empty:
        inv_cols = invalid_df["column"].tolist()
        recs.append({
            "category": "Invalid Values",
            "target": ", ".join(inv_cols[:5]) + (
                "..." if len(inv_cols) > 5 else ""
            ),
            "recommendation": (
                f"Found {len(invalid_df)} columns with invalid values. "
                "Inspect cleaning logic."
            ),
        })

    # Recommendations on infinite values
    if not infinite_df.empty:
        inf_cols = infinite_df["column"].tolist()
        recs.append({
            "category": "Infinite Values",
            "target": ", ".join(inf_cols[:5]) + (
                "..." if len(inf_cols) > 5 else ""
            ),
            "recommendation": (
                "Numeric values containing infinity. Clamp or impute."
            ),
        })

    # Recommendations on outliers
    if not outlier_df.empty:
        out_cols = outlier_df.sort_values(
            by="outliers_iqr_pct", ascending=False
        )["column"].head(5).tolist()
        recs.append({
            "category": "Outlier Extremes",
            "target": ", ".join(out_cols),
            "recommendation": (
                "High volume of extreme values detected. "
                "Tree-based modeling or robust scaling advised."
            ),
        })

    # Recommendations on schema consistency
    mismatches = consistency_dict["schema_alignment"]["mismatch_count"]
    type_mismatches = consistency_dict["type_checks"]["mismatch_count"]
    if mismatches > 0 or type_mismatches > 0:
        recs.append({
            "category": "Schema Inconsistency",
            "target": "Train vs Test schemas",
            "recommendation": (
                f"Found {mismatches} columns mismatched and "
                f"{type_mismatches} type check mismatches."
            ),
        })

    # Fallback default if no warnings
    if not recs:
        recs.append({
            "category": "Clean Quality Check",
            "target": "All features",
            "recommendation": "No critical data quality issues flagged.",
        })

    # 4. Score Quality
    n_total_cols = len(assessor.common_cols)
    const_pct = (
        float(len(const_df) / n_total_cols * 100) if n_total_cols > 0 else 0.0
    )
    near_const_pct = (
        float(len(near_const_df) / n_total_cols * 100) if n_total_cols > 0 else 0.0
    )
    outlier_pct = (
        float(outlier_df["outliers_iqr_pct"].mean())
        if not outlier_df.empty
        else 0.0
    )

    scoring_metrics = {
        "missing_pct_train": missing_summary["train_overall_missing_pct"],
        "duplicate_trans_pct_train": float(
            dup_df[
                dup_df["Metric"] == "Duplicate Transaction Records (Train)"
            ]["Percentage"].iloc[0]
        ),
        "invalid_count": len(invalid_df),
        "constant_pct": const_pct,
        "near_constant_pct": near_const_pct,
        "outlier_pct": outlier_pct,
        "schema_mismatches": mismatches,
        "type_mismatches": type_mismatches,
    }

    quality_summary = assessor.compute_quality_score(
        report_dir, scoring_metrics
    )

    # 5. Compile HTML reports
    assessor.compile_html_report(
        report_dir=report_dir,
        summary=quality_summary,
        recs=recs,
        missing_df=missing_df,
        dup_df=dup_df,
        const_df=const_df,
        near_const_df=near_const_df,
        invalid_df=invalid_df,
        outlier_df=outlier_df,
    )

    logger.info(
        "Data Quality Assessment finished successfully. Reports saved to %s",
        report_dir,
    )


if __name__ == "__main__":
    main()
