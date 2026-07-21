"""Data drift monitoring module.

Computes Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.utils.logging import setup_logger

logger = setup_logger("drift_analysis")


def calculate_psi(
    train_series: pd.Series, test_series: pd.Series, num_buckets: int = 10
) -> float:
    """Computes Population Stability Index (PSI) between train and test distributions.

    Args:
        train_series: Reference pandas series.
        test_series: Comparison pandas series.
        num_buckets: Number of quantiles for binning continuous columns.

    Returns:
        Computed PSI score.
    """
    # Drop missing values for calculation
    train_clean = train_series.dropna()
    test_clean = test_series.dropna()

    if len(train_clean) == 0 or len(test_clean) == 0:
        return 0.0

    # Categorical vs Numerical binning
    is_numeric = pd.api.types.is_numeric_dtype(train_clean.dtype)

    if is_numeric and train_clean.nunique() > num_buckets:
        # Create quantile bin boundaries based on training data
        try:
            percentiles = np.linspace(0, 100, num_buckets + 1)
            bins = np.percentile(train_clean, percentiles)
            # Clip ends to cover test outliers
            bins[0] = -np.inf
            bins[-1] = np.inf
            # Ensure unique bins
            bins = np.unique(bins)
            if len(bins) < 2:
                bins = np.array([-np.inf, np.inf])

            train_counts = pd.cut(train_clean, bins=bins).value_counts()
            test_counts = pd.cut(test_clean, bins=bins).value_counts()
        except Exception:
            # Fallback to category logic if binning fails
            train_counts = train_clean.value_counts()
            test_counts = test_clean.value_counts()
    else:
        # Categorical labels as bins
        train_counts = train_clean.value_counts()
        test_counts = test_clean.value_counts()

    # Reindex test counts to match train bins
    all_categories = train_counts.index.union(test_counts.index)
    train_counts = train_counts.reindex(all_categories, fill_value=0)
    test_counts = test_counts.reindex(all_categories, fill_value=0)

    # Convert to percentages
    train_pcts = train_counts / len(train_clean)
    test_pcts = test_counts / len(test_clean)

    # Adjust 0s to avoid divide by zero or log(0)
    train_pcts = train_pcts.replace(0, 1e-4)
    test_pcts = test_pcts.replace(0, 1e-4)

    # Compute PSI
    psi_val = 0.0
    for t_pct, p_pct in zip(test_pcts, train_pcts, strict=False):
        psi_val += (t_pct - p_pct) * math.log(t_pct / p_pct)

    return float(psi_val)


def calculate_ks_test(
    train_series: pd.Series, test_series: pd.Series
) -> tuple[float, float]:
    """Computes the Kolmogorov-Smirnov statistic and p-value for numerical columns.

    Args:
        train_series: Reference pandas series.
        test_series: Comparison pandas series.

    Returns:
        Tuple of (KS statistic, p-value).
    """
    train_clean = train_series.dropna()
    test_clean = test_series.dropna()

    if len(train_clean) == 0 or len(test_clean) == 0:
        return 0.0, 1.0

    if not pd.api.types.is_numeric_dtype(train_clean.dtype):
        return 0.0, 1.0

    # Perform KS test
    res = ks_2samp(train_clean, test_clean)
    return float(res.statistic), float(res.pvalue)


def generate_drift_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Runs drift analysis on matching columns and saves CSV and HTML reports.

    Args:
        train_df: Reference training DataFrame.
        test_df: Out-of-time test DataFrame.
        output_dir: Folder to dump statistics and reports.

    Returns:
        Drift report overview dictionary.
    """
    logger.info("Starting population drift analysis.")
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_cols = [
        c for c in train_df.columns if c in test_df.columns and c != "isFraud"
    ]

    drift_records = []
    summary = {"severe_drift_count": 0, "moderate_drift_count": 0, "stable_count": 0}

    for col in shared_cols:
        psi = calculate_psi(train_df[col], test_df[col])

        # KS only works on numerics
        if pd.api.types.is_numeric_dtype(train_df[col].dtype):
            ks_stat, ks_pval = calculate_ks_test(train_df[col], test_df[col])
        else:
            ks_stat, ks_pval = 0.0, 1.0

        # Classify drift severity
        if psi > 0.25:
            severity = "Severe Drift"
            summary["severe_drift_count"] += 1
        elif psi >= 0.1:
            severity = "Moderate Drift"
            summary["moderate_drift_count"] += 1
        else:
            severity = "Stable"
            summary["stable_count"] += 1

        drift_records.append(
            {
                "feature": col,
                "psi_score": psi,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pval,
                "status": severity,
            }
        )

    drift_df = pd.DataFrame(drift_records).sort_values(by="psi_score", ascending=False)

    # Save CSV sheets
    drift_df[["feature", "psi_score", "status"]].to_csv(
        output_dir / "psi_scores.csv", index=False
    )
    drift_df[["feature", "ks_statistic", "ks_pvalue"]].to_csv(
        output_dir / "ks_scores.csv", index=False
    )

    # Generate high quality static HTML table
    html_rows = []
    for _, row in drift_df.iterrows():
        status_color = "green"
        if row["status"] == "Severe Drift":
            status_color = "red"
        elif row["status"] == "Moderate Drift":
            status_color = "orange"

        html_rows.append(
            f"<tr>"
            f"<td>{row['feature']}</td>"
            f"<td>{row['psi_score']:.4f}</td>"
            f"<td>{row['ks_statistic']:.4f}</td>"
            f"<td>{row['ks_pvalue']:.4e}</td>"
            f"<td style='color: {status_color}; font-weight: bold;'>"
            f"{row['status']}</td>"
            f"</tr>"
        )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Train / Test Drift Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f9f9f9;
            }}
            h1 {{ color: #333; }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
                background-color: #fff;
            }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; color: #333; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .summary-box {{
                padding: 15px;
                margin-bottom: 20px;
                background-color: #eef2f7;
                border-left: 5px solid #0056b3;
            }}
        </style>
    </head>
    <body>
        <h1>Train / Test Data Drift Report</h1>
        <div class="summary-box">
            <h3>Summary Metrics:</h3>
            <p>Stable Features: {summary['stable_count']}</p>
            <p>Moderate Drift Features: {summary['moderate_drift_count']}</p>
            <p>Severe Drift Features: {summary['severe_drift_count']}</p>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>PSI Score</th>
                    <th>KS Statistic</th>
                    <th>KS p-Value</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {"".join(html_rows)}
            </tbody>
        </table>
    </body>
    </html>
    """

    with (output_dir / "drift_report.html").open("w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Saved complete drift report output.")
    return {"summary": summary, "results": drift_records}
