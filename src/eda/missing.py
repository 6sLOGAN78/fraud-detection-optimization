"""Missing Value Analysis engine for IEEE-CIS Fraud Detection dataset contents."""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def classify_missing_family(col: str) -> str:
    """Classifies a column into one of the 11 feature families."""
    col_lower = col.lower()
    if col.startswith("card"):
        return "Card"
    if col.startswith("addr"):
        return "Address"
    if col.startswith("dist"):
        return "Distance"
    if "emaildomain" in col_lower:
        return "Email"
    if col in ["DeviceInfo", "DeviceType"] or col.startswith("device"):
        return "Device"
    if col.startswith("id_"):
        return "Identity"
    if col.startswith("C") and col[1:].isdigit():
        return "C Features"
    if col.startswith("D") and col[1:].isdigit():
        return "D Features"
    if col.startswith("M") and col[1:].isdigit():
        return "M Features"
    if col.startswith("V") and col[1:].isdigit():
        return "V Features"
    return "Transaction"


class MissingValueAnalyzer:
    """Performs deep missingness auditing, correlation, patterns, and HTML."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_col: str = "isFraud",
    ):
        self.df_train = df_train
        self.df_test = df_test
        self.target_col = target_col

        # Identify common and analysis columns (excluding identifiers)
        excl_cols = ["TransactionID", "TransactionDT", self.target_col]
        self.all_cols = [
            c for c in self.df_train.columns if c not in excl_cols
        ]
        self.common_cols = [
            c for c in self.all_cols if c in self.df_test.columns
        ]

    def analyze_missing_percentages(self, report_dir: Path) -> pd.DataFrame:
        """Calculates missingness counts and saves percentage charts."""
        records = []
        n_train = len(self.df_train)
        n_test = len(self.df_test)

        for col in self.all_cols:
            miss_tr = int(self.df_train[col].isnull().sum())
            avail_tr = n_train - miss_tr
            pct_tr = float(miss_tr / n_train * 100) if n_train > 0 else 0.0

            # Test set
            if col in self.df_test.columns:
                miss_te = int(self.df_test[col].isnull().sum())
                avail_te = n_test - miss_te
                pct_te = float(miss_te / n_test * 100) if n_test > 0 else 0.0
            else:
                miss_te = 0
                avail_te = 0
                pct_te = 0.0

            # Classification
            if pct_tr == 0.0:
                cat = "Complete"
            elif pct_tr <= 10.0:
                cat = "Low Missing"
            elif pct_tr <= 30.0:
                cat = "Moderate Missing"
            elif pct_tr <= 60.0:
                cat = "High Missing"
            else:
                cat = "Very High Missing"

            records.append({
                "column": col,
                "family": classify_missing_family(col),
                "missing_count_train": miss_tr,
                "missing_pct_train": pct_tr,
                "available_count_train": avail_tr,
                "available_pct_train": (
                    float(avail_tr / n_train * 100) if n_train > 0 else 0.0
                ),
                "missing_count_test": miss_te,
                "missing_pct_test": pct_te,
                "available_count_test": avail_te,
                "available_pct_test": (
                    float(avail_te / n_test * 100) if n_test > 0 else 0.0
                ),
                "category_train": cat,
            })

        df_pct = pd.DataFrame(records)
        df_pct = df_pct.sort_values(by="missing_pct_train", ascending=False)
        df_pct.to_csv(report_dir / "missing_percentage.csv", index=False)

        # Write summary JSON
        summary = {
            "total_features": len(self.all_cols),
            "complete_count": int(
                (df_pct["category_train"] == "Complete").sum()
            ),
            "low_missing_count": int(
                (df_pct["category_train"] == "Low Missing").sum()
            ),
            "moderate_missing_count": int(
                (df_pct["category_train"] == "Moderate Missing").sum()
            ),
            "high_missing_count": int(
                (df_pct["category_train"] == "High Missing").sum()
            ),
            "very_high_missing_count": int(
                (df_pct["category_train"] == "Very High Missing").sum()
            ),
            "train_overall_missing_cells": int(df_pct["missing_count_train"].sum()),
            "train_total_cells": int(len(self.all_cols) * n_train),
            "train_overall_missing_pct": float(
                df_pct["missing_count_train"].sum()
                / (len(self.all_cols) * n_train)
                * 100
            ) if n_train > 0 and len(self.all_cols) > 0 else 0.0,
        }
        with (report_dir / "missing_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        # Plot 1: Top 50 Missing Features
        plt.figure(figsize=(14, 6))
        top_50 = df_pct.head(50)
        sns.barplot(
            x="missing_pct_train",
            y="column",
            data=top_50,
            hue="column",
            palette="mako",
            legend=False,
        )
        plt.title("Top 50 Features by Missingness Percentage (Train)")
        plt.xlabel("Missing Percentage (%)")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(report_dir / "missing_percentage_bar.png", dpi=100)
        plt.close()

        # Plot 2: Missingness Distribution Histogram
        plt.figure(figsize=(10, 5))
        sns.histplot(
            df_pct["missing_pct_train"],
            bins=20,
            kde=True,
            color="#2c3e50",
        )
        plt.title("Distribution of Missingness Percentages Across All Features")
        plt.xlabel("Missing Percentage (%)")
        plt.ylabel("Number of Features")
        plt.tight_layout()
        plt.savefig(report_dir / "missing_distribution_hist.png", dpi=100)
        plt.close()

        return df_pct

    def generate_missing_heatmaps(self, report_dir: Path) -> None:
        """Generates sparsity heatmaps for train/test and family blocks."""
        # Downsample to speed up file generation and keep memory low
        sample_size = min(1000, len(self.df_train))
        df_sample_tr = self.df_train[self.common_cols].sample(
            n=sample_size, random_state=42
        )

        # 1. Dataset Missing Heatmap (Train)
        plt.figure(figsize=(15, 8))
        sns.heatmap(df_sample_tr.isnull(), cbar=False, cmap="binary")
        plt.title("Train Dataset Missingness Map (1000 observations snapshot)")
        plt.xlabel("Features")
        plt.ylabel("Observations")
        plt.tight_layout()
        plt.savefig(report_dir / "missing_heatmap.png", dpi=100)
        plt.close()

        # Segment by features family: V-features tend to dominate.
        family_null_counts = {}
        for col in self.common_cols:
            fam = classify_missing_family(col)
            null_ser = self.df_train[col].isnull().astype(int)
            if fam not in family_null_counts:
                family_null_counts[fam] = null_ser
            else:
                family_null_counts[fam] += null_ser

        df_fam_nulls = pd.DataFrame(family_null_counts)
        plt.figure(figsize=(12, 6))
        sns.heatmap(df_fam_nulls.sample(n=sample_size, random_state=42), cmap="Purples")
        plt.title("Missing Value Counts by Pattern Family (Train)")
        plt.tight_layout()
        plt.savefig(report_dir / "missing_family_heatmap.png", dpi=100)
        plt.close()

    def analyze_missing_correlations(
        self, report_dir: Path
    ) -> pd.DataFrame:
        """Computes Phi/Pearson missingness correlation matrices."""
        # Filter features that have some missingness
        n_train = len(self.df_train)
        missing_cols = []
        for col in self.common_cols:
            n_miss = self.df_train[col].isnull().sum()
            if 0 < n_miss < n_train:
                missing_cols.append(col)

        if not missing_cols:
            # Empty fallback dataframe
            df_corr = pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
            df_corr.to_csv(report_dir / "missing_correlation.csv", index=False)
            return df_corr

        # Build binary indicators
        df_indicators = self.df_train[missing_cols].isnull().astype(float)
        corr_matrix = df_indicators.corr(method="pearson")

        # Extract pairwise correlations
        records = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                val = float(corr_matrix.loc[c1, c2])
                if not np.isnan(val):
                    records.append({
                        "feature_1": c1,
                        "feature_2": c2,
                        "correlation": val,
                    })

        df_corr = pd.DataFrame(records)
        df_corr = df_corr.sort_values(by="correlation", ascending=False)
        df_corr.to_csv(report_dir / "missing_correlation.csv", index=False)

        # Plot top correlation heatmap
        top_correlated_cols = set()
        for _, row in df_corr.head(30).iterrows():
            top_correlated_cols.add(row["feature_1"])
            top_correlated_cols.add(row["feature_2"])

        if list(top_correlated_cols):
            cols_list = list(top_correlated_cols)
            sub_corr = corr_matrix.loc[cols_list, cols_list]
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                sub_corr, cmap="coolwarm", center=0, annot=False, square=True
            )
            plt.title("Missingness Co-occurrence Correlation Matrix (Top Correlated)")
            plt.tight_layout()
            plt.savefig(report_dir / "missing_correlation_heatmap.png", dpi=100)
            plt.close()
        else:
            plt.figure()
            plt.text(0.5, 0.5, "No correlated missing fields found")
            plt.savefig(report_dir / "missing_correlation_heatmap.png")
            plt.close()

        return df_corr

    def analyze_missing_vs_target(self, report_dir: Path) -> pd.DataFrame:
        """Evaluates missingness vs target fraud rates."""
        records = []
        if self.target_col not in self.df_train.columns:
            logger.warning(
                "Target %s not found. Skipping missing vs fraud.",
                self.target_col
            )
            # Create empty placeholder
            df_fraud = pd.DataFrame(
                columns=[
                    "column",
                    "missing_fraud_pct",
                    "available_fraud_pct",
                    "difference",
                    "relative_risk",
                ]
            )
            df_fraud.to_csv(report_dir / "missing_vs_fraud.csv", index=False)
            return df_fraud

        # Extract target counts
        y_train = self.df_train[self.target_col]
        for col in self.all_cols:
            is_null = self.df_train[col].isnull()

            # Missing Group
            miss_cnt = int(is_null.sum())
            miss_fraud = int(y_train[is_null].sum())
            miss_fraud_pct = float(miss_fraud / miss_cnt * 100) if miss_cnt > 0 else 0.0

            # Present Group
            avail_cnt = len(self.df_train) - miss_cnt
            avail_fraud = int(y_train[~is_null].sum())
            avail_fraud_pct = (
                float(avail_fraud / avail_cnt * 100) if avail_cnt > 0 else 0.0
            )

            diff = miss_fraud_pct - avail_fraud_pct

            # Relative Risk calculations
            if avail_fraud_pct > 0.0:
                rr = miss_fraud_pct / avail_fraud_pct
            else:
                rr = 1.0 if miss_fraud_pct == 0.0 else np.nan

            records.append({
                "column": col,
                "family": classify_missing_family(col),
                "missing_count": miss_cnt,
                "missing_fraud_count": miss_fraud,
                "missing_fraud_pct": miss_fraud_pct,
                "available_count": avail_cnt,
                "available_fraud_count": avail_fraud,
                "available_fraud_pct": avail_fraud_pct,
                "difference": diff,
                "relative_risk": float(rr) if not np.isnan(rr) else None,
            })

        df_fraud = pd.DataFrame(records)
        df_fraud = df_fraud.sort_values(by="difference", key=abs, ascending=False)
        df_fraud.to_csv(report_dir / "missing_vs_fraud.csv", index=False)

        # Plot 3: Missing vs Fraud Bar Chart
        plt.figure(figsize=(14, 6))
        # Top 15 positive risk difference metrics plus top 15 negative
        top_diff = df_fraud.head(20)
        sns.barplot(
            x="difference",
            y="column",
            data=top_diff,
            hue="column",
            palette="viridis",
            legend=False,
        )
        plt.title("Difference in Fraud Rate: Missing vs Present (Top 20 Features)")
        plt.xlabel("Fraud Rate Difference (Missing % - Available %)")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(report_dir / "missing_vs_fraud_bar.png", dpi=100)
        plt.close()

        return df_fraud

    def analyze_missing_patterns(
        self, report_dir: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Performs row level missingness counts and pattern combination analysis."""
        n_train = len(self.df_train)

        # Row level counts
        null_counts_row = self.df_train[self.common_cols].isnull().sum(axis=1)
        null_ratio_row = null_counts_row / len(self.common_cols)

        df_row = pd.DataFrame({
            "TransactionID": self.df_train["TransactionID"],
            "missing_column_count": null_counts_row,
            "missing_column_ratio": null_ratio_row,
        })
        df_row.to_csv(report_dir / "row_missing_statistics.csv", index=False)

        # Plot 4: Row missingness distribution
        plt.figure(figsize=(10, 5))
        sns.histplot(null_ratio_row, bins=25, kde=True, color="#d35400")
        plt.title("Distribution of Missing Column Ratios per Row/Transaction")
        plt.xlabel("Missing Column Ratio (%)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(report_dir / "row_missing_distribution.png", dpi=100)
        plt.close()

        # Find blocks of missingness across rows (co-missing groups)
        # Select columns representing pattern features
        pattern_cols = [
            c for c in self.common_cols
            if self.df_train[c].isnull().sum() > 0
        ]
        if not pattern_cols:
            df_pat_summary = pd.DataFrame(
                columns=["pattern_signature", "occurrence_count", "percentage"]
            )
            df_pat_summary.to_csv(
                report_dir / "missing_patterns.csv", index=False
            )
            return df_row, df_pat_summary

        # Build mask string patterns or groupers.
        # To avoid high memory, select top families.
        fam_indicators = {}
        for col in pattern_cols:
            fam = classify_missing_family(col)
            if fam not in fam_indicators:
                fam_indicators[fam] = self.df_train[col].isnull()
            else:
                fam_indicators[fam] = fam_indicators[fam] | self.df_train[col].isnull()

        df_fam_ind = pd.DataFrame(fam_indicators)
        pattern_sigs = df_fam_ind.apply(
            lambda r: "|".join([fam for fam, val in r.items() if val]),
            axis=1
        )
        # Compile counts
        pattern_counts = pattern_sigs.value_counts()
        pat_records = []
        for pat, count in pattern_counts.items():
            pat_records.append({
                "pattern_signature": pat if pat else "No Missing Families",
                "occurrence_count": int(count),
                "percentage": float(count / n_train * 100),
            })

        df_pat_summary = pd.DataFrame(pat_records)
        df_pat_summary.to_csv(report_dir / "missing_patterns.csv", index=False)

        return df_row, df_pat_summary

    def analyze_family_missingness(self, report_dir: Path) -> pd.DataFrame:
        """Aggregates metrics for the 11 feature families."""
        families = [
            "Transaction", "Card", "Address", "Distance", "Email",
            "Identity", "Device", "C Features", "D Features", "M Features", "V Features"
        ]

        n_train = len(self.df_train)
        records = []

        for fam in families:
            cols = [c for c in self.all_cols if classify_missing_family(c) == fam]
            if not cols:
                continue

            col_pcts = []
            cells_missing = 0.0

            for col in cols:
                m_cnt = float(self.df_train[col].isnull().sum())
                col_pcts.append(m_cnt / n_train * 100)
                cells_missing += m_cnt

            records.append({
                "family": fam,
                "column_count": len(cols),
                "avg_missing_pct": float(np.mean(col_pcts)),
                "max_missing_pct": float(np.max(col_pcts)),
                "min_missing_pct": float(np.min(col_pcts)),
                "total_missing_cells": int(cells_missing),
            })

        df_fam = pd.DataFrame(records)
        df_fam.to_csv(report_dir / "feature_family_missing.csv", index=False)

        # Plot 5: Family comparison Chart
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x="avg_missing_pct",
            y="family",
            data=df_fam.sort_values(by="avg_missing_pct", ascending=False),
            hue="family",
            palette="rocket",
            legend=False,
        )
        plt.title("Average Missingness Percentage by Feature Family (Train)")
        plt.xlabel("Average Missing Percentage (%)")
        plt.ylabel("Feature Family")
        plt.tight_layout()
        plt.savefig(report_dir / "family_missing_chart.png", dpi=100)
        plt.close()

        return df_fam

    def compare_train_test_missingness(
        self, report_dir: Path
    ) -> pd.DataFrame:
        """Flags drift (> 5% difference) of missing values between partitions."""
        records = []
        n_train = len(self.df_train)
        n_test = len(self.df_test)

        for col in self.common_cols:
            tr_miss = self.df_train[col].isnull().sum()
            te_miss = self.df_test[col].isnull().sum()
            tr_pct = float(tr_miss / n_train * 100) if n_train > 0 else 0.0
            te_pct = float(te_miss / n_test * 100) if n_test > 0 else 0.0
            diff = abs(tr_pct - te_pct)

            records.append({
                "column": col,
                "family": classify_missing_family(col),
                "train_missing_pct": tr_pct,
                "test_missing_pct": te_pct,
                "absolute_difference": diff,
                "drift_detected": bool(diff > 5.0),
            })

        df_comp = pd.DataFrame(records)
        df_comp = df_comp.sort_values(by="absolute_difference", ascending=False)
        df_comp.to_csv(report_dir / "train_test_missing_comparison.csv", index=False)

        return df_comp

    def generate_recommendations(
        self, report_dir: Path, df_pct: pd.DataFrame
    ) -> pd.DataFrame:
        """Formulates rule-based handling recommendations based on missing levels."""
        records = []

        for _, row in df_pct.iterrows():
            col = row["column"]
            pct = row["missing_pct_train"]

            if pct == 0.0:
                rec_lbl = "Keep"
                strategy = "Preserve (No Missing Data)"
            elif pct < 5.0:
                rec_lbl = "Impute"
                strategy = "Preserve or Simple Imputation (Mean/Mode/Median)"
            elif pct <= 30.0:
                rec_lbl = "Evaluate Importance"
                strategy = "Evaluate Importance / Tree model handling"
            elif pct <= 60.0:
                rec_lbl = "Create Missing Indicator"
                strategy = "Create Missing Indicator feature + Imputation"
            else:
                rec_lbl = "Drop or Specialized"
                strategy = "Consider Removal or specialized category indicator encoding"

            records.append({
                "column": col,
                "family": row["family"],
                "missing_pct": pct,
                "recommendation": rec_lbl,
                "handling_strategy": strategy,
            })

        df_recs = pd.DataFrame(records)
        df_recs.to_csv(report_dir / "missing_recommendations.csv", index=False)

        return df_recs

    def compile_html_report(
        self,
        report_dir: Path,
        summary: dict[str, Any],
        df_pct: pd.DataFrame,
        df_fraud: pd.DataFrame,
        df_fam: pd.DataFrame,
        df_comp: pd.DataFrame,
        df_recs: pd.DataFrame,
    ) -> None:
        """Compiles clean glassmorphic HTML report dashboard document."""
        # Top 15 missingness table records
        missing_rows = ""
        for _, row in df_pct.head(15).iterrows():
            category = row['category_train']
            missing_rows += f"""
            <tr>
                <td style="font-weight: 700; color: #ffffff;">{row['column']}</td>
                <td><span class="badge badge-teal">{row['family']}</span></td>
                <td>{row['missing_count_train']:,}</td>
                <td>{row['missing_pct_train']:.2f}%</td>
                <td>{row['missing_count_test']:,}</td>
                <td>{row['missing_pct_test']:.2f}%</td>
                <td><span class="badge badge-warning">{category}</span></td>
            </tr>
            """

        # Top 15 fraud difference features
        fraud_rows = ""
        for _, row in df_fraud.head(15).iterrows():
            rr_val = row['relative_risk']
            rr_str = f"{rr_val:.2f}x" if rr_val is not None else "N/A"
            color_cls = "trend-up" if row['difference'] > 0 else "trend-down"
            fraud_rows += f"""
            <tr>
                <td style="font-weight: 700; color: #ffffff;">{row['column']}</td>
                <td><span class="badge badge-teal">{row['family']}</span></td>
                <td>{row['missing_fraud_pct']:.2f}%</td>
                <td>{row['available_fraud_pct']:.2f}%</td>
                <td><span class="{color_cls}">{row['difference']:.2f}%</span></td>
                <td>{rr_str}</td>
            </tr>
            """

        # Family performance table records
        fam_rows = ""
        for _, row in df_fam.iterrows():
            fam_rows += f"""
            <tr>
                <td style="font-weight: 700; color: #ffffff;">{row['family']}</td>
                <td>{row['column_count']}</td>
                <td>{row['avg_missing_pct']:.2f}%</td>
                <td>{row['max_missing_pct']:.2f}%</td>
                <td>{row['min_missing_pct']:.2f}%</td>
                <td>{row['total_missing_cells']:,}</td>
            </tr>
            """

        # Drift mismatch warnings
        drift_cols = df_comp[df_comp["drift_detected"]]
        drift_warnings = ""
        if not drift_cols.empty:
            for _, row in drift_cols.head(10).iterrows():
                drift_warnings += f"""
                <div class="alert-box alert-danger">
                    <strong>{row['column']}</strong> ({row['family']}):
                    Train missing is {row['train_missing_pct']:.2f}%
                    vs Test {row['test_missing_pct']:.2f}%
                    (Amt mismatch: {row['absolute_difference']:.2f}%)
                </div>
                """
        else:
            drift_warnings = """
            <div class="alert-box alert-success">
                All common features align within the 5.0% missingness drift margin.
            </div>
            """

        # Recommendations list items
        rec_list = ""
        for _, row in df_recs.head(15).iterrows():
            rec_list += f"""
            <div class="rec-card">
                <div style="font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: 0.9rem; color: #fff; letter-spacing: 0.05em; text-transform: uppercase;">
                    {row['column']}
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--text-muted);
                            margin-top: 2px; margin-bottom: 8px;">
                    Family: {row['family']} | Missing: {row['missing_pct']:.2f}%
                </div>
                <div class="badge badge-teal"
                     style="font-size: 0.7rem; font-weight: 700;">
                    {row['recommendation']}
                </div>
                <p style="margin-top: 10px; margin-bottom: 0; font-size: 0.85rem; color: var(--text-muted); line-height: 1.4;">
                    {row['handling_strategy']}
                </p>
            </div>
            """

        # Compile glassmorphic theme stylesheet and HTML DOM
        font_url = (
            "https://fonts.googleapis.com/css2?"
            "family=Orbitron:wght@400;600;800;900&"
            "family=JetBrains+Mono:wght@400;700&"
            "family=Inter:wght@400;600&"
            "display=swap"
        )
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IEEE-CIS Missing Pattern Diagnostics</title>
    <link href="{font_url}" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #06070b;
            --card-glass: rgba(18, 22, 32, 0.45);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent: #ffffff;
            --accent-glow: rgba(255, 255, 255, 0.12);
            --text-color: #ffffff;
            --text-muted: #8e97a4;
            --border-color: rgba(255, 255, 255, 0.08);
            --success: #13b981;
            --warning: #8e97a4;
            --danger: #d63031;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        @keyframes scanline {{
            0% {{ transform: translateY(-100%); }}
            100% {{ transform: translateY(100%); }}
        }}
        @keyframes pulse-grey {{
            0% {{ opacity: 0.4; }}
            50% {{ opacity: 1.0; }}
            100% {{ opacity: 0.4; }}
        }}
        .scanline-overlay {{
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: linear-gradient(
                rgba(255, 255, 255, 0),
                rgba(255, 255, 255, 0.012) 50%,
                rgba(255, 255, 255, 0) 100%
            );
            background-size: 100% 4px;
            animation: scanline 12s linear infinite;
            pointer-events: none;
            z-index: 9999;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            position: relative;
            overflow-x: hidden;
        }}

        .hud-grid-bg {{
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background-image: linear-gradient(rgba(255,255,255,0.01) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(255,255,255,0.01) 1px, transparent 1px);
            background-size: 40px 40px;
            pointer-events: none;
            z-index: 0;
        }}

        header {{
            background: rgba(6, 7, 11, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--card-border);
            padding: 20px 40px;
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        h1 {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.6rem;
            font-weight: 900;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #ffffff;
            text-shadow: 0 0 15px rgba(255, 255, 255, 0.15);
        }}

        .nav {{ display: flex; gap: 20px; }}
        .nav a {{
            color: var(--text-muted);
            text-decoration: none;
            font-size: 0.8rem;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            transition: color 0.2s;
        }}
        .nav a:hover {{ color: #ffffff; }}

        .hud-status {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.03);
            padding: 6px 12px;
            border: 1px solid var(--card-border);
            border-radius: 4px;
        }}
        .pulse-dot {{
            width: 8px;
            height: 8px;
            background-color: var(--text-muted);
            border-radius: 50%;
            animation: pulse-grey 2s infinite;
        }}

        main {{
            max-width: 1400px;
            margin: 40px auto;
            padding: 0 20px;
            position: relative;
            z-index: 1;
        }}

        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }}

        .glass-card {{
            background: var(--card-glass);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 4px;
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }}
        .glass-card::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; width: 6px; height: 6px;
            border-top: 1px solid var(--text-muted);
            border-left: 1px solid var(--text-muted);
        }}
        .glass-card:hover {{
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.08);
            transform: translateY(-2px);
        }}

        .metric-title {{
            font-size: 0.75rem;
            font-family: 'Orbitron', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 700;
            color: #fff;
            margin: 0;
        }}
        .metric-desc {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 5px;
        }}

        .badge {{
            padding: 4px 8px;
            border-radius: 0;
            font-size: 0.7rem;
            font-weight: 750;
            display: inline-block;
            text-transform: uppercase;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.05em;
        }}
        .badge-teal {{
            background: rgba(255, 255, 255, 0.05);
            color: #ffffff;
            border: 1px solid var(--border-color);
        }}
        .badge-warning {{
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-muted);
            border: 1px solid var(--border-color);
        }}
        .badge-danger {{
            background: rgba(214, 48, 49, 0.08);
            color: var(--danger);
            border: 1px solid rgba(214, 48, 49, 0.2);
        }}

        .sect-title {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .sect-title::after {{
            content: "[SECTION.MISSING]";
            font-size: 0.70rem;
            color: var(--text-muted);
        }}

        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }}

        @media (max-width: 1024px) {{
            .two-col {{ grid-template-columns: 1fr; }}
        }}

        .styled-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            margin-top: 10px;
        }}
        .styled-table th {{
            background: rgba(255, 255, 255, 0.02);
            text-align: left;
            padding: 10px 12px;
            color: #fff;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            border-bottom: 2px solid var(--border-color);
        }}
        .styled-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-muted);
            font-family: 'Inter', sans-serif;
        }}
        .styled-table tr:hover {{ background: rgba(255, 255, 255, 0.03); }}

        .trend-up {{ color: var(--danger); font-family: 'JetBrains Mono', monospace; font-weight: 700; }}
        .trend-down {{ color: var(--success); font-family: 'JetBrains Mono', monospace; font-weight: 700; }}

        .alert-box {{
            padding: 16px;
            border-radius: 4px;
            margin-bottom: 12px;
            font-size: 0.85rem;
        }}
        .alert-danger {{
            background: rgba(214, 48, 49, 0.08);
            border: 1px solid rgba(214, 48, 49, 0.2);
            color: var(--danger);
        }}
        .alert-success {{
            background: rgba(19, 185, 129, 0.08);
            border: 1px solid rgba(19, 185, 129, 0.2);
            color: var(--success);
        }}

        .rec-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
        }}
        .rec-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 0;
            border-left: 2px solid #ffffff;
            padding: 20px;
            transition: background 0.3s;
        }}
        .rec-card:hover {{ background: rgba(255, 255, 255, 0.04); }}

        .visual-img {{
            max-width: 100%;
            height: auto;
            border-radius: 0;
            border: 1px solid var(--card-border);
            display: block;
            margin: 0 auto;
            filter: grayscale(100%) contrast(1.1) brightness(0.9);
        }}
    </style>
</head>
<body>
    <div class="scanline-overlay"></div>
    <div class="hud-grid-bg"></div>

    <header>
        <h1>IEEE-CIS Missing Pattern Diagnostics</h1>
        <div class="nav">
            <a href="#stats">Statistics</a>
            <a href="#visuals">Visuals</a>
            <a href="#fraud">Fraud Rates</a>
            <a href="#drift">Drift</a>
            <a href="#recs">Recommendations</a>
        </div>
        <div class="hud-status">
            <span class="pulse-dot"></span>
            <span>SYSTEM ACTIVE // DETECTORS ONLINE</span>
        </div>
    </header>

    <main>
        <!-- Info Cards -->
        <div class="dashboard-grid">
            <div class="glass-card">
                <div class="metric-title">Total Swept Features</div>
                <p class="metric-value">{summary['total_features']}</p>
                <div class="metric-desc">
                    Columns audited (excluding targeting keys)
                </div>
            </div>
            <div class="glass-card">
                <div class="metric-title">Complete Columns</div>
                <p class="metric-value">{summary['complete_count']}</p>
                <div class="metric-desc">0% missing data</div>
            </div>
            <div class="glass-card">
                <div class="metric-title">Sparsity Ratio (Train)</div>
                <p class="metric-value">
                    {summary['train_overall_missing_pct']:.2f}%
                </p>
                <div class="metric-desc">Overall missing cell density</div>
            </div>
            <div class="glass-card">
                <div class="metric-title">High/Very High Missing</div>
                <p class="metric-value">
                    {summary['high_missing_count'] +
                     summary['very_high_missing_count']}
                </p>
                <div class="metric-desc">
                    Features exceeding 30% missingness
                </div>
            </div>
        </div>

        <!-- Section 1 & 2: Top Missing & Family Stats -->
        <div class="two-col" id="stats">
            <div class="glass-card">
                <div class="sect-title">Top 15 Missingness Features</div>
                <div style="overflow-x: auto;">
                    <table class="styled-table">
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Family</th>
                                <th>Train Missing</th>
                                <th>Train %</th>
                                <th>Test Missing</th>
                                <th>Test %</th>
                                <th>Category</th>
                            </tr>
                        </thead>
                        <tbody>
                            {missing_rows}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="glass-card">
                <div class="sect-title">Missing Statistics by Feature Family</div>
                <div style="overflow-x: auto;">
                    <table class="styled-table">
                        <thead>
                            <tr>
                                <th>Family</th>
                                <th>Column Count</th>
                                <th>Average Missing %</th>
                                <th>Maximum Missing %</th>
                                <th>Minimum Missing %</th>
                                <th>Total Missing Cells</th>
                            </tr>
                        </thead>
                        <tbody>
                            {fam_rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Section 3: Visualisations -->
        <div class="sect-title" id="visuals">Visual Pattern Analysis</div>
        <div class="two-col" style="margin-top: 10px;">
            <div class="glass-card">
                <div class="metric-title" style="margin-bottom:15px;">
                    Top 50 Missing Features
                </div>
                <img src="missing_percentage_bar.png"
                     alt="Top 50 Missing Features Barplot"
                     class="visual-img">
            </div>
            <div class="glass-card">
                <div class="metric-title" style="margin-bottom:15px;">
                    Sparsity Distribution Map
                </div>
                <img src="missing_heatmap.png"
                     alt="Missing Heatmap"
                     class="visual-img">
            </div>
        </div>

        <div class="two-col" style="margin-top: 20px;">
            <div class="glass-card">
                <div class="metric-title" style="margin-bottom:15px;">
                    Average Family Missingness
                </div>
                <img src="family_missing_chart.png"
                     alt="Family Missingness Pie/Bar"
                     class="visual-img">
            </div>
            <div class="glass-card">
                <div class="metric-title" style="margin-bottom:15px;">
                    Missingness Co-occurrence Heatmap
                </div>
                <img src="missing_correlation_heatmap.png"
                     alt="Missing Correlation Heatmap"
                     class="visual-img">
            </div>
        </div>

        <!-- Section 4: Missing vs Fraud -->
        <div class="two-col" id="fraud" style="margin-top: 40px;">
            <div class="glass-card">
                <div class="sect-title">
                    Missingness Impact on Target (Top 15 Differential)
                </div>
                <div style="overflow-x: auto;">
                    <table class="styled-table">
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Family</th>
                                <th>Fraud Rate (Missing)</th>
                                <th>Fraud Rate (Present)</th>
                                <th>Rate Diff (%)</th>
                                <th>Relative Risk</th>
                            </tr>
                        </thead>
                        <tbody>
                            {fraud_rows}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="glass-card"
                 style="display: flex;
                        flex-direction: column;
                        justify-content: space-between;">
                <div>
                    <div class="sect-title">Missing vs Fraud Divergence Plot</div>
                    <img src="missing_vs_fraud_bar.png"
                         alt="Missing vs Fraud Bar"
                         class="visual-img">
                <div style="margin-top: 20px;">
                    <p style="font-size: 0.85rem;
                              color: var(--text-muted);
                              line-height: 1.6;">
                        <strong>Key Insight:</strong> If the Relative Risk is
                        significantly higher/lower than 1.0 (or difference
                        deviates from 0), the missingness carries important signals.
                        For tree models, simple zero imputation or keeping NaNs
                        is critical.
                    </p>
                </div>
            </div>
        </div>

        <!-- Section 5: Train vs Test Drift -->
        <div class="glass-card" id="drift" style="margin-top: 40px; margin-bottom: 40px;">
            <div class="sect-title">Train vs Test Partition Drift Warnings</div>
            <div class="alert-container">
                {drift_warnings}
            </div>
        </div>

        <!-- Section 6: Recommendations -->
        <div class="sect-title" id="recs">Missing value Strategy Recommendations</div>
        <div class="rec-container">
            {rec_list}
        </div>
    </main>
</body>
</html>
"""
        html_path = report_dir / "missing_report.html"
        with html_path.open("w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info("Saved report to %s", html_path)
