"""Target Variable Analysis module for evaluating isFraud properties."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging import setup_logger

logger = setup_logger("target_analysis")


def classify_imbalance_severity(ratio_value: float) -> str:
    """Classifies class imbalance severity.

    Args:
        ratio_value: Majority to minority class ratio value (e.g. 20.0 for 20:1).

    Returns:
        Severity string ('Low', 'Moderate', 'Severe').
    """
    if ratio_value < 5.0:
        return "Low"
    if ratio_value <= 20.0:
        return "Moderate"
    return "Severe"


class TargetVariableAnalyzer:
    """Performs deep target analysis and class imbalance checks."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_col: str = "isFraud",
        time_col: str = "TransactionDT",
        amt_col: str = "TransactionAmt",
    ) -> None:
        """Initializes analyzer with train/test datasets.

        Args:
            df_train: Train DataFrame containing target.
            df_test: Test DataFrame (may not contain target).
            target_col: Target variable column name.
            time_col: Transaction timestamp column name.
            amt_col: Transaction amount column name.
        """
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.target_col = target_col
        self.time_col = time_col
        self.amt_col = amt_col

        # Identify features
        self.all_cols = list(df_train.columns)
        self.common_cols = [c for c in self.all_cols if c in df_test.columns]

        # Identify identity columns
        self.identity_cols = [
            c for c in self.all_cols
            if c.startswith("id_") and c[3:].split("_")[0].isdigit()
        ]
        # Include DeviceInfo and DeviceType
        for device_col in ["DeviceInfo", "DeviceType"]:
            if device_col in self.all_cols:
                self.identity_cols.append(device_col)

    def analyze_all(self, report_dir: Path) -> None:
        """Runs the entire target variable diagnostics suite.

        Args:
            report_dir: Output base directory.
        """
        logger.info("Initializing Target Variable Analysis diagnostics...")
        report_dir.mkdir(parents=True, exist_ok=True)

        # 1. Target distribution
        df_dist = self.analyze_distribution(report_dir)

        # 2. Class imbalance severity
        df_imb = self.analyze_class_imbalance(report_dir)

        # 3. Overall fraud rates
        df_rate = self.analyze_fraud_rates(report_dir)

        # 4. Temporal patterns
        df_time = self.analyze_fraud_by_time(report_dir)

        # 5. Identity-based risk
        df_ident = self.analyze_fraud_by_identity(report_dir)

        # 6. Transaction amount risk bins
        df_amt = self.analyze_fraud_by_amount(report_dir)

        # 7. Statistical summaries
        df_stats = self.generate_statistical_summary(report_dir)

        # 8. Recommender engine
        recommendations = self.generate_recommendations(report_dir)

        total_tx_row = df_dist[df_dist["class"] == "Total"]
        fraud_row = df_dist[df_dist["class"] == "Fraud"]
        legit_row = df_dist[df_dist["class"] == "Legitimate"]

        # Save targeting metadata JSON
        target_summary = {
            "total_transactions": int(total_tx_row["count"].values[0]),
            "fraud_count": int(fraud_row["count"].values[0]),
            "legit_count": int(legit_row["count"].values[0]),
            "fraud_pct": float(fraud_row["percentage"].values[0]),
            "imbalance_ratio": float(df_imb["imbalance_ratio"].values[0]),
            "imbalance_severity": str(df_imb["severity"].values[0]),
            "recommendations": recommendations,
        }
        with (report_dir / "target_analysis.json").open("w", encoding="utf-8") as f:
            json.dump(target_summary, f, indent=4)

        # 9. HTML Report Dashboard
        self.compile_html_report(
            report_dir,
            target_summary,
            df_dist,
            df_imb,
            df_rate,
            df_time,
            df_ident,
            df_amt,
            df_stats,
        )
        logger.info("Target Variable Analysis completed successfully.")

    def analyze_distribution(self, report_dir: Path) -> pd.DataFrame:
        """Analyzes overall distribution of the target variable.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame of fraud distribution properties.
        """
        logger.info("Analyzing fraud distribution...")
        total_tx = len(self.df_train)
        if self.target_col not in self.df_train.columns:
            raise ValueError(f"Target column '{self.target_col}' not in training data.")

        y_counts = self.df_train[self.target_col].value_counts(dropna=False)
        fraud_cnt = int(y_counts.get(1, 0))
        legit_cnt = int(y_counts.get(0, 0))
        nan_cnt = int(y_counts.drop([0, 1], errors="ignore").sum())

        records = [
            {"class": "Total", "count": total_tx, "percentage": 100.0},
            {
                "class": "Fraud",
                "count": fraud_cnt,
                "percentage": (
                    float(fraud_cnt / total_tx * 100) if total_tx > 0 else 0.0
                ),
            },
            {
                "class": "Legitimate",
                "count": legit_cnt,
                "percentage": (
                    float(legit_cnt / total_tx * 100) if total_tx > 0 else 0.0
                ),
            },
        ]
        if nan_cnt > 0:
            records.append({
                "class": "NaN",
                "count": nan_cnt,
                "percentage": float(nan_cnt / total_tx * 100) if total_tx > 0 else 0.0,
            })

        df_dist = pd.DataFrame(records)
        df_dist.to_csv(report_dir / "fraud_distribution.csv", index=False)

        # Convert to JSON dict specifically for saving the raw JSON
        dist_json = {
            "total_transactions": total_tx,
            "fraud_transactions": fraud_cnt,
            "legitimate_transactions": legit_cnt,
            "fraud_percentage": (
                float(fraud_cnt / total_tx * 100) if total_tx > 0 else 0.0
            ),
            "non_fraud_percentage": (
                float(legit_cnt / total_tx * 100) if total_tx > 0 else 0.0
            ),
        }
        with (report_dir / "fraud_distribution.json").open("w", encoding="utf-8") as f:
            json.dump(dist_json, f, indent=4)

        # Visualization
        plt.figure(figsize=(12, 5))

        # 1. Count plot
        plt.subplot(1, 2, 1)
        sns.barplot(
            x=["Legitimate", "Fraud"],
            y=[legit_cnt, fraud_cnt],
            hue=["Legitimate", "Fraud"],
            palette=["#00e5ff", "#ff3366"],
            legend=False,
        )
        plt.title("Transaction Counts by Class")
        plt.ylabel("Count")
        plt.yscale("log" if fraud_cnt > 0 else "linear")
        plt.grid(axis="y", linestyle="--", alpha=0.3)

        # 2. Pie chart
        plt.subplot(1, 2, 2)
        sizes = [legit_cnt, fraud_cnt]
        if sum(sizes) > 0:
            plt.pie(
                sizes,
                labels=["Legitimate", "Fraud"],
                autopct="%1.2f%%",
                startangle=140,
                colors=["#00e5ff", "#ff3366"],
                textprops={"color": "white", "weight": "bold"},
                wedgeprops={"edgecolor": "#1a1f2c", "linewidth": 2},
            )
            # Make circular pie
            plt.gcf().set_facecolor("#1a1f2c")
            plt.gca().set_facecolor("#1a1f2c")
            plt.title("Class Percentage Share", color="white")
        else:
            plt.text(0.5, 0.5, "No data available", ha="center")

        plt.tight_layout()
        plt.savefig(report_dir / "fraud_distribution_plot.png", dpi=100)
        plt.close()

        return df_dist

    def analyze_class_imbalance(self, report_dir: Path) -> pd.DataFrame:
        """Evaluates details of class imbalance severity.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame containing class imbalance severity indicators.
        """
        logger.info("Analyzing class imbalance severity...")
        total_tx = len(self.df_train)
        y_counts = self.df_train[self.target_col].value_counts()
        fraud_cnt = int(y_counts.get(1, 0))
        legit_cnt = int(y_counts.get(0, 0))

        ratio_val = float(legit_cnt / fraud_cnt) if fraud_cnt > 0 else 0.0
        minority_pct = float(fraud_cnt / total_tx * 100) if total_tx > 0 else 0.0
        majority_pct = float(legit_cnt / total_tx * 100) if total_tx > 0 else 0.0
        severity = classify_imbalance_severity(ratio_val)

        rec_actions = (
            "Model training requires extreme precaution: "
            "1. Use Stratified K-Fold to prevent empty targets. "
            "2. Optimize for AUPRC/AUROC instead of Accuracy. "
            "3. Apply scale_pos_weight or class_weights."
        ) if severity == "Severe" else (
            "Model training requires standard precaution: "
            "1. Utilize stratified splits. "
            "2. Use balanced metrics (AUPRC)."
        )

        df_imb = pd.DataFrame([{
            "legitimate_count": legit_cnt,
            "fraud_count": fraud_cnt,
            "majority_percentage": majority_pct,
            "minority_percentage": minority_pct,
            "class_ratio": f"{ratio_val:.2f}:1",
            "imbalance_ratio": ratio_val,
            "severity": severity,
            "recommended_strategy": rec_actions,
        }])
        df_imb.to_csv(report_dir / "class_imbalance_report.csv", index=False)
        return df_imb

    def analyze_fraud_rates(self, report_dir: Path) -> pd.DataFrame:
        """Computes basic fraud rate aggregates.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame of overarching fraud/legitimate rates.
        """
        logger.info("Computing overall fraud rates...")
        total = len(self.df_train)
        y_counts = self.df_train[self.target_col].value_counts()
        fraud_cnt = y_counts.get(1, 0)
        legit_cnt = y_counts.get(0, 0)

        df_rate = pd.DataFrame([{
            "metric": "Fraud Rate",
            "description": "Percentage of fraudulent transactions out of the total",
            "value": float(fraud_cnt / total * 100) if total > 0 else 0.0,
        }, {
            "metric": "Legitimate Rate",
            "description": "Percentage of legitimate transactions out of the total",
            "value": float(legit_cnt / total * 100) if total > 0 else 0.0,
        }])
        df_rate.to_csv(report_dir / "fraud_rate_summary.csv", index=False)
        return df_rate

    def analyze_fraud_by_time(self, report_dir: Path) -> pd.DataFrame:
        """Evaluates fraud rates and transaction volumes over TransactionDT.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame with hourly, daily, and weekly stats.
        """
        logger.info("Analyzing fraud trends over time (TransactionDT)...")

        # In IEEE-CIS, TransactionDT represents time offsets in seconds.
        # We construct virtual bins:
        # Hour of day (0-23)
        # Day of study (integer starting from 0)
        # Day of week (0-6)
        # Week of study (integer starting from 0)
        df_temp = self.df_train[[self.time_col, self.target_col]].copy()

        df_temp["hour"] = (df_temp[self.time_col] // 3600) % 24
        df_temp["day"] = df_temp[self.time_col] // 86400
        df_temp["day_of_week"] = df_temp["day"] % 7
        df_temp["week"] = df_temp[self.time_col] // (86400 * 7)

        # 1. Hourly Trend
        hourly_summary = df_temp.groupby("hour")[self.target_col].agg(
            total_count="count",
            fraud_count="sum",
            fraud_rate=lambda x: float(x.sum() / len(x) * 100) if len(x) > 0 else 0.0,
        ).reset_index()

        # 2. Daily Trend
        daily_summary = df_temp.groupby("day")[self.target_col].agg(
            total_count="count",
            fraud_count="sum",
            fraud_rate=lambda x: float(x.sum() / len(x) * 100) if len(x) > 0 else 0.0,
        ).reset_index()

        # 3. Weekly Trend
        weekly_summary = df_temp.groupby("week")[self.target_col].agg(
            total_count="count",
            fraud_count="sum",
            fraud_rate=lambda x: float(x.sum() / len(x) * 100) if len(x) > 0 else 0.0,
        ).reset_index()

        # Export consolidated CSV format
        records = []
        for _, row in hourly_summary.iterrows():
            records.append({
                "time_bucket": "hour",
                "bucket_value": int(row["hour"]),
                "total_transactions": int(row["total_count"]),
                "fraud_count": int(row["fraud_count"]),
                "fraud_rate": float(row["fraud_rate"]),
            })
        for _, row in daily_summary.iterrows():
            records.append({
                "time_bucket": "day",
                "bucket_value": int(row["day"]),
                "total_transactions": int(row["total_count"]),
                "fraud_count": int(row["fraud_count"]),
                "fraud_rate": float(row["fraud_rate"]),
            })
        for _, row in weekly_summary.iterrows():
            records.append({
                "time_bucket": "week",
                "bucket_value": int(row["week"]),
                "total_transactions": int(row["total_count"]),
                "fraud_count": int(row["fraud_count"]),
                "fraud_rate": float(row["fraud_rate"]),
            })

        df_time = pd.DataFrame(records)
        df_time.to_csv(report_dir / "fraud_by_time.csv", index=False)

        # Plot figures
        _fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Hourly distribution
        sns.barplot(
            data=hourly_summary,
            x="hour",
            y="fraud_rate",
            ax=axes[0, 0],
            color="#ff3366",
        )
        axes[0, 0].set_title("Fraud Rate by Hour of Day")
        axes[0, 0].set_ylabel("Fraud Rate (%)")
        axes[0, 0].set_xlabel("Hour (0 - 23)")
        axes[0, 0].grid(linestyle="--", alpha=0.3)

        # Daily Trend line
        sns.lineplot(
            data=daily_summary,
            x="day",
            y="fraud_rate",
            ax=axes[0, 1],
            color="#ff3366",
            linewidth=2,
            marker="o",
        )
        axes[0, 1].set_title("Daily Fraud Rate Trend")
        axes[0, 1].set_ylabel("Fraud Rate (%)")
        axes[0, 1].set_xlabel("Relative Day of Study")
        axes[0, 1].grid(linestyle="--", alpha=0.3)

        # Transaction Volume vs Fraud Volume
        sns.lineplot(
            data=daily_summary,
            x="day",
            y="total_count",
            ax=axes[1, 0],
            color="#00e5ff",
            label="Total Vol",
            linewidth=2,
        )
        ax2 = axes[1, 0].twinx()
        sns.lineplot(
            data=daily_summary,
            x="day",
            y="fraud_count",
            ax=ax2,
            color="#ff3366",
            label="Fraud Vol",
            linewidth=2,
        )
        axes[1, 0].set_title("Daily Transaction vs Fraud Volume")
        axes[1, 0].set_ylabel("Total Transactions Count")
        ax2.set_ylabel("Fraud Count (Right)")
        axes[1, 0].grid(linestyle="--", alpha=0.3)

        # Rolling Fraud Rate (7-day window)
        daily_summary["rolling_rate"] = (
            daily_summary["fraud_rate"].rolling(window=7, min_periods=1).mean()
        )
        sns.lineplot(
            data=daily_summary,
            x="day",
            y="rolling_rate",
            ax=axes[1, 1],
            color="#ffc107",
            linewidth=2,
        )
        axes[1, 1].set_title("7-Day Rolling Average Fraud Rate")
        axes[1, 1].set_ylabel("Rolling Avg Fraud Rate (%)")
        axes[1, 1].set_xlabel("Relative Day of Study")
        axes[1, 1].grid(linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(report_dir / "fraud_by_time_plot.png", dpi=100)
        plt.close()

        return df_time

    def analyze_fraud_by_identity(self, report_dir: Path) -> pd.DataFrame:
        """Analyzes fraud rates and completeness across identity features.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame of identity columns summary metrics.
        """
        logger.info("Analyzing fraud rates by Identity features...")
        records = []

        for col in self.identity_cols:
            if col not in self.df_train.columns:
                continue

            series = self.df_train[col]
            y = self.df_train[self.target_col]

            # Null indicator
            is_null = series.isna()
            null_count = int(is_null.sum())
            null_pct = float(null_count / len(self.df_train) * 100)

            # Null group fraud rate
            null_y = y[is_null]

            # Available group properties
            avail_y = y[~is_null]
            avail_total = len(avail_y)
            avail_fraud_rate = float(avail_y.mean() * 100) if avail_total > 0 else 0.0

            # Top categories analysis (if categorical cardinality is reasonable)
            top_cat_name = "N/A"
            top_cat_fraud_rate = 0.0
            cat_nunique = int(series.nunique())

            if cat_nunique > 0 and series.dtype == "object":
                top_cats = series.value_counts().head(20).index
                mask = series.isin(top_cats)
                cat_summary = (
                    self.df_train[mask]
                    .groupby(col)[self.target_col]
                    .agg(
                        count="count",
                        fraud_rate=lambda x: (
                            float(x.sum() / len(x) * 100) if len(x) > 0 else 0.0
                        )
                    )
                    .reset_index()
                )
                if not cat_summary.empty:
                    sorted_cats = cat_summary.sort_values(
                        by="fraud_rate", ascending=False
                    )
                    top_row = sorted_cats.iloc[0]
                    top_cat_name = str(top_row[col])
                    top_cat_fraud_rate = float(top_row["fraud_rate"])

            records.append({
                "identity_column": col,
                "missing_pct": null_pct,
                "missing_fraud_rate": null_y.mean() if len(null_y) > 0 else 0.0,
                "available_count": avail_total,
                "available_fraud_rate": avail_fraud_rate,
                "cardinality": cat_nunique,
                "highest_risk_category": top_cat_name,
                "highest_risk_category_fraud_rate": top_cat_fraud_rate,
            })

        df_ident = pd.DataFrame(records)
        df_ident = df_ident.sort_values(by="available_fraud_rate", ascending=False)
        df_ident.to_csv(report_dir / "fraud_identity_analysis.csv", index=False)

        # Plot Heatmap or Top identities fraud rate comparison
        top_ident_display = df_ident.head(15)
        if not top_ident_display.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=top_ident_display,
                y="identity_column",
                x="available_fraud_rate",
                color="#00e5ff",
            )
            plt.title("Available Group Fraud Rate (%) across Top 15 Identity Columns")
            plt.xlabel("Fraud Rate (%)")
            plt.ylabel("Identity Column")
            plt.grid(axis="x", linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.savefig(report_dir / "fraud_identity_plot.png", dpi=100)
            plt.close()

        return df_ident

    def analyze_fraud_by_amount(self, report_dir: Path) -> pd.DataFrame:
        """Analyzes fraud rates by binned transaction amount.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame of transaction amount ranges.
        """
        logger.info("Analyzing fraud rates by transaction amount partitions...")

        df_temp = self.df_train[[self.amt_col, self.target_col]].copy()
        df_temp = df_temp.dropna(subset=[self.amt_col])

        # Define bins using quantiles
        # 0 - 10%, 10% - 30%, 30% - 70%, 70% - 90%, 90% - 100%
        bins = [0.0, 10.0, 30.0, 70.0, 90.0, 100.0]
        bin_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

        q_cuts = [df_temp[self.amt_col].quantile(q / 100.0) for q in bins]
        # Ensure unique cut edges
        q_cuts = sorted(set(q_cuts))
        if len(q_cuts) < 2:
            q_cuts = [0.0, df_temp[self.amt_col].max() + 1.0]
            bin_labels = ["All"]
        else:
            q_cuts[0] = 0.0  # Force start at 0
            bin_labels = bin_labels[:len(q_cuts) - 1]

        df_temp["amount_bin"] = pd.cut(
            df_temp[self.amt_col],
            bins=q_cuts,
            labels=bin_labels,
            include_lowest=True,
        )

        amount_summary = (
            df_temp.groupby("amount_bin", observed=True)[self.target_col]
            .agg(
                total_count="count",
                fraud_count="sum",
                fraud_rate=lambda x: (
                    float(x.sum() / len(x) * 100) if len(x) > 0 else 0.0
                )
            )
            .reset_index()
        )

        # Compute range bounds
        range_bounds = []
        for i in range(len(q_cuts) - 1):
            range_bounds.append(f"${q_cuts[i]:.2f} - ${q_cuts[i+1]:.2f}")

        amount_summary["amount_range"] = range_bounds[:len(amount_summary)]
        amount_summary.to_csv(report_dir / "fraud_amount_analysis.csv", index=False)

        # Plot charts
        _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Bar of Fraud Rate by Bin
        sns.barplot(
            data=amount_summary,
            x="amount_bin",
            y="fraud_rate",
            ax=axes[0],
            palette="plasma",
            hue="amount_bin",
            legend=False,
        )
        axes[0].set_title("Fraud Rate (%) by Transaction Amount Range")
        axes[0].set_ylabel("Fraud Rate (%)")
        axes[0].set_xlabel("Amount Bin")
        axes[0].grid(axis="y", linestyle="--", alpha=0.3)

        # 2. KDE Boxplot log of transaction amount
        df_temp["log_amt"] = np.log1p(df_temp[self.amt_col])
        sns.boxplot(
            data=df_temp,
            x=self.target_col,
            y="log_amt",
            ax=axes[1],
            palette=["#00e5ff", "#ff3366"],
            hue=self.target_col,
            legend=False,
        )
        axes[1].set_title("Log-transformed Trans Amt Distribution: Legitimate vs Fraud")
        axes[1].set_xticklabels(["Legitimate", "Fraud"])
        axes[1].set_ylabel("Log(TransactionAmt + 1)")
        axes[1].grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(report_dir / "fraud_amount_plot.png", dpi=100)
        plt.close()

        return amount_summary

    def generate_statistical_summary(self, report_dir: Path) -> pd.DataFrame:
        """Computes summary statistics for fraud and legitimate populations.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame containing summarized stats for key numeric variables.
        """
        logger.info("Generating target-specific statistical summary...")

        # Select numeric columns to summarize
        valid_dtypes = ["float32", "float64", "int64", "int32"]
        v_cols = [
            c for c in self.common_cols
            if self.df_train[c].dtype in valid_dtypes
        ]
        v_cols = [
            c for c in v_cols
            if c not in [self.target_col, "TransactionID", self.time_col]
        ]

        # Keep top columns: e.g. TransactionAmt, C-columns, D-columns
        key_cols = [self.amt_col]
        chk_cols = [
            "card1", "card2", "card3", "card5", "dist1",
            "C1", "C14", "D1", "D15"
        ]
        for col in chk_cols:
            if col in v_cols:
                key_cols.append(col)

        records = []
        for col in key_cols:
            series = self.df_train[col]
            y = self.df_train[self.target_col]

            for class_val, class_name in [(0, "Legitimate"), (1, "Fraud")]:
                subset = series[y == class_val].dropna()
                records.append({
                    "column": col,
                    "class": class_name,
                    "count": len(subset),
                    "mean": float(subset.mean()) if len(subset) > 0 else 0.0,
                    "std": float(subset.std()) if len(subset) > 1 else 0.0,
                    "min": float(subset.min()) if len(subset) > 0 else 0.0,
                    "p25": float(subset.quantile(0.25)) if len(subset) > 0 else 0.0,
                    "median": float(subset.median()) if len(subset) > 0 else 0.0,
                    "p75": float(subset.quantile(0.75)) if len(subset) > 0 else 0.0,
                    "max": float(subset.max()) if len(subset) > 0 else 0.0,
                })

        df_stats = pd.DataFrame(records)
        df_stats.to_csv(report_dir / "fraud_statistics.csv", index=False)
        return df_stats

    def generate_recommendations(self, _report_dir: Path) -> dict[str, Any]:
        """Automatically generates modeling strategy recommendations.

        Args:
            _report_dir: Output base directory.

        Returns:
            Recommendations dictionary structure.
        """
        logger.info("Formulating class imbalance handling recommendations...")

        metrics_recs = [
            "Use Area Under ROC Curve (AUROC) and Area Under "
            "Precision-Recall Curve (AUPRC) as optimization targets.",
            "Avoid Accuracy metric, which is highly misleading "
            "under severe class imbalance."
        ]

        cross_val_recs = [
            "Implement Stratified Cross-Validation (K-Fold) "
            "to maintain consistent class proportions.",
            "To safeguard against temporal leakage, utilize "
            "time-based splits (e.g. TimeSeriesSplit) or walk-forward "
            "validation."
        ]

        imbalance_recs = [
            "Enable loss weighting or class weights (e.g. scale_pos_weight "
            "in XGBoost, class_weight='balanced' in Random Forest).",
            "Consider downsampling the majority class (legitimate "
            "transactions) or using synthetic oversampling (SMOTE) "
            "with caution."
        ]

        engineering_recs = [
            "Log-transform the TransactionAmt feature due to "
            "heavy skewness.",
            "Engineer identity aggregation features (e.g., mean/max "
            "transaction amounts grouped by DeviceInfo / Browser)."
        ]

        recs = {
            "evaluation_metrics": metrics_recs,
            "cross_validation_strategy": cross_val_recs,
            "imbalance_mitigation": imbalance_recs,
            "feature_engineering_suggestions": engineering_recs,
        }
        return recs

    def compile_html_report(
        self,
        report_dir: Path,
        summary: dict[str, Any],
        df_dist: pd.DataFrame,
        _df_imb: pd.DataFrame,
        df_rate: pd.DataFrame,
        _df_time: pd.DataFrame,
        df_ident: pd.DataFrame,
        df_amt: pd.DataFrame,
        df_stats: pd.DataFrame,
    ) -> None:
        """Compiles clean glassmorphic HTML report dashboard document."""
        # Top distribution records
        dist_rows = ""
        for _, row in df_dist.iterrows():
            dist_rows += f"""
            <tr>
                <td style="font-weight: 500;">{row['class']}</td>
                <td>{row['count']:,}</td>
                <td>{row['percentage']:.4f}%</td>
            </tr>
            """

        # Rates summary records
        rate_rows = ""
        for _, row in df_rate.iterrows():
            rate_rows += f"""
            <tr>
                <td style="font-weight: 500;">{row['metric']}</td>
                <td>{row['description']}</td>
                <td><span class="badge badge-teal">{row['value']:.4f}%</span></td>
            </tr>
            """

        # Amount ranges summary records
        amt_rows = ""
        for _, row in df_amt.iterrows():
            amt_rows += f"""
            <tr>
                <td style="font-weight: 500;">{row['amount_bin']}</td>
                <td>{row['amount_range']}</td>
                <td>{row['total_count']:,}</td>
                <td>{row['fraud_count']:,}</td>
                <td>
                    <span class="badge badge-warning">
                        {row['fraud_rate']:.4f}%
                    </span>
                </td>
            </tr>
            """

        # Risk Identity features summary records
        ident_rows = ""
        for _, row in df_ident.head(15).iterrows():
            ident_rows += f"""
            <tr>
                <td style="font-weight: 500;">{row['identity_column']}</td>
                <td>{row['missing_pct']:.2f}%</td>
                <td>{row['available_count']:,}</td>
                <td>{row['available_fraud_rate']:.4f}%</td>
                <td>{row['highest_risk_category']}</td>
                <td>{row['highest_risk_category_fraud_rate']:.2f}%</td>
            </tr>
            """

        # Statistics records
        stats_rows = ""
        for _, row in df_stats.head(20).iterrows():
            stats_rows += f"""
            <tr>
                <td style="font-weight: 500;">{row['column']}</td>
                <td><span class="badge badge-teal">{row['class']}</span></td>
                <td>{row['count']:,}</td>
                <td>{row['mean']:.2f}</td>
                <td>{row['std']:.2f}</td>
                <td>{row['min']:.2f}</td>
                <td>{row['median']:.2f}</td>
                <td>{row['max']:.2f}</td>
            </tr>
            """

        # Gather recommendation blocks into list tags
        recs = summary["recommendations"]
        eval_list = "".join(
            f"<li>{item}</li>" for item in recs["evaluation_metrics"]
        )
        cv_list = "".join(
            f"<li>{item}</li>" for item in recs["cross_validation_strategy"]
        )
        imb_list = "".join(
            f"<li>{item}</li>" for item in recs["imbalance_mitigation"]
        )
        eng_list = "".join(
            f"<li>{item}</li>"
            for item in recs["feature_engineering_suggestions"]
        )

        # Compile CSS styles and HTML template
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
    <title>IEEE-CIS Target Variable Diagnostics</title>
    <link href="{font_url}" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #06070b;
            --card-bg: rgba(18, 22, 32, 0.45);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-main: #ffffff;
            --text-muted: #8e97a4;
            --accent: #ffffff;
            --warning: #8e97a4;
            --danger: #d63031;
            --success: #13b981;
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
            color: var(--text-main);
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
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .logo {{
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
            padding: 40px;
            max-width: 1600px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }}

        .grid-4 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }}

        .glass-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 4px;
            padding: 25px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
            position: relative;
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
            border: 1px solid var(--card-border);
        }}
        .badge-warning {{
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-muted);
            border: 1px solid var(--card-border);
        }}
        .badge-danger {{
            background: rgba(214, 48, 49, 0.08);
            color: var(--danger);
            border: 1px solid rgba(214, 48, 49, 0.2);
        }}

        .sect-title {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 8px;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .sect-title::after {{
            content: "[SECTION.TARGET]";
            font-size: 0.70rem;
            color: var(--text-muted);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 0.85rem;
        }}
        th {{
            text-align: left;
            padding: 10px 12px;
            border-bottom: 2px solid var(--card-border);
            color: #ffffff;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--card-border);
            color: var(--text-muted);
        }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.03); }}

        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }}

        @media(max-width: 768px) {{
            .grid-2 {{
                grid-template-columns: 1fr;
            }}
        }}

        .visual-img {{
            width: 100%;
            border-radius: 0;
            border: 1px solid var(--card-border);
            margin-top: 10px;
            filter: grayscale(100%) contrast(1.1) brightness(0.9);
        }}

        .alert-box {{
            padding: 16px;
            border-radius: 4px;
            margin-bottom: 20px;
            font-size: 0.85rem;
            line-height: 1.5;
        }}
        .alert-danger {{
            background: rgba(214, 48, 49, 0.08);
            border: 1px solid rgba(214, 48, 49, 0.2);
            color: var(--danger);
        }}

        .rec-rec {{
            margin-bottom: 25px;
            border-left: 2px solid #ffffff;
            padding-left: 15px;
        }}
        .rec-rec h4 {{
            margin: 5px 0;
            font-family: 'Orbitron', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.95rem;
            color: #ffffff;
        }}
        .rec-rec ul {{
            margin: 5px 0;
            padding-left: 20px;
            font-size: 0.85rem;
            color: var(--text-muted);
            line-height: 1.6;
        }}
        .rec-rec li {{
            margin-bottom: 4px;
        }}
    </style>
</head>
<body>
    <div class="scanline-overlay"></div>
    <div class="hud-grid-bg"></div>
    <header>
        <div class="logo">IEEE-CIS Target variable diagnostics</div>
        <div class="nav">
            <a href="#summary">Summary</a>
            <a href="#visuals">Visuals</a>
            <a href="#tables">Tables</a>
            <a href="#recommendations">Recommendations</a>
        </div>
        <div class="hud-status">
            <span class="pulse-dot"></span>
            <span>TARGET ANALYZER OK</span>
        </div>
    </header>
    <main>
        <div id="summary" class="grid-4">
            <div class="glass-card">
                <div class="metric-title">Total Transactions</div>
                <div class="metric-value">
                    {summary['total_transactions']:,}
                </div>
                <div class="metric-desc">Overall observations analyzed</div>
            </div>
            <div class="glass-card">
                <div class="metric-title">Fraud Transactions</div>
                <div class="metric-value" style="color: var(--danger);">
                    {summary['fraud_count']:,}
                </div>
                <div class="metric-desc">Count of labeled fraud points</div>
            </div>
            <div class="glass-card">
                <div class="metric-title">Legitimate Transactions</div>
                <div class="metric-value" style="color: var(--success);">
                    {summary['legit_count']:,}
                </div>
                <div class="metric-desc">Count of legitimate points</div>
            </div>
            <div class="glass-card">
                <div class="metric-title">Imbalance Severity</div>
                <div class="metric-value">
                    {summary['imbalance_severity']}
                </div>
                <div class="metric-desc">
                    Ratio: {summary['imbalance_ratio']:.2f}:1
                </div>
            </div>
        </div>

        <div class="alert-box alert-danger">
            <strong>Imbalance Flagged:</strong>
            Legitimate vs Fraud ratio is {summary['imbalance_ratio']:.2f}:1.
            This constitutes <strong>{summary['imbalance_severity']}</strong>
            class imbalance. Accuracy is not a valid evaluation metric.
            You must optimize models using AUROC/AUPRC and apply loss weighting.
        </div>

        <div id="visuals" class="grid-2">
            <div class="glass-card">
                <div class="sect-title">Target Class Distribution</div>
                <img src="fraud_distribution_plot.png"
                     class="visual-img"
                     alt="Target Distribution">
            </div>
            <div class="glass-card">
                <div class="sect-title">Daily & Hour Temporal Trends</div>
                <img src="fraud_by_time_plot.png"
                     class="visual-img"
                     alt="Temporal Trends">
            </div>
        </div>

        <div class="grid-2">
            <div class="glass-card">
                <div class="sect-title">Identity features Risk</div>
                <img src="fraud_identity_plot.png"
                     class="visual-img"
                     alt="Identity Risk">
            </div>
            <div class="glass-card">
                <div class="sect-title">Transaction Amount Risk Bins</div>
                <img src="fraud_amount_plot.png"
                     class="visual-img"
                     alt="Amount Bins">
            </div>
        </div>

        <div id="tables" class="grid-2">
            <div class="glass-card">
                <div class="sect-title">Class Distribution Grid</div>
                <table>
                    <thead>
                        <tr>
                            <th>Class</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {dist_rows}
                    </tbody>
                </table>
            </div>

            <div class="glass-card">
                <div class="sect-title">Fraud rates overview</div>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Description</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rate_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="glass-card" style="margin-bottom: 40px;">
            <div class="sect-title">Transaction Amount Range Analysis</div>
            <table>
                <thead>
                    <tr>
                        <th>Bin Label</th>
                        <th>Amount Bounds</th>
                        <th>Total Count</th>
                        <th>Fraud Count</th>
                        <th>Fraud Rate (%)</th>
                    </tr>
                </thead>
                <tbody>
                    {amt_rows}
                </tbody>
            </table>
        </div>

        <div class="glass-card" style="margin-bottom: 40px;">
            <div class="sect-title">
                Risk Analysis on Top 15 Identity Columns (DeviceInfo/DeviceType/id_*)
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Identity column</th>
                        <th>Missing %</th>
                        <th>Available Samples</th>
                        <th>Available Fraud Rate (%)</th>
                        <th>Highest Risk Category</th>
                        <th>Highest Risk Group Rate (%)</th>
                    </tr>
                </thead>
                <tbody>
                    {ident_rows}
                </tbody>
            </table>
        </div>

        <div class="glass-card" style="margin-bottom: 40px;">
            <div class="sect-title">
                Populations summary statistics (Mean/Std/Quartiles)
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Class Population</th>
                        <th>Valid Count</th>
                        <th>Mean</th>
                        <th>Std</th>
                        <th>Min</th>
                        <th>Median</th>
                        <th>Max</th>
                    </tr>
                </thead>
                <tbody>
                    {stats_rows}
                </tbody>
            </table>
        </div>

        <div id="recommendations" class="glass-card">
            <div class="sect-title">Modeling Strategy Recommendations</div>
            <div class="rec-rec">
                <h4>1. Evaluation Metric Selection</h4>
                <ul>{eval_list}</ul>
            </div>
            <div class="rec-rec">
                <h4>2. Cross-Validation Configuration</h4>
                <ul>{cv_list}</ul>
            </div>
            <div class="rec-rec">
                <h4>3. Class Imbalance Mitigation</h4>
                <ul>{imb_list}</ul>
            </div>
            <div class="rec-rec">
                <h4>4. Suggested Feature Engineering</h4>
                <ul>{eng_list}</ul>
            </div>
        </div>
    </main>
</body>
</html>
"""
        report_path = report_dir / "target_analysis_report.html"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(
            "Compiled Target HTML dashboard to: %s",
            report_path
        )
