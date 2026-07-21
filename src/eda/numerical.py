# ruff: noqa: E501
"""Numerical Feature Analysis engine for evaluating dataset numeric properties."""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def classify_skewness(skew: float) -> str:
    """Classifies skewness into categorical ranges.

    Args:
        skew: Skewness coefficient value.

    Returns:
        Categorical description of shape.
    """
    if pd.isna(skew):
        return "N/A"
    if -0.5 <= skew <= 0.5:
        return "Approximately Symmetric"
    if 0.5 < skew <= 1.0:
        return "Moderately Right-Skewed"
    if -1.0 <= skew < -0.5:
        return "Moderately Left-Skewed"
    if skew > 1.0:
        return "Highly Right-Skewed"
    return "Highly Left-Skewed"


def classify_kurtosis(kurt: float) -> str:
    """Classifies kurtosis to evaluate tail weight.

    Args:
        kurt: Kurtosis value (unadjusted or excess).

    Returns:
        Categorical description of tail thickness.
    """
    if pd.isna(kurt):
        return "N/A"
    # Excess Kurtosis near 0 is normal (mesokurtic)
    excess_kurt = kurt
    if -0.5 <= excess_kurt <= 0.5:
        return "Normal"
    if excess_kurt > 0.5:
        return "Heavy-Tailed"
    return "Light-Tailed"


class NumericalFeatureAnalyzer:
    """Performs statistical audits for numerical columns in the dataset."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_col: str = "isFraud",
    ) -> None:
        """Initializes NumericalFeatureAnalyzer.

        Args:
            df_train: Merged train DataFrame.
            df_test: Merged test DataFrame.
            target_col: Target column name.
        """
        self.df_train = df_train
        self.df_test = df_test
        self.target_col = target_col

        # Exclude IDs, datetimes and labels from numerical list
        self.exclude_cols = ["TransactionID", "TransactionDT", self.target_col]

        # Scan numerical features from training columns
        self.numerical_cols = [
            col for col in self.df_train.columns
            if col not in self.exclude_cols
            and self.df_train[col].dtype in [
                "float32", "float64", "int32", "int64"
            ]
        ]
        logger.info(
            "Identified %d numerical features for analysis.",
            len(self.numerical_cols)
        )

        # Subset profile features to perform visualization plotting (avoids V-column processing bottlenecks)
        self.plot_cols = [
            "TransactionAmt", "card1", "card2", "card5", "dist1",
            "C1", "C14", "D1", "D15", "V10", "V50", "V300"
        ]
        self.plot_cols = [
            col for col in self.plot_cols if col in self.numerical_cols
        ]

    def analyze_all(self, report_dir: Path) -> None:
        """Executes all analysis tasks and outputs results.

        Args:
            report_dir: Output reports directory.
        """
        logger.info("Initializing Numerical Feature Diagnostics...")
        report_dir.mkdir(parents=True, exist_ok=True)

        # 1. Feature inventory listing
        self.save_feature_lists(report_dir)

        # 2. Basic stats & distributions
        df_dist = self.analyze_distributions(report_dir)

        # 3. Outlier diagnostics
        df_outliers, outlier_sum = self.analyze_outliers(report_dir)

        # 4. Skewness and Kurtosis evaluation
        df_skew = self.analyze_skewness(report_dir)
        df_kurt = self.analyze_kurtosis(report_dir)

        # 5. Log Transformations Recommendations
        df_recs = self.formulate_transformations(report_dir, df_dist, df_skew, df_kurt)

        # 6. Generate plots
        self.generate_plots(report_dir)

        # 7. Compile JSON Summary
        summary = {
            "total_numerical_features": len(self.numerical_cols),
            "plot_subset_count": len(self.plot_cols),
            "highly_skewed_count": int((df_skew["skewness"].abs() > 1.0).sum()),
            "heavy_tailed_count": int(
                (df_kurt["kurtosis_type"] == "Heavy-Tailed").sum()
            ),
            "outlier_summary": outlier_sum,
        }
        with (report_dir / "numerical_analysis.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        # 8. Render HTML layout
        self.compile_html_dashboard(
            report_dir,
            summary,
            df_dist,
            df_outliers,
            df_skew,
            df_kurt,
            df_recs,
        )
        logger.info("Numerical Feature Analysis completed successfully.")

    def save_feature_lists(self, report_dir: Path) -> None:
        """Saves identified numerical feature lists.

        Args:
            report_dir: Output base directory.
        """
        df_feat = pd.DataFrame({"feature": self.numerical_cols})
        df_feat.to_csv(report_dir / "numerical_features.csv", index=False)

        summary = {
            "total_numerical_count": len(self.numerical_cols),
            "features": self.numerical_cols,
        }
        with (report_dir / "numerical_feature_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

    def analyze_distributions(self, report_dir: Path) -> pd.DataFrame:
        """Computes statistical distribution properties.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame containing distribution metrics.
        """
        logger.info("Evaluating numerical feature distributions...")
        records = []

        for col in self.numerical_cols:
            series = self.df_train[col].dropna()
            if series.empty:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            records.append({
                "feature": col,
                "count": len(series),
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()) if len(series) > 1 else 0.0,
                "variance": float(series.var()) if len(series) > 1 else 0.0,
                "range": float(series.max() - series.min()),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
            })

        df_dist = pd.DataFrame(records)
        df_dist.to_csv(report_dir / "distribution_statistics.csv", index=False)
        return df_dist

    def analyze_outliers(self, report_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Scans and calculates outliers using IQR, Z-Score, and Percentile methods.

        Args:
            report_dir: Output base directory.

        Returns:
            Tuple of outlier DataFrame and overall summary dict.
        """
        logger.info("Scanning for outliers...")
        records = []
        total_iqr_outliers = 0
        total_z_outliers = 0

        for col in self.numerical_cols:
            series = self.df_train[col].dropna()
            n = len(series)
            if n == 0:
                continue

            # 1. IQR Method
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            iqr_lower = q1 - 1.5 * iqr
            iqr_upper = q3 + 1.5 * iqr
            iqr_outliers_cnt = int(((series < iqr_lower) | (series > iqr_upper)).sum())
            iqr_pct = float(iqr_outliers_cnt / n * 100)
            total_iqr_outliers += iqr_outliers_cnt

            # 2. Z-Score Method
            mean = series.mean()
            std = series.std()
            if std > 0:
                z_scores = (series - mean) / std
                z_outliers_cnt = int((z_scores.abs() > 3).sum())
            else:
                z_outliers_cnt = 0
            z_pct = float(z_outliers_cnt / n * 100)
            total_z_outliers += z_outliers_cnt

            # 3. Percentile Method (1st and 99th percentiles)
            p1 = series.quantile(0.01)
            p99 = series.quantile(0.99)
            perc_outliers_cnt = int(((series < p1) | (series > p99)).sum())
            perc_pct = float(perc_outliers_cnt / n * 100)

            records.append({
                "feature": col,
                "iqr_outliers_count": iqr_outliers_cnt,
                "iqr_outliers_pct": iqr_pct,
                "zscore_outliers_count": z_outliers_cnt,
                "zscore_outliers_pct": z_pct,
                "percentile_outliers_count": perc_outliers_cnt,
                "percentile_outliers_pct": perc_pct,
            })

        df_outliers = pd.DataFrame(records)
        df_outliers = df_outliers.sort_values(by="iqr_outliers_pct", ascending=False)
        df_outliers.to_csv(report_dir / "outlier_analysis.csv", index=False)

        summary = {
            "total_iqr_outliers": total_iqr_outliers,
            "total_zscore_outliers": total_z_outliers,
        }
        with (report_dir / "outlier_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        return df_outliers, summary

    def analyze_skewness(self, report_dir: Path) -> pd.DataFrame:
        """Measures skewness coefficient for columns.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame containing skewness details.
        """
        logger.info("Computing skewness ratings...")
        records = []
        for col in self.numerical_cols:
            skew_val = float(self.df_train[col].skew())
            records.append({
                "feature": col,
                "skewness": skew_val,
                "skewness_classification": classify_skewness(skew_val),
            })

        df_skew = pd.DataFrame(records)
        df_skew = df_skew.sort_values(by="skewness", key=abs, ascending=False)
        df_skew.to_csv(report_dir / "skewness_report.csv", index=False)
        return df_skew

    def analyze_kurtosis(self, report_dir: Path) -> pd.DataFrame:
        """Measures excess kurtosis for columns.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame containing kurtosis details.
        """
        logger.info("Evaluating kurtosis tails...")
        records = []
        for col in self.numerical_cols:
            kurt_val = float(self.df_train[col].kurtosis())
            records.append({
                "feature": col,
                "kurtosis": kurt_val,
                "kurtosis_type": classify_kurtosis(kurt_val),
            })

        df_kurt = pd.DataFrame(records)
        df_kurt = df_kurt.sort_values(by="kurtosis", key=abs, ascending=False)
        df_kurt.to_csv(report_dir / "kurtosis_report.csv", index=False)
        return df_kurt

    def formulate_transformations(
        self,
        report_dir: Path,
        df_dist: pd.DataFrame,
        df_skew: pd.DataFrame,
        df_kurt: pd.DataFrame,
    ) -> pd.DataFrame:
        """Formulates log/root/robust transformations recommendations based on bounds/skew.

        Args:
            report_dir: Output base directory.
            df_dist: Distribution stats.
            df_skew: Skewness analysis.
            df_kurt: Kurtosis analysis.

        Returns:
            DataFrame recommendation options.
        """
        logger.info("Formulating transform recommendations...")

        df_m = df_dist.merge(df_skew, on="feature")
        df_m = df_m.merge(df_kurt, on="feature")

        recs = []
        for _, row in df_m.iterrows():
            col = row["feature"]
            min_val = row["min"]
            skew = row["skewness"]
            kurt_type = row["kurtosis_type"]

            suggestion = "No Transformation"
            reason = "Feature is relatively symmetric and normal."
            improvement = "Maintains original scale."

            # RightSkew logic
            if skew > 1.0:
                if min_val >= 0.0:
                    suggestion = "Log Transformation (log1p)"
                    reason = (
                        f"Highly right-skewed (skew={skew:.2f}) and positive bound. "
                        "Compresses right tail."
                    )
                    improvement = "Reduces skewness closer to symmetric range."
                else:
                    # Positive-negative elements, use Yeo-Johnson
                    suggestion = "Yeo-Johnson"
                    reason = (
                        f"Highly right-skewed (skew={skew:.2f}) and has negative values. "
                        "Handles zero/negative scales."
                    )
                    improvement = "Normalizes skewed components without positivity constraints."
            elif skew < -1.0:
                suggestion = "Quantile Transformation"
                reason = f"Highly left-skewed distribution (skew={skew:.2f})."
                improvement = "Forces linear relationships and standard normal shape."
            elif kurt_type == "Heavy-Tailed":
                suggestion = "Robust Scaler (IQR-based)"
                reason = "Heavy tails and high outlier density can destabilize Standard Scaler."
                improvement = "Robust scaling using median and quartile ranges."

            recs.append({
                "feature": col,
                "current_skewness": skew,
                "current_kurtosis": row["kurtosis"],
                "suggested_transformation": suggestion,
                "reasoning": reason,
                "expected_gain": improvement,
            })

        df_recs = pd.DataFrame(recs)
        df_recs.to_csv(report_dir / "transformation_recommendations.csv", index=False)
        return df_recs

    def generate_plots(self, report_dir: Path) -> None:
        """Produces distributions, KDE overlays, histograms, boxplots and violins.

        Args:
            report_dir: Output base directory.
        """
        logger.info(
            "Generating plots for subset columns..."
        )

        subdirs = ["kde_plots", "histograms", "boxplots", "violin_plots"]
        for sd in subdirs:
            (report_dir / sd).mkdir(parents=True, exist_ok=True)

        for col in self.plot_cols:
            subset_train = self.df_train[[col, self.target_col]].dropna()
            if subset_train.empty:
                continue

            # 1. Standard + Fraud/Legit KDE Overlays
            plt.figure(figsize=(10, 6))
            try:
                sns.kdeplot(
                    data=subset_train,
                    x=col,
                    hue=self.target_col,
                    common_norm=False,
                    fill=True,
                    alpha=0.4,
                    palette={0: "#00e5ff", 1: "#ff3366"},
                )
                plt.title(f"Kernel Density Estimation (KDE) - {col} (Legit vs Fraud)")
                plt.xlabel(col)
                plt.ylabel("Density")
                plt.grid(linestyle="--", alpha=0.3)
                plt.tight_layout()
                plt.savefig(report_dir / "kde_plots" / f"{col}_kde.png", dpi=100)
                plt.close()
            except Exception as e:
                logger.warning("Could not draw KDE for %s: %s", col, e)
                plt.close()

            # 2. Standard Histograms
            plt.figure(figsize=(10, 6))
            try:
                sns.histplot(
                    data=subset_train,
                    x=col,
                    hue=self.target_col,
                    multiple="stack",
                    bins=30,
                    alpha=0.6,
                    palette={0: "#00e5ff", 1: "#ff3366"},
                )
                plt.title(f"Histogram Plot - {col} Frequency Count")
                plt.grid(linestyle="--", alpha=0.3)
                plt.tight_layout()
                plt.savefig(report_dir / "histograms" / f"{col}_hist.png", dpi=100)
                plt.close()
            except Exception as e:
                logger.warning("Could not draw Histogram for %s: %s", col, e)
                plt.close()

            # 3. Boxplots
            plt.figure(figsize=(10, 6))
            try:
                sns.boxplot(
                    data=subset_train,
                    y=col,
                    x=self.target_col,
                    hue=self.target_col,
                    legend=False,
                    palette={0: "#00e5ff", 1: "#ff3366", "0": "#00e5ff", "1": "#ff3366"},
                )
                plt.title(f"Boxplot Outliers Diagnostics - {col}")
                plt.xticks([0, 1], ["Legitimate", "Fraud"])
                plt.grid(axis="y", linestyle="--", alpha=0.3)
                plt.tight_layout()
                plt.savefig(report_dir / "boxplots" / f"{col}_boxplot.png", dpi=100)
                plt.close()
            except Exception as e:
                logger.warning("Could not draw Boxplot for %s: %s", col, e)
                plt.close()

            # 4. Violin plots
            plt.figure(figsize=(10, 6))
            try:
                sns.violinplot(
                    data=subset_train,
                    y=col,
                    x=self.target_col,
                    hue=self.target_col,
                    legend=False,
                    palette={0: "#00e5ff", 1: "#ff3366", "0": "#00e5ff", "1": "#ff3366"},
                    split=False,
                )
                plt.title(f"Violin Density Shape - {col}")
                plt.xticks([0, 1], ["Legitimate", "Fraud"])
                plt.grid(axis="y", linestyle="--", alpha=0.3)
                plt.tight_layout()
                plt.savefig(report_dir / "violin_plots" / f"{col}_violin.png", dpi=100)
                plt.close()
            except Exception as e:
                logger.warning("Could not draw Violin plot for %s: %s", col, e)
                plt.close()

    def compile_html_dashboard(
        self,
        report_dir: Path,
        summary: dict[str, Any],
        df_dist: pd.DataFrame,
        df_outliers: pd.DataFrame,
        df_skew: pd.DataFrame,
        df_kurt: pd.DataFrame,
        df_recs: pd.DataFrame,
    ) -> None:
        """Assembles responsive premium glassmorphic report dashboard.

        Args:
            report_dir: Direction target.
            summary: General metrics.
            df_dist: Distribution stats.
            df_outliers: Outliers details.
            df_skew: Skew metrics.
            df_kurt: Kurtosis metrics.
            df_recs: Transformations suggestions.
        """
        # Build Stats Grid HTML
        stats_rows = ""
        for _, row in df_dist.head(20).iterrows():
            stats_rows += f"""
            <tr>
                <td style="font-weight: 600;">{row['feature']}</td>
                <td>{int(row['count']):,}</td>
                <td>{row['mean']:.4f}</td>
                <td>{row['median']:.4f}</td>
                <td>{row['std']:.4f}</td>
                <td>{row['min']:.4f}</td>
                <td>{row['max']:.4f}</td>
            </tr>
            """

        # Outliers Grid HTML
        outlier_rows = ""
        for _, row in df_outliers.head(20).iterrows():
            badge_color = "badge-teal" if row['iqr_outliers_pct'] < 5.0 else "badge-warning"
            outlier_rows += f"""
            <tr>
                <td style="font-weight: 600;">{row['feature']}</td>
                <td>{int(row['iqr_outliers_count']):,}</td>
                <td>
                    <span class="badge {badge_color}">
                        {row['iqr_outliers_pct']:.2f}%
                    </span>
                </td>
                <td>{int(row['zscore_outliers_count']):,}</td>
                <td>{row['zscore_outliers_pct']:.2f}%</td>
                <td>{int(row['percentile_outliers_count']):,}</td>
                <td>{row['percentile_outliers_pct']:.2f}%</td>
            </tr>
            """

        # Skew/Kurt Table HTML
        shape_rows = ""
        df_sh = df_skew.merge(df_kurt, on="feature")
        for _, row in df_sh.head(20).iterrows():
            skew_badge = "badge-teal" if "Symmetric" in row['skewness_classification'] else "badge-warning"
            kurt_badge = "badge-teal" if "Normal" in row['kurtosis_type'] else "badge-warning"
            shape_rows += f"""
            <tr>
                <td style="font-weight: 600;">{row['feature']}</td>
                <td>{row['skewness']:.4f}</td>
                <td><span class="badge {skew_badge}">{row['skewness_classification']}</span></td>
                <td>{row['kurtosis']:.4f}</td>
                <td><span class="badge {kurt_badge}">{row['kurtosis_type']}</span></td>
            </tr>
            """

        # Recs Table HTML
        rec_rows = ""
        for _, row in df_recs.head(25).iterrows():
            action_badge = (
                "badge-teal" if row['suggested_transformation'] == "No Transformation"
                else "badge-warning"
            )
            rec_rows += f"""
            <tr>
                <td style="font-weight: 600;">{row['feature']}</td>
                <td>{row['current_skewness']:.4f}</td>
                <td>
                    <span class="badge {action_badge}">
                        {row['suggested_transformation']}
                    </span>
                </td>
                <td>{row['reasoning']}</td>
                <td>{row['expected_gain']}</td>
            </tr>
            """

        # Embed Carousel slides for visualized features of interest
        slides = ""
        for idx, col in enumerate(self.plot_cols[:4]):
            slide_active = "active" if idx == 0 else ""
            slides += f"""
            <div class="carousel-slide {slide_active}" id="slide-{col}">
                <h3>Feature Detail Visualizations: {col}</h3>
                <div class="grid-2" style="margin-top: 15px;">
                    <div class="visual-card">
                        <div class="v-title">KDE Probability Density (Legit vs Fraud)</div>
                        <img src="kde_plots/{col}_kde.png" alt="KDE {col}">
                    </div>
                    <div class="visual-card">
                        <div class="v-title">Boxplot Outlier Fences</div>
                        <img src="boxplots/{col}_boxplot.png" alt="Boxplot {col}">
                    </div>
                </div>
                <div class="grid-2" style="margin-top: 20px;">
                    <div class="visual-card">
                        <div class="v-title">Standard Stacked Histogram</div>
                        <img src="histograms/{col}_hist.png" alt="Histogram {col}">
                    </div>
                    <div class="visual-card">
                        <div class="v-title">Violin Density Shape</div>
                        <img src="violin_plots/{col}_violin.png" alt="Violin {col}">
                    </div>
                </div>
            </div>
            """

        # Selector links for carousel
        carousel_nav = ""
        for col in self.plot_cols[:4]:
            carousel_nav += f"""
            <button onclick="showSlide('{col}')">{col}</button>
            """

        # Recommendations list items
        recs_summary_list = """
        <li>Apply <strong>log1p</strong> to features flagged as Highly Right-Skewed (e.g. TransactionAmt, C-columns) with non-negative bounds to align model weights.</li>
        <li>Preserve features categorized as Approximately Symmetric (skew in -0.5 to 0.5) to maintain numerical interpretation.</li>
        <li>Evaluate extreme outlier features using Robust Scaler (using median/IQR fractions) rather than Standard Scaler to avoid distribution skewing.</li>
        <li>Investigate features with high Z-score outlier ratios (&gt;10%) for potential data entry limits or transaction boundaries before exclusion.</li>
        <li>Enable tree-based ensembles (XGBoost, Random Forest) without standard scaling since they remain scale-invariant for raw distributions.</li>
        """

        font_url = (
            "https://fonts.googleapis.com/css2?"
            "family=Orbitron:wght@400;600;800;900&"
            "family=JetBrains+Mono:wght@400;700&"
            "family=Inter:wght@400;600&"
            "display=swap"
        )

        # Standard HTML String Template - No prefix to prevent JS/CSS formatting conflicts
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IEEE-CIS Numerical Feature Diagnostics</title>
    <link href="__FONT_URL__" rel="stylesheet">
    <style>
        :root {
            --bg-color: #06070b;
            --card-bg: rgba(18, 22, 32, 0.45);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-main: #ffffff;
            --text-muted: #8e97a4;
            --accent: #ffffff;
            --warning: #8e97a4;
            --danger: #d63031;
            --success: #13b981;
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        @keyframes scanline {
            0% { transform: translateY(-100%); }
            100% { transform: translateY(100%); }
        }
        @keyframes pulse-grey {
            0% { opacity: 0.4; }
            50% { opacity: 1.0; }
            100% { opacity: 0.4; }
        }
        .scanline-overlay {
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
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            padding-bottom: 60px;
            position: relative;
            overflow-x: hidden;
        }

        .hud-grid-bg {
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background-image: linear-gradient(rgba(255,255,255,0.01) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(255,255,255,0.01) 1px, transparent 1px);
            background-size: 40px 40px;
            pointer-events: none;
            z-index: 0;
        }

        header {
            background: rgba(6, 7, 11, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--card-border);
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.5rem;
            font-weight: 800;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .logo span {
            color: var(--text-muted);
            font-weight: 400;
        }
        .nav {
            display: flex;
            gap: 20px;
        }
        .nav a {
            color: var(--text-muted);
            text-decoration: none;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-weight: 500;
            font-size: 0.8rem;
            transition: color 0.2s;
        }
        .nav a:hover {
            color: #ffffff;
        }

        .hud-status {
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
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: var(--text-muted);
            border-radius: 50%;
            animation: pulse-grey 2s infinite;
        }

        main {
            max-width: 1400px;
            margin: 40px auto;
            padding: 0 20px;
            position: relative;
            z-index: 1;
        }
        .grid-4 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 45px;
        }
        .grid-2 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 25px;
            margin-bottom: 45px;
        }

        @media(max-width: 768px) {
            .grid-2 {
                grid-template-columns: 1fr;
            }
        }

        .glass-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 4px;
            padding: 24px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
            position: relative;
            transition: all 0.3s ease;
        }
        .glass-card::before {
            content: "";
            position: absolute;
            top: 0; left: 0; width: 6px; height: 6px;
            border-top: 1px solid var(--text-muted);
            border-left: 1px solid var(--text-muted);
        }
        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.08);
        }

        .metric-title {
            font-size: 0.75rem;
            font-family: 'Orbitron', sans-serif;
            text-transform: uppercase;
            color: var(--text-muted);
            font-weight: 600;
            letter-spacing: 0.08em;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: #fff;
            margin-bottom: 5px;
        }
        .metric-desc {
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .sect-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
            margin-bottom: 20px;
            color: #fff;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .sect-title::after {
            content: "[SECTION.NUMERICAL]";
            font-size: 0.70rem;
            color: var(--text-muted);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 0.85rem;
        }
        th, td {
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid var(--card-border);
        }
        th {
            color: #ffffff;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-weight: 600;
            font-size: 0.75rem;
        }
        td {
            color: var(--text-muted);
        }
        tr:hover td {
            background-color: rgba(255, 255, 255, 0.03);
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 0;
            font-size: 0.7rem;
            font-weight: 750;
            text-transform: uppercase;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.05em;
        }
        .badge-teal {
            background-color: rgba(255, 255, 255, 0.05);
            color: #ffffff;
            border: 1px solid var(--card-border);
        }
        .badge-warning {
            background-color: rgba(255, 255, 255, 0.03);
            color: var(--text-muted);
            border: 1px solid var(--card-border);
        }
        .badge-danger {
            background-color: rgba(214, 48, 49, 0.08);
            color: var(--danger);
            border: 1px solid rgba(214, 48, 49, 0.2);
        }
        .carousel-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .carousel-tabs button {
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--card-border);
            color: var(--text-muted);
            padding: 8px 16px;
            border-radius: 0;
            cursor: pointer;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            transition: all 0.2s;
        }
        .carousel-tabs button:hover, .carousel-tabs button.active {
            background-color: rgba(255, 255, 255, 0.08);
            color: #ffffff;
            border-color: rgba(255, 255, 255, 0.2);
        }
        .carousel-slide {
            display: none;
        }
        .carousel-slide h3 {
            font-family: 'Orbitron', sans-serif;
            font-size: 1rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #ffffff;
            margin-bottom: 15px;
        }
        .carousel-slide.active {
            display: block;
        }
        .visual-card {
            background-color: rgba(255, 255, 255, 0.01);
            border: 1px solid var(--card-border);
            border-radius: 0;
            padding: 16px;
            text-align: center;
        }
        .v-title {
            font-size: 0.75rem;
            font-family: 'Orbitron', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 12px;
            font-weight: 500;
        }
        .visual-card img {
            max-width: 100%;
            height: auto;
            border-radius: 0;
            border: 1px solid var(--card-border);
            filter: grayscale(100%) contrast(1.1) brightness(0.9);
        }
        .recs-box ul {
            list-style: none;
            padding-left: 0;
        }
        .recs-box li {
            margin-bottom: 16px;
            font-size: 0.85rem;
            line-height: 1.6;
            padding-left: 28px;
            position: relative;
            color: var(--text-muted);
        }
        .recs-box li::before {
            content: "✦";
            position: absolute;
            left: 0;
            top: 0;
            color: #ffffff;
            font-size: 1.1rem;
        }
    </style>
    <script>
        function showSlide(colName) {
            const slides = document.querySelectorAll('.carousel-slide');
            slides.forEach(slide => slide.classList.remove('active'));
            const buttons = document.querySelectorAll('.carousel-tabs button');
            buttons.forEach(btn => btn.classList.remove('active'));
            document.getElementById('slide-' + colName).classList.add('active');
            for (let btn of buttons) {
                if (btn.textContent.trim() === colName) {
                    btn.classList.add('active');
                    break;
                }
            }
        }
    </script>
</head>
<body>
    <div class="scanline-overlay"></div>
    <div class="hud-grid-bg"></div>
    <header>
        <div class="logo">IEEE-CIS <span>Numerical Diagnostics</span></div>
        <div class="nav">
            <a href="#summary">Summary</a>
            <a href="#visuals">Visuals</a>
            <a href="#statistics">Statistics</a>
            <a href="#recs">Recommendations</a>
        </div>
        <div class="hud-status">
            <span class="pulse-dot"></span>
            <span>NUMERICAL ANALYZER OK</span>
        </div>
    </header>
    <main>
        <div id="summary" class="grid-4">
            <div class="glass-card">
                <div class="metric-title">Total Numerical Features</div>
                <div class="metric-value">__TOTAL_FEATURES__</div>
                <div class="metric-desc">Count of non-ID/DT metric fields</div>
            </div>
            <div class="glass-card">
                <div class="metric-title">Skewed Features</div>
                <div class="metric-value" style="color: var(--warning);">__SKEWED_COUNT__</div>
                <div class="metric-desc">Features with skewness &gt; 1.0</div>
            </div>
            <div class="glass-card">
                <div class="metric-title">Heavy-Tailed Columns</div>
                <div class="metric-value" style="color: var(--danger);">__HEAVY_TAILED_COUNT__</div>
                <div class="metric-desc">Excess kurtosis &gt; 3.0</div>
            </div>
            <div class="glass-card">
                <div class="metric-title">Outlier Aggregations</div>
                <div class="metric-value">__TOTAL_IQR_OUTLIERS__</div>
                <div class="metric-desc">Total IQR outlier points identified</div>
            </div>
        </div>

        <div id="visuals" class="glass-card" style="margin-bottom: 40px;">
            <div class="sect-title">Detailed Visual Plots (KDE / Histogram / Boxplot / Violin)</div>
            <div class="carousel-tabs">
                __CAROUSEL_NAV__
            </div>
            <div class="carousel-slides">
                __CAROUSEL_SLIDES__
            </div>
        </div>

        <div id="statistics" class="grid-2">
            <div class="glass-card">
                <div class="sect-title">Basic Distribution statistics (Top 20 Features)</div>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Count</th>
                                <th>Mean</th>
                                <th>Median</th>
                                <th>Std Dev</th>
                                <th>Min</th>
                                <th>Max</th>
                            </tr>
                        </thead>
                        <tbody>
                            __STATS_ROWS__
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="glass-card">
                <div class="sect-title">Outlier Ratios Summary (Top 20 Features)</div>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>IQR Cnt</th>
                                <th>IQR %</th>
                                <th>Z-score Cnt</th>
                                <th>Z %</th>
                                <th>P99 Cnt</th>
                                <th>P99 %</th>
                            </tr>
                        </thead>
                        <tbody>
                            __OUTLIER_ROWS__
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="grid-2">
            <div class="glass-card">
                <div class="sect-title">Distribution Shape & Tails (Skewness & Kurtosis)</div>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Skewness</th>
                                <th>Skew Type</th>
                                <th>Kurtosis</th>
                                <th>Kurt Type</th>
                            </tr>
                        </thead>
                        <tbody>
                            __SHAPE_ROWS__
                        </tbody>
                    </table>
                </div>
            </div>

            <div id="recs" class="glass-card recs-box">
                <div class="sect-title">Log Scaling & Transformation Recommendations</div>
                <div style="overflow-x: auto; max-height: 400px;">
                    <table>
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Current Skew</th>
                                <th>Suggested</th>
                                <th>Reasoning</th>
                                <th>Expected Gain</th>
                            </tr>
                        </thead>
                        <tbody>
                            __REC_ROWS__
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="glass-card recs-box" style="margin-top: 20px;">
            <div class="sect-title">Modeling & Translation Strategies summary</div>
            <ul>
                __RECS_LIST__
            </ul>
        </div>
    </main>
    <script>
        if (document.querySelector('.carousel-tabs button')) {
            document.querySelector('.carousel-tabs button').click();
        }
    </script>
</body>
</html>
"""

        # Perform replacement logic on HTML string
        html_content = html_template.replace("__FONT_URL__", font_url)
        html_content = html_content.replace(
            "__TOTAL_FEATURES__", str(summary['total_numerical_features'])
        )
        html_content = html_content.replace(
            "__SKEWED_COUNT__", str(summary['highly_skewed_count'])
        )
        html_content = html_content.replace(
            "__HEAVY_TAILED_COUNT__", str(summary['heavy_tailed_count'])
        )
        html_content = html_content.replace(
            "__TOTAL_IQR_OUTLIERS__",
            f"{summary['outlier_summary']['total_iqr_outliers']:,}"
        )
        html_content = html_content.replace("__CAROUSEL_NAV__", carousel_nav)
        html_content = html_content.replace("__CAROUSEL_SLIDES__", slides)
        html_content = html_content.replace("__STATS_ROWS__", stats_rows)
        html_content = html_content.replace("__OUTLIER_ROWS__", outlier_rows)
        html_content = html_content.replace("__SHAPE_ROWS__", shape_rows)
        html_content = html_content.replace("__REC_ROWS__", rec_rows)
        html_content = html_content.replace("__RECS_LIST__", recs_summary_list)

        report_path = report_dir / "numerical_analysis_report.html"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(
            "Compiled Numerical HTML dashboard to: %s",
            report_path
        )
