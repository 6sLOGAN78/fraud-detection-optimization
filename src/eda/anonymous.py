# ruff: noqa: E501
"""Anonymous Feature Analysis engine — Part 3.11 IEEE-CIS Fraud Detection EDA."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnonymousFeatureAnalyzer:
    """Comprehensive analysis engine for IEEE-CIS Anonymous Features (V, C, D, M series).

    Sub-modules:
    - 3.11.4  Anonymous Feature Inventory
    - 3.11.5  V-Series Analysis (V1-V339)
    - 3.11.6  C-Series Analysis (C1-C14)
    - 3.11.7  D-Series Analysis (D1-D15)
    - 3.11.8  M-Series Analysis (M1-M9)
    - 3.11.9  Distribution Analysis
    - 3.11.10 Missingness Analysis
    - 3.11.11 Feature Importance Analysis (ANOVA / F-score)
    - 3.11.12 Correlation & Redundancy Analysis (Pairwise correlations + clusters)
    - 3.11.13 Anonymous Feature Interaction Analysis
    - 3.11.14 Feature Engineering Opportunities
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_col: str = "isFraud",
    ) -> None:
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.target_col = target_col
        self._global_fraud_rate = float(df_train[target_col].mean()) if target_col in df_train.columns else 0.035

        # Subsample for expensive computations (e.g. correlation matrix / mutual info)
        self.sample_size = min(25000, len(self.df_train))
        self.df_train_sample = self.df_train.sample(n=self.sample_size, random_state=42) if len(self.df_train) > self.sample_size else self.df_train.copy()

        # Identify all anonymous columns present in datasets
        self.v_cols = [c for c in self.df_train.columns if c.startswith("V") and c[1:].isdigit()]
        self.c_cols = [c for c in self.df_train.columns if c.startswith("C") and c[1:].isdigit()]
        self.d_cols = [c for c in self.df_train.columns if c.startswith("D") and c[1:].isdigit()]
        self.m_cols = [c for c in self.df_train.columns if c.startswith("M") and c[1:].isdigit()]
        self.all_anon_cols = self.v_cols + self.c_cols + self.d_cols + self.m_cols

        logger.info(
            "AnonymousFeatureAnalyzer initialized. Columns identified: V: %d, C: %d, D: %d, M: %d. Total: %d",
            len(self.v_cols), len(self.c_cols), len(self.d_cols), len(self.m_cols), len(self.all_anon_cols)
        )

    # ------------------------------------------------------------------ #
    # Orchestrator                                                         #
    # ------------------------------------------------------------------ #

    def analyze_all(self, report_dir: Path) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = report_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("--- Stage 3.11: Anonymous Feature Analysis ---")

        df_inv = self.analyze_anonymous_inventory(report_dir)
        df_v = self.analyze_v_series(report_dir)
        df_c = self.analyze_c_series(report_dir)
        df_d = self.analyze_d_series(report_dir)
        df_m = self.analyze_m_series(report_dir)
        df_dist = self.analyze_distributions(report_dir)
        df_miss = self.analyze_missingness(report_dir)
        df_imp = self.analyze_feature_importance(report_dir)
        df_corr, df_clusters = self.analyze_correlation_redundancy(report_dir)
        df_interact = self.analyze_interactions(report_dir)
        df_recs = self.generate_anonymous_recommendations(report_dir)

        # Generate plots (Top 4 HUD-styled charts)
        self.generate_plots(plots_dir, df_imp)

        summary = self._build_summary(df_inv, df_miss, df_imp, df_clusters)

        self.compile_html_dashboard(
            report_dir=report_dir,
            summary=summary,
            df_inv=df_inv,
            df_v=df_v.head(50),  # Show top 50 in HTML to prevent rendering lag
            df_c=df_c,
            df_d=df_d,
            df_m=df_m,
            df_dist=df_dist.head(50),
            df_miss=df_miss.head(50),
            df_imp=df_imp.head(55),
            df_corr=df_corr.head(50) if df_corr is not None else pd.DataFrame(),
            df_clusters=df_clusters.head(50) if df_clusters is not None else pd.DataFrame(),
            df_interact=df_interact,
            df_recs=df_recs,
        )

        with (report_dir / "anonymous_analysis.json").open("w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Anonymous Feature Analysis complete. Reports → %s", report_dir)

    # ------------------------------------------------------------------ #
    # 3.11.4 Anonymous Feature Inventory                                 #
    # ------------------------------------------------------------------ #

    def analyze_anonymous_inventory(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Building Anonymous Feature Inventory...")
        records = []
        for col in self.all_anon_cols:
            grp_name = "V-Series" if col in self.v_cols else "C-Series" if col in self.c_cols else "D-Series" if col in self.d_cols else "M-Series"
            miss_pct = float(self.df_train[col].isna().mean() * 100)
            card = int(self.df_train[col].nunique(dropna=True))
            dtype = str(self.df_train[col].dtype)
            avail = "High" if miss_pct < 20 else "Medium" if miss_pct < 60 else "Low" if miss_pct < 95 else "None"

            records.append({
                "feature_name": col,
                "feature_group": grp_name,
                "data_type": dtype,
                "missing_percentage": round(miss_pct, 4),
                "unique_values": card,
                "cardinality": "Binary" if card == 2 else "Low" if card <= 10 else "High",
                "availability": avail,
            })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "anonymous_feature_inventory.csv", index=False)

        summary_meta = {
            "total_count": len(self.all_anon_cols),
            "v_count": len(self.v_cols),
            "c_count": len(self.c_cols),
            "d_count": len(self.d_cols),
            "m_count": len(self.m_cols),
        }
        with (report_dir / "anonymous_feature_metadata.json").open("w") as f:
            json.dump(summary_meta, f, indent=2)

        return df

    # ------------------------------------------------------------------ #
    # 3.11.5 V-Series Analysis                                           #
    # ------------------------------------------------------------------ #

    def analyze_v_series(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing V-Series features...")
        if not self.v_cols:
            return pd.DataFrame()

        # Compute stats for all V columns
        desc = self.df_train[self.v_cols].describe().T.reset_index()
        desc.rename(columns={"index": "feature_name"}, inplace=True)

        # Skew, Kurtosis, Outlier %
        skews = self.df_train[self.v_cols].skew().round(4)
        kurt = self.df_train[self.v_cols].kurt().round(4)
        
        outliers_list = []
        for c in self.v_cols:
            col_data = self.df_train[c].dropna()
            if len(col_data) > 0:
                q25, q75 = np.percentile(col_data, [25, 75])
                iqr = q75 - q25
                out_cnt = ((col_data < q25 - 1.5 * iqr) | (col_data > q75 + 1.5 * iqr)).sum()
                outliers_list.append(round(out_cnt / len(col_data) * 100, 4))
            else:
                outliers_list.append(0.0)

        desc["skewness"] = desc["feature_name"].map(skews).fillna(0)
        desc["kurtosis"] = desc["feature_name"].map(kurt).fillna(0)
        desc["outlier_percentage"] = outliers_list

        desc.to_csv(report_dir / "v_feature_analysis.csv", index=False)
        return desc

    # ------------------------------------------------------------------ #
    # 3.11.6 C-Series Analysis                                           #
    # ------------------------------------------------------------------ #

    def analyze_c_series(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing C-Series features...")
        if not self.c_cols:
            return pd.DataFrame()

        desc = self.df_train[self.c_cols].describe().T.reset_index()
        desc.rename(columns={"index": "feature_name"}, inplace=True)

        # Skew, Outlier %
        skews = self.df_train[self.c_cols].skew().round(4)
        
        outliers_list = []
        for c in self.c_cols:
            col_data = self.df_train[c].dropna()
            if len(col_data) > 0:
                q25, q75 = np.percentile(col_data, [25, 75])
                iqr = q75 - q25
                out_cnt = ((col_data < q25 - 1.5 * iqr) | (col_data > q75 + 1.5 * iqr)).sum()
                outliers_list.append(round(out_cnt / len(col_data) * 100, 4))
            else:
                outliers_list.append(0.0)

        desc["skewness"] = desc["feature_name"].map(skews).fillna(0)
        desc["outlier_percentage"] = outliers_list

        # Add Relative Risk (Correlation to Target)
        if self.target_col in self.df_train.columns:
            corrs = self.df_train[self.c_cols].corrwith(self.df_train[self.target_col]).round(4)
            desc["correlation_to_target"] = desc["feature_name"].map(corrs).fillna(0)
        else:
            desc["correlation_to_target"] = 0.0

        desc.to_csv(report_dir / "c_feature_analysis.csv", index=False)
        return desc

    # ------------------------------------------------------------------ #
    # 3.11.7 D-Series Analysis                                           #
    # ------------------------------------------------------------------ #

    def analyze_d_series(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing D-Series features...")
        if not self.d_cols:
            return pd.DataFrame()

        desc = self.df_train[self.d_cols].describe().T.reset_index()
        desc.rename(columns={"index": "feature_name"}, inplace=True)
        
        # Missing percent
        miss = (self.df_train[self.d_cols].isna().mean() * 100).round(4)
        desc["missing_percentage"] = desc["feature_name"].map(miss).fillna(0)

        if self.target_col in self.df_train.columns:
            corrs = self.df_train[self.d_cols].corrwith(self.df_train[self.target_col]).round(4)
            desc["correlation_to_target"] = desc["feature_name"].map(corrs).fillna(0)
        else:
            desc["correlation_to_target"] = 0.0

        desc.to_csv(report_dir / "d_feature_analysis.csv", index=False)
        return desc

    # ------------------------------------------------------------------ #
    # 3.11.8 M-Series Analysis                                           #
    # ------------------------------------------------------------------ #

    def analyze_m_series(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing M-Series features...")
        if not self.m_cols:
            return pd.DataFrame()

        records = []
        has_target = self.target_col in self.df_train.columns

        for col in self.m_cols:
            # Dropna category counts
            val_counts = self.df_train[col].astype(str).value_counts(dropna=False)
            missing_count = int(self.df_train[col].isna().sum())
            missing_pct = round(missing_count / len(self.df_train) * 100, 4)

            # Let's count high-risk categories (fraud rate > 1.5 * global rate)
            if has_target:
                grp = self.df_train.groupby(col, dropna=False)[self.target_col].agg(["count", "mean"])
                grp["relative_risk"] = grp["mean"] / self._global_fraud_rate
                max_risk_row = grp.loc[grp["relative_risk"].idxmax()] if not grp.empty else None
                max_risk_val = str(grp["relative_risk"].idxmax()) if not grp.empty else "N/A"
                max_risk = float(max_risk_row["relative_risk"]) if max_risk_row is not None else 1.0
                max_fraud_rate = float(max_risk_row["mean"] * 100) if max_risk_row is not None else 0.0
            else:
                max_risk_val = "N/A"
                max_risk = 1.0
                max_fraud_rate = 0.0

            records.append({
                "feature_name": col,
                "missing_percentage": missing_pct,
                "cardinality": len(val_counts),
                "highest_risk_value": max_risk_val,
                "max_relative_risk": round(max_risk, 4),
                "max_fraud_rate_pct": round(max_fraud_rate, 4),
            })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "m_feature_analysis.csv", index=False)
        return df

    # ------------------------------------------------------------------ #
    # 3.11.9 Distribution Analysis                                       #
    # ------------------------------------------------------------------ #

    def analyze_distributions(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing shape distributions...")
        records = []
        all_numeric = self.v_cols + self.c_cols + self.d_cols
        
        skews = self.df_train[all_numeric].skew().round(4)
        for col in all_numeric:
            skew_val = skews[col]
            if abs(skew_val) > 2.0:
                shape = "Highly Skewed"
            elif abs(skew_val) > 0.5:
                shape = "Moderately Skewed"
            else:
                shape = "Symmetric"

            unique = self.df_train[col].nunique()
            if unique <= 1:
                shape = "Near-Constant"

            records.append({
                "feature_name": col,
                "skewness": skew_val,
                "distribution_shape": shape,
                "unique_values": unique,
            })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "anonymous_distribution_analysis.csv", index=False)
        return df

    # ------------------------------------------------------------------ #
    # 3.11.10 Missingness Analysis                                       #
    # ------------------------------------------------------------------ #

    def analyze_missingness(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing missingness behavior...")
        records = []
        has_target = self.target_col in self.df_train.columns

        for col in self.all_anon_cols:
            miss_count = int(self.df_train[col].isna().sum())
            miss_pct = miss_count / len(self.df_train) * 100

            classification = "Complete" if miss_pct < 1 else "Low Missing" if miss_pct < 20 else "Moderate Missing" if miss_pct < 60 else "High Missing" if miss_pct < 95 else "Extremely Sparse"

            if has_target and miss_count > 0 and miss_count < len(self.df_train):
                # Fraud rate of missing vs non-missing
                is_missing = self.df_train[col].isna()
                fr_missing = self.df_train.loc[is_missing, self.target_col].mean()
                fr_present = self.df_train.loc[~is_missing, self.target_col].mean()
                risk_ratio = fr_missing / fr_present if fr_present > 0 else 1.0
            else:
                risk_ratio = 1.0

            records.append({
                "feature_name": col,
                "missing_count": miss_count,
                "missing_percentage": round(miss_pct, 4),
                "missingness_classification": classification,
                "informative_missingness_ratio": round(risk_ratio, 4),
            })

        df = pd.DataFrame(records)
        df.sort_values(by="missing_percentage", ascending=False, inplace=True)
        df.to_csv(report_dir / "anonymous_missingness_analysis.csv", index=False)
        return df

    # ------------------------------------------------------------------ #
    # 3.11.11 Feature Importance Analysis                                #
    # ------------------------------------------------------------------ #

    def analyze_feature_importance(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing predictive feature importance using ANOVA F-value...")
        if self.target_col not in self.df_train.columns:
            return pd.DataFrame()

        num_cols = self.v_cols + self.c_cols + self.d_cols
        
        # Vectorized ANOVA F-test
        # Fill missing values with median to make it robust for ANOVA
        X = self.df_train_sample[num_cols].copy()
        for col in num_cols:
            mean_val = X[col].mean()
            X[col] = X[col].fillna(mean_val if pd.notna(mean_val) else 0)

        y = self.df_train_sample[self.target_col]

        f_vals, p_vals = f_classif(X, y)

        records = []
        for i, col in enumerate(num_cols):
            f_val = f_vals[i]
            p_val = p_vals[i]
            records.append({
                "feature_name": col,
                "anova_f_value": float(f_val) if np.isfinite(f_val) else 0.0,
                "p_value": float(p_val) if np.isfinite(p_val) else 1.0,
            })

        df = pd.DataFrame(records)
        df["predictive_strength"] = np.where(df["p_value"] < 0.01, "Strong", "Weak")
        df.sort_values(by="anova_f_value", ascending=False, inplace=True)
        df.to_csv(report_dir / "anonymous_feature_importance.csv", index=False)
        return df

    # ------------------------------------------------------------------ #
    # 3.11.12 Correlation & Redundancy Analysis                         #
    # ------------------------------------------------------------------ #

    def analyze_correlation_redundancy(self, report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Computing Spearman Correlation and finding clusters...")
        # Since calculating all pairs on 339 V features takes time, we select the top 20 V cols,
        # plus all C and D columns (total ~50 columns) for detailed pairwise evaluation.
        top_v = self.v_cols[:20]
        selected_cols = top_v + self.c_cols + self.d_cols
        
        X = self.df_train_sample[selected_cols].dropna(how="all")
        corr_matrix = X.corr(method="spearman").fillna(0)

        # Convert matrix to flat dataframe
        flat_records = []
        for i in range(len(selected_cols)):
            for j in range(i + 1, len(selected_cols)):
                c1 = selected_cols[i]
                c2 = selected_cols[j]
                val = corr_matrix.loc[c1, c2]
                flat_records.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "spearman_correlation": round(float(val), 4),
                })
        
        df_corr = pd.DataFrame(flat_records)
        df_corr.sort_values(by="spearman_correlation", key=abs, ascending=False, inplace=True)
        df_corr.to_csv(report_dir / "anonymous_correlation_analysis.csv", index=False)

        # Basic clustering of redundant pairs (corr > 0.90)
        clusters: list[set[str]] = []
        redundant_pairs = df_corr[df_corr["spearman_correlation"].abs() > 0.90]

        for _, row in redundant_pairs.iterrows():
            f1, f2 = row["feature_1"], row["feature_2"]
            placed = False
            for cluster in clusters:
                if f1 in cluster or f2 in cluster:
                    cluster.update([f1, f2])
                    placed = True
                    break
            if not placed:
                clusters.append({f1, f2})

        cluster_records = []
        for idx, cluster in enumerate(clusters):
            for feat in cluster:
                cluster_records.append({
                    "cluster_id": idx + 1,
                    "feature_name": feat,
                })

        df_clusters = pd.DataFrame(cluster_records)
        df_clusters.to_csv(report_dir / "feature_clusters.csv", index=False)

        return df_corr, df_clusters

    # ------------------------------------------------------------------ #
    # 3.11.13 Anonymous Feature Interaction Analysis                      #
    # ------------------------------------------------------------------ #

    def analyze_interactions(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Computing interaction profiles...")
        if self.target_col not in self.df_train.columns:
            return pd.DataFrame()

        # Let's compute a few critical cross-group interactions (e.g. M4 x C1, M4 x D1)
        interactions = [
            ("M4", "C1"),
            ("M4", "C13"),
            ("M4", "D1"),
            ("M4", "D15"),
        ]

        records = []
        for col1, col2 in interactions:
            if col1 in self.df_train.columns and col2 in self.df_train.columns:
                # Group by both, compute count and fraud rate
                # Bin numeric column into quantiles if cardinality is high (e.g. C1 / D1)
                df_temp = self.df_train[[col1, col2, self.target_col]].copy()
                df_temp[col2] = pd.qcut(df_temp[col2], q=4, duplicates="drop").astype(str)

                grp = df_temp.groupby([col1, col2]).agg(
                    count=(self.target_col, "count"),
                    fraud_rate=(self.target_col, "mean"),
                ).reset_index()

                # Get max fraud rate interaction segment
                max_idx = grp["fraud_rate"].idxmax() if not grp.empty else None
                if max_idx is not None:
                    max_row = grp.loc[max_idx]
                    records.append({
                        "interaction_pair": f"{col1} × {col2}",
                        "peak_segment": f"{col1}={max_row[col1]} | {col2}={max_row[col2]}",
                        "segment_count": int(max_row["count"]),
                        "segment_fraud_rate_pct": round(float(max_row["fraud_rate"] * 100), 4),
                        "relative_risk": round(float(max_row["fraud_rate"] / self._global_fraud_rate), 4),
                    })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "anonymous_feature_interactions.csv", index=False)
        return df

    # ------------------------------------------------------------------ #
    # 3.11.14 Feature Recommendation                                      #
    # ------------------------------------------------------------------ #

    def generate_anonymous_recommendations(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Generating feature engineering recommendations...")
        records = [
            {"category": "Distribution-Based", "group": "C-Series", "recommendation": "Log1p Transformation", "rationale": "High right-skew/heavy tail behavior maps counts to logistic scale."},
            {"category": "Missingness Features", "group": "D-Series", "recommendation": "Missing Indicator Flags", "rationale": "D features carry significant predictive value in their presence/absence patterns."},
            {"category": "Statistical Features", "group": "V-Series", "recommendation": "PCA / Autoencoders", "rationale": "High dimensionality within V1-V339 requires projection to prevent overfitting."},
            {"category": "Interaction Features", "group": "C & M Series", "recommendation": "Ratios & Products", "rationale": "Cross-interaction metrics capture velocity matching abnormalities."},
        ]
        df = pd.DataFrame(records)
        df.to_csv(report_dir / "anonymous_feature_recommendations.csv", index=False)
        return df

    # ------------------------------------------------------------------ #
    # Core Plots                                                         #
    # ------------------------------------------------------------------ #

    def generate_plots(self, plots_dir: Path, df_imp: pd.DataFrame) -> None:
        logger.info("Generating plot assets...")
        plt.style.use("dark_background")
        _bg = "#06070b"
        _fg = "#8e97a4"
        _red = "#d63031"

        def _save(fig: plt.Figure, name: str) -> None:
            fig.savefig(plots_dir / name, dpi=110, bbox_inches="tight", facecolor=_bg)
            plt.close(fig)

        # 1. Distribution of V-series missingness
        if self.v_cols:
            try:
                miss_counts = self.df_train[self.v_cols].isna().mean() * 100
                fig, ax = plt.subplots(figsize=(8, 4), facecolor=_bg)
                ax.set_facecolor(_bg)
                ax.hist(miss_counts, bins=20, color=_fg, edgecolor="#ffffff11", alpha=0.8)
                ax.set_title("[V-SERIES] Distribution of Missingness (%)", color="#fff", fontsize=10)
                ax.set_xlabel("Missing Percentage", color=_fg)
                ax.tick_params(colors=_fg)
                ax.spines[:].set_color("#ffffff11")
                plt.tight_layout()
                _save(fig, "v_missingness_distribution.png")
            except Exception as exc:
                logger.warning("V missingness plot failed: %s", exc)

        # 2. Daily/hourly distributions of values in C1-C14
        if self.c_cols:
            try:
                fig, ax = plt.subplots(figsize=(8, 4), facecolor=_bg)
                ax.set_facecolor(_bg)
                self.df_train_sample[self.c_cols[:7]].boxplot(ax=ax, color=dict(boxes=_fg, whiskers=_fg, medians="#fff", caps=_fg), patch_artist=False)
                ax.set_title("[C-SERIES] Value distributions (C1 - C7)", color="#fff", fontsize=10)
                ax.tick_params(colors=_fg)
                ax.spines[:].set_color("#ffffff11")
                plt.tight_layout()
                _save(fig, "c_distributions.png")
            except Exception as exc:
                logger.warning("C distribution plot failed: %s", exc)

        # 3. D-series missingness vs fraud rate
        if self.d_cols and self.target_col in self.df_train.columns:
            try:
                # Plot D1 and D15 scatter/bar distributions dynamically
                fig, ax = plt.subplots(figsize=(8, 4), facecolor=_bg)
                ax.set_facecolor(_bg)
                d1_col = "D1" if "D1" in self.df_train_sample.columns else (self.d_cols[0] if self.d_cols else None)
                d2_col = "D15" if "D15" in self.df_train_sample.columns else (self.d_cols[-1] if self.d_cols else None)
                if d1_col and d2_col:
                    d_sample = self.df_train_sample[[d1_col, d2_col]].dropna()
                    ax.scatter(d_sample[d1_col], d_sample[d2_col], color=_fg, alpha=0.3, edgecolors="none")
                    ax.set_title(f"[D-SERIES] Temporal/Interval Interaction ({d1_col} vs {d2_col})", color="#fff", fontsize=10)
                    ax.set_xlabel(d1_col, color=_fg)
                    ax.set_ylabel(d2_col, color=_fg)
                    ax.tick_params(colors=_fg)
                    ax.spines[:].set_color("#ffffff11")
                    plt.tight_layout()
                    _save(fig, "d_interaction.png")
            except Exception as exc:
                logger.warning("D temporal plot failed: %s", exc)

        # 4. Feature importance rankings (Top 15)
        if not df_imp.empty:
            try:
                top_15 = df_imp.head(15)
                fig, ax = plt.subplots(figsize=(8, 4), facecolor=_bg)
                ax.set_facecolor(_bg)
                ax.barh(top_15["feature_name"][::-1], top_15["anova_f_value"][::-1], color=_fg, edgecolor="#ffffff11")
                ax.set_title("[IMPORTANCE] Top 15 Anonymous Features (ANOVA F-value)", color="#fff", fontsize=10)
                ax.tick_params(colors=_fg)
                ax.spines[:].set_color("#ffffff11")
                plt.tight_layout()
                _save(fig, "importance_ranking.png")
            except Exception as exc:
                logger.warning("Importance ranking plot failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Summary Builder                                                    #
    # ------------------------------------------------------------------ #

    def _build_summary(
        self,
        df_inv: pd.DataFrame,
        df_miss: pd.DataFrame,
        df_imp: pd.DataFrame,
        df_clusters: pd.DataFrame,
    ) -> dict[str, Any]:
        
        # Sparse features count (> 80% missing)
        sparse_cnt = 0
        if not df_miss.empty and "missing_percentage" in df_miss.columns:
            sparse_cnt = int((df_miss["missing_percentage"] > 80.0).sum())

        # Top predictive feature
        top_feat = "N/A"
        top_f_val = 0.0
        if not df_imp.empty and "feature_name" in df_imp.columns:
            top_feat = str(df_imp.iloc[0]["feature_name"])
            top_f_val = float(df_imp.iloc[0]["anova_f_value"])

        # Redundant clusters count
        cluster_cnt = 0
        if not df_clusters.empty and "cluster_id" in df_clusters.columns:
            cluster_cnt = int(df_clusters["cluster_id"].nunique())

        total_anon = len(self.all_anon_cols)

        return {
            "total_anonymous_features": total_anon,
            "sparse_anonymous_features": sparse_cnt,
            "top_predictive_feature": top_feat,
            "top_predictive_f_value": round(top_f_val, 4),
            "redundant_clusters_detected": cluster_cnt,
        }

    # ------------------------------------------------------------------ #
    # HTML compilation                                                   #
    # ------------------------------------------------------------------ #

    def compile_html_dashboard(
        self,
        report_dir: Path,
        summary: dict[str, Any],
        df_inv: pd.DataFrame,
        df_v: pd.DataFrame,
        df_c: pd.DataFrame,
        df_d: pd.DataFrame,
        df_m: pd.DataFrame,
        df_dist: pd.DataFrame,
        df_miss: pd.DataFrame,
        df_imp: pd.DataFrame,
        df_corr: pd.DataFrame,
        df_clusters: pd.DataFrame,
        df_interact: pd.DataFrame,
        df_recs: pd.DataFrame,
    ) -> None:
        logger.info("Compiling Anonymous Features HTML Dashboard...")

        def _to_html(df: pd.DataFrame) -> str:
            if df.empty:
                return "<div class='no-data'>NO COMPATIBLE DATA FOUND</div>"
            return df.to_html(
                classes="hud-table",
                index=False,
                border=0,
                justify="left",
            )

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>IEEE-CIS ANONYMOUS MODULE DIAGNOSTICS</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Orbitron:wght@500;700;900&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #06070b;
            --panel-bg: rgba(14, 16, 22, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-color: #8e97a4;
            --text-white: #ffffff;
            --accent-glow: #ffffff;
            --alert-red: #ff3838;
            --alert-green: #2ed573;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'JetBrains Mono', monospace;
            padding: 2.5rem;
            position: relative;
            min-height: 100vh;
            overflow-x: hidden;
        }}

        /* Grid Overlay & Scanlines */
        body::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background-image: 
                linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
            background-size: 32px 32px;
            pointer-events: none;
            z-index: 1;
        }}

        body::after {{
            content: " ";
            display: block;
            position: fixed;
            top: 0; left: 0; bottom: 0; right: 0;
            background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
            z-index: 9999;
            background-size: 100% 4px, 6px 100%;
            pointer-events: none;
            opacity: 0.45;
        }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 1.5rem;
            margin-bottom: 2.5rem;
            position: relative;
            z-index: 10;
        }}

        h1 {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.8rem;
            letter-spacing: 2px;
            color: var(--text-white);
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
        }}

        .status-pill {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.8rem;
            padding: 0.4rem 1rem;
            border: 1px solid var(--alert-green);
            background-color: rgba(46, 213, 115, 0.1);
            color: var(--alert-green);
            border-radius: 4px;
            text-shadow: 0 0 5px rgba(46, 213, 115, 0.3);
        }}

        /* Key Metrics Grid */
        .hud-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin-bottom: 2.5rem;
            position: relative;
            z-index: 10;
        }}

        .hud-panel {{
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1.5rem;
            backdrop-filter: blur(16px) saturate(120%);
        }}

        .metric-label {{
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
            color: var(--text-color);
        }}

        .metric-value {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-white);
        }}

        .metric-val-red {{
            color: var(--alert-red);
        }}

        /* Main Dashboard Sections */
        .dashboard-body {{
            display: grid;
            grid-template-columns: 1.25fr 1fr;
            gap: 2rem;
            margin-bottom: 2.5rem;
            position: relative;
            z-index: 10;
        }}

        .visualizer-card {{
            display: flex;
            flex-direction: column;
            height: 100%;
        }}

        .carousel-tabs {{
            display: flex;
            gap: 0.25rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }}

        .carousel-tab {{
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-color);
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem;
            padding: 0.5rem 0.75rem;
            cursor: pointer;
            border-radius: 4px;
            transition: all 0.2s ease;
        }}

        .carousel-tab:hover {{
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-white);
        }}

        .carousel-tab.active {{
            background: #fff;
            color: var(--bg-color);
            border-color: #fff;
            text-shadow: none;
        }}

        .carousel-content {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 1rem;
        }}

        .carousel-img {{
            max-width: 100%;
            max-height: 380px;
            height: auto;
            border-radius: 2px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}

        .hud-table-wrapper {{
            overflow-x: auto;
            max-height: 485px;
        }}

        .hud-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.75rem;
        }}

        .hud-table th {{
            font-family: 'Orbitron', sans-serif;
            background-color: rgba(255, 255, 255, 0.05);
            color: var(--text-white);
            text-align: left;
            padding: 0.75rem;
            font-size: 0.7rem;
            letter-spacing: 1px;
            border: 1px solid var(--border-color);
        }}

        .hud-table td {{
            padding: 0.75rem;
            border: 1px solid var(--border-color);
        }}

        .hud-table tr:hover {{
            background-color: rgba(255, 255, 255, 0.02);
            color: #fff;
        }}

        .secondary-panel-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            margin-bottom: 2.5rem;
            position: relative;
            z-index: 10;
        }}

        h2 {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1rem;
            letter-spacing: 1px;
            color: var(--text-white);
            margin-bottom: 1.25rem;
            border-left: 3px solid var(--text-white);
            padding-left: 0.75rem;
        }}

        .tag-pill {{
            display: inline-block;
            font-size: 0.65rem;
            padding: 0.15rem 0.4rem;
            border-radius: 3px;
            background-color: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border-color);
            margin-right: 0.25rem;
        }}

        .tag-pill-high {{
            background-color: rgba(214, 48, 49, 0.15);
            border-color: rgba(214, 48, 49, 0.3);
            color: #ff7675;
        }}
    </style>
</head>
<body>

    <header>
        <div>
            <h1>IEEE-CIS ANONYMOUS MODULE DIAGNOSTICS</h1>
            <p style="font-size: 0.65rem; color: var(--text-color); margin-top: 0.25rem; letter-spacing: 1px;">STAGE 3.11: SYSTEM ANONYMOUS PROPERTIES</p>
        </div>
        <div class="status-pill">ANONYMOUS ANALYZER OK</div>
    </header>

    <div class="hud-grid">
        <div class="hud-panel">
            <p class="metric-label">Anonymous Features</p>
            <p class="metric-value">{summary['total_anonymous_features']:,}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Sparse Fields (&gt;80%)</p>
            <p class="metric-value">{summary['sparse_anonymous_features']}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Top Predictor</p>
            <p class="metric-value">{summary['top_predictive_feature']}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Redundant Clusters</p>
            <p class="metric-value metric-val-red">{summary['redundant_clusters_detected']}</p>
        </div>
    </div>

    <div class="dashboard-body">
        <div class="hud-panel visualizer-card">
            <h2>DIAGNOSTIC VISUALIZATIONS</h2>
            <div class="carousel-tabs">
                <button class="carousel-tab active" onclick="switchTab(0)">Missingness (V)</button>
                <button class="carousel-tab" onclick="switchTab(1)">C Distributions</button>
                <button class="carousel-tab" onclick="switchTab(2)">D Timeline</button>
                <button class="carousel-tab" onclick="switchTab(3)">Feature Importance</button>
            </div>
            <div class="carousel-content">
                <img id="carousel-img" class="carousel-img" src="plots/v_missingness_distribution.png" alt="Missingness Plot">
            </div>
        </div>

        <div class="hud-panel">
            <h2>FEATURE IMPORTANCE RANKINGS (ANOVA)</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_imp)}
            </div>
        </div>
    </div>

    <div class="secondary-panel-grid">
        <div class="hud-panel" style="grid-column: span 2;">
            <h2>FEATURE ENGINEERING RECOMMENDATIONS</h2>
            <table class="hud-table">
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Feature Group</th>
                        <th>Proposed Transformation</th>
                        <th>System Rationale</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f"<tr><td><span class='tag-pill tag-pill-high'>{row['category']}</span></td><td><strong>{row['group']}</strong></td><td><code>{row['recommendation']}</code></td><td>{row['rationale']}</td></tr>" for idx, row in df_recs.iterrows())}
                </tbody>
            </table>
        </div>
    </div>

    <div class="secondary-panel-grid">
        <div class="hud-panel">
            <h2>MISSINGNESS DETECTORS</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_miss)}
            </div>
        </div>
        <div class="hud-panel">
            <h2>REDUNDANCY CLUSTERING (CORRELATION &gt; 0.90)</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_clusters)}
            </div>
        </div>
    </div>

    <div class="secondary-panel-grid">
        <div class="hud-panel">
            <h2>D-SERIES METRICS</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_d)}
            </div>
        </div>
        <div class="hud-panel">
            <h2>M-SERIES METRICS</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_m)}
            </div>
        </div>
    </div>

    <div class="secondary-panel-grid">
        <div class="hud-panel" style="grid-column: span 2;">
            <h2>CROSS-GROUP INTERACTIONS</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_interact)}
            </div>
        </div>
    </div>

    <script>
        const imagePaths = [
            "plots/v_missingness_distribution.png",
            "plots/c_distributions.png",
            "plots/d_interaction.png",
            "plots/importance_ranking.png"
        ];

        function switchTab(index) {{
            const tabs = document.querySelectorAll(".carousel-tab");
            const img = document.getElementById("carousel-img");
            
            tabs.forEach((tab, idx) => {{
                if (idx === index) {{
                    tab.classList.add("active");
                }} else {{
                    tab.classList.remove("active");
                }}
            }});

            img.src = imagePaths[index];
        }}
    </script>
</body>
</html>
"""

        with (report_dir / "anonymous_analysis_report.html").open("w") as f:
            f.write(html_template)
        logger.info("Compiled Anonymous HTML dashboard saved to: %s", report_dir / "anonymous_analysis_report.html")
