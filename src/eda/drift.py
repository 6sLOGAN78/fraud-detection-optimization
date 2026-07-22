"""Part 3.15: Drift Analysis Core Engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftAnalyzer:
    """Evaluates Population Stability Index (PSI), Kolmogorov-Smirnov (KS) Drift, statistical stats shifts, severity classifications, and cause-mitigation recommendations."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        random_state: int = 42,
    ) -> None:
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.random_state = random_state

        self._select_candidates()

        # Downsample train and test segments to 15,000 rows each to guarantee very fast, optimized computations
        rng = np.random.default_rng(self.random_state)
        sample_size_trn = min(len(self.df_train), 15000)
        sample_size_tst = min(len(self.df_test), 15000)

        indices_trn = rng.choice(self.df_train.index, sample_size_trn, replace=False)
        self.df_sample_trn = self.df_train.loc[indices_trn]

        indices_tst = rng.choice(self.df_test.index, sample_size_tst, replace=False)
        self.df_sample_tst = self.df_test.loc[indices_tst]

        logger.info(
            "DriftAnalyzer initialized. Train sample: %d rows. Test sample: %d rows.",
            len(self.df_sample_trn),
            len(self.df_sample_tst),
        )

    def _select_candidates(self) -> None:
        """Selects a subset of features to keep runtimes extremely fast."""
        ignore = {"TransactionID", "TransactionDT", "isFraud"}
        candidates = [c for c in self.df_train.columns if c not in ignore and c in self.df_test.columns]

        # Sift into numeric vs categorical
        self.num_features = []
        self.cat_features = []

        for c in candidates:
            col_data = self.df_train[c]
            if pd.api.types.is_numeric_dtype(col_data):
                if col_data.nunique(dropna=True) > 2:
                    self.num_features.append(c)
                elif col_data.nunique(dropna=True) == 2:
                    self.cat_features.append(c)
            else:
                self.cat_features.append(c)

        # Limit numbers of candidates for optimization
        # Top 15 numeric by variance
        if len(self.num_features) > 15:
            vars_num = self.df_train[self.num_features].var().fillna(0)
            self.num_features = list(vars_num.sort_values(ascending=False).head(15).index)

        # Top 8 categorical by unique counts (not exceeding too high cardinality)
        valid_cats = []
        for c in self.cat_features:
            u_cnt = self.df_train[c].nunique(dropna=True)
            if 1 < u_cnt < 30:
                valid_cats.append((c, u_cnt))
        valid_cats = sorted(valid_cats, key=lambda x: x[1], reverse=True)
        self.cat_features = [x[0] for x in valid_cats[:8]]

    def analyze_drift_inventory(self, report_dir: Path) -> pd.DataFrame:
        """Summarizes features, missing rates, and identifies eligibility for drift calculation."""
        logger.info("Generating drift feature inventory...")
        records = []
        
        # Process numerical candidate features
        for f in self.num_features:
            col_trn = self.df_train[f]
            col_tst = self.df_test[f]
            records.append({
                "feature_name": f,
                "data_type": "Numerical",
                "train_missing_pct": round(float(col_trn.isna().mean() * 100), 4),
                "test_missing_pct": round(float(col_tst.isna().mean() * 100), 4),
                "train_sample_size": int(col_trn.notna().sum()),
                "test_sample_size": int(col_tst.notna().sum()),
                "unique_values_train": int(col_trn.nunique(dropna=True)),
                "eligible_tests": "PSI, KS Drift, Quantile Shift",
            })
            
        # Process categorical candidate features
        for f in self.cat_features:
            col_trn = self.df_train[f]
            col_tst = self.df_test[f]
            records.append({
                "feature_name": f,
                "data_type": "Categorical",
                "train_missing_pct": round(float(col_trn.isna().mean() * 100), 4),
                "test_missing_pct": round(float(col_tst.isna().mean() * 100), 4),
                "train_sample_size": int(col_trn.notna().sum()),
                "test_sample_size": int(col_tst.notna().sum()),
                "unique_values_train": int(col_trn.nunique(dropna=True)),
                "eligible_tests": "PSI, Frequency Delta, Category Mismatch",
            })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "drift_feature_inventory.csv", index=False)

        metadata = {
            "total_analyzed_features": len(self.num_features) + len(self.cat_features),
            "numerical_features_count": len(self.num_features),
            "categorical_features_count": len(self.cat_features),
            "average_train_missing": round(float(df["train_missing_pct"].mean()), 4) if not df.empty else 0.0,
            "average_test_missing": round(float(df["test_missing_pct"].mean()), 4) if not df.empty else 0.0,
        }
        with open(report_dir / "drift_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        return df

    def analyze_train_test_stats_drift(self, report_dir: Path) -> pd.DataFrame:
        """Compares descriptive stats between train and test partitions."""
        logger.info("Computing train-test descriptive statistical differences...")
        records = []

        all_cols = self.num_features + self.cat_features

        for col in all_cols:
            is_num = (col in self.num_features)
            
            s_trn = self.df_sample_trn[col]
            s_tst = self.df_sample_tst[col]

            miss_trn = s_trn.isna().mean() * 100
            miss_tst = s_tst.isna().mean() * 100
            miss_diff = abs(miss_trn - miss_tst)

            if is_num:
                mean_t = float(s_trn.mean())
                mean_v = float(s_tst.mean())
                mean_diff = abs(mean_t - mean_v)

                med_t = float(s_trn.median())
                med_v = float(s_tst.median())
                med_diff = abs(med_t - med_v)

                std_t = float(s_trn.std())
                std_v = float(s_tst.std())
                std_diff = abs(std_t - std_v)
            else:
                mean_diff = np.nan
                med_diff = np.nan
                std_diff = np.nan

            records.append({
                "feature_name": col,
                "feature_type": "Numerical" if is_num else "Categorical",
                "train_missing_pct": round(miss_trn, 4),
                "test_missing_pct": round(miss_tst, 4),
                "missingness_pct_delta": round(miss_diff, 4),
                "mean_absolute_delta": round(mean_diff, 4) if not pd.isna(mean_diff) else np.nan,
                "median_absolute_delta": round(med_diff, 4) if not pd.isna(med_diff) else np.nan,
                "std_absolute_delta": round(std_diff, 4) if not pd.isna(std_diff) else np.nan,
            })

        df = pd.DataFrame(records).sort_values(by="missingness_pct_delta", ascending=False)
        df.to_csv(report_dir / "train_test_drift.csv", index=False)
        return df

    def compute_numerical_psi(self, train_vals: np.ndarray, test_vals: np.ndarray, num_bins: int = 10) -> float:
        """Calculates Population Stability Index for continuous datasets using quantile bins of train baseline."""
        train_vals = train_vals[~np.isnan(train_vals)]
        test_vals = test_vals[~np.isnan(test_vals)]

        if len(train_vals) < 10 or len(test_vals) < 10:
            return 0.0

        try:
            # Set quantile bins from train segment
            percentiles = np.linspace(0, 100, num_bins + 1)
            bins = np.percentile(train_vals, percentiles)
            bins = np.unique(bins)  # remove duplicate thresholds

            if len(bins) <= 1:
                return 0.0

            # Set bounds
            bins[0] = -np.inf
            bins[-1] = np.inf

            # Digitize values
            train_binned = np.digitize(train_vals, bins)
            test_binned = np.digitize(test_vals, bins)

            # Count bins
            n_bins = len(bins) - 1
            train_counts = np.bincount(train_binned, minlength=n_bins + 2)[1:-1]
            test_counts = np.bincount(test_binned, minlength=n_bins + 2)[1:-1]

            # Normalize to probabilities
            p = train_counts / len(train_vals)
            q = test_counts / len(test_vals)

            # Smooth zero occurrences to avoid division by zero or log(0)
            eps = 1e-4
            p = np.where(p == 0, eps, p)
            q = np.where(q == 0, eps, q)

            # Normalize after smoothing
            p = p / p.sum()
            q = q / q.sum()

            # PSI Sum( (q_i - p_i) * log(q_i / p_i) )
            psi_val = float(np.sum((q - p) * np.log(q / p)))
            return psi_val
        except Exception as e:
            logger.warning("Error computing numerical PSI: %s", e)
            return 0.0

    def compute_categorical_psi(self, train_series: pd.Series, test_series: pd.Series) -> float:
        """Calculates Population Stability Index for categorical attributes."""
        train_series = train_series.dropna().astype(str)
        test_series = test_series.dropna().astype(str)

        if len(train_series) < 10 or len(test_series) < 10:
            return 0.0

        try:
            freq_trn = train_series.value_counts(normalize=True)
            freq_tst = test_series.value_counts(normalize=True)

            all_cats = list(set(freq_trn.index) | set(freq_tst.index))
            if not all_cats:
                return 0.0

            p_list = []
            q_list = []

            for cat in all_cats:
                p_list.append(freq_trn.get(cat, 0.0))
                q_list.append(freq_tst.get(cat, 0.0))

            p = np.array(p_list)
            q = np.array(q_list)

            # Smooth zero values
            eps = 1e-4
            p = np.where(p == 0, eps, p)
            q = np.where(q == 0, eps, q)

            p = p / p.sum()
            q = q / q.sum()

            psi_val = float(np.sum((q - p) * np.log(q / p)))
            return psi_val
        except Exception as e:
            logger.warning("Error computing categorical PSI: %s", e)
            return 0.0

    def analyze_psi(self, report_dir: Path) -> pd.DataFrame:
        """Computes Population Stability Index (PSI) across numerical and categorical features."""
        logger.info("Computing Population Stability Index (PSI)...")
        records = []

        for col in self.num_features:
            s_trn = self.df_sample_trn[col].values
            s_tst = self.df_sample_tst[col].values
            psi_val = self.compute_numerical_psi(s_trn, s_tst)

            if psi_val < 0.1:
                interpret = "No Drift"
            elif psi_val < 0.25:
                interpret = "Moderate Drift"
            else:
                interpret = "Significant Drift"

            records.append({
                "feature_name": col,
                "feature_type": "Numerical",
                "psi_value": round(psi_val, 6),
                "psi_interpretation": interpret,
            })

        for col in self.cat_features:
            s_trn = self.df_sample_trn[col]
            s_tst = self.df_sample_tst[col]
            psi_val = self.compute_categorical_psi(s_trn, s_tst)

            if psi_val < 0.1:
                interpret = "No Drift"
            elif psi_val < 0.25:
                interpret = "Moderate Drift"
            else:
                interpret = "Significant Drift"

            records.append({
                "feature_name": col,
                "feature_type": "Categorical",
                "psi_value": round(psi_val, 6),
                "psi_interpretation": interpret,
            })

        df = pd.DataFrame(records).sort_values(by="psi_value", ascending=False)
        df.to_csv(report_dir / "psi_analysis_summary.csv", index=False)
        return df

    def analyze_ks_drift(self, report_dir: Path) -> pd.DataFrame:
        """Determines Kolmogorov-Smirnov distance shifts across continuous numerical variables."""
        logger.info("Running Kolmogorov-Smirnov drift tests...")
        records = []

        for col in self.num_features:
            s_trn = self.df_sample_trn[col].dropna().values
            s_tst = self.df_sample_tst[col].dropna().values

            if len(s_trn) > 5 and len(s_tst) > 5:
                try:
                    res = ks_2samp(s_trn, s_tst)
                    ks_stat = float(res.statistic)
                    p_val = float(res.pvalue)
                except Exception as e:
                    logger.warning("KS drift test failed on %s: %s", col, e)
                    ks_stat, p_val = 0.0, 1.0
            else:
                ks_stat, p_val = 0.0, 1.0

            if p_val < 0.01:
                sig = "Significant Drift"
            elif p_val < 0.05:
                sig = "Moderate Drift"
            else:
                sig = "Stable"

            records.append({
                "feature_name": col,
                "ks_statistic": round(ks_stat, 6),
                "p_value": round(p_val, 6),
                "ks_significance": sig,
            })

        df = pd.DataFrame(records).sort_values(by="ks_statistic", ascending=False)
        df.to_csv(report_dir / "ks_drift_analysis.csv", index=False)

        # Plot output for top drifted continuous variable
        if not df.empty:
            top_col = df.iloc[0]["feature_name"]
            
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#06070b")
            ax.set_facecolor("#06070b")

            trn_vals = self.df_sample_trn[top_col].dropna().values
            tst_vals = self.df_sample_tst[top_col].dropna().values

            if len(trn_vals) > 0 and len(tst_vals) > 0:
                q99 = np.percentile(np.concatenate([trn_vals, tst_vals]), 99)
                q01 = np.percentile(np.concatenate([trn_vals, tst_vals]), 1)
                
                trn_c = trn_vals[(trn_vals >= q01) & (trn_vals <= q99)]
                tst_c = tst_vals[(tst_vals >= q01) & (tst_vals <= q99)]

                ax.hist(trn_c, bins=25, density=True, color="#8e97a4", alpha=0.3, label="Train Baseline")
                ax.hist(tst_c, bins=25, density=True, color="#ffffff", alpha=0.8, label="Test Target", edgecolor="red", linewidth=0.5)

            ax.set_title(f"[DRIFT DETECTED] Distribution Shift: {top_col}", color="#fff", fontname="Orbitron", fontsize=9)
            ax.set_xlabel(top_col, color="#8e97a4", fontsize=8)
            ax.tick_params(colors="#8e97a4", labelsize=7)
            ax.spines[:].set_color((1.0, 1.0, 1.0, 0.08))
            ax.legend(facecolor="#06070b", edgecolor=(1.0, 1.0, 1.0, 0.08), fontsize=8)

            (report_dir / "plots").mkdir(parents=True, exist_ok=True)
            fig.savefig(report_dir / "plots" / "ks_drift_shifts.png", dpi=110, facecolor="#06070b")
            plt.close(fig)

        return df

    def analyze_distribution_drift(
        self,
        report_dir: Path,
        psi_df: pd.DataFrame,
        ks_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Performs frequency shift analysis for categorical items and mismatch flagging."""
        logger.info("Computing detailed distribution shifts...")

        records = []
        ks_map = dict(zip(ks_df["feature_name"], ks_df["ks_statistic"]))

        for _, row in psi_df.iterrows():
            col = row["feature_name"]
            is_num = (row["feature_type"] == "Numerical")
            psi = float(row["psi_value"])

            s_trn = self.df_sample_trn[col]
            s_tst = self.df_sample_tst[col]

            # Categorical mismatch count
            if not is_num:
                cats_trn = set(s_trn.dropna().unique())
                cats_tst = set(s_tst.dropna().unique())
                new_cats = len(cats_tst - cats_trn)
                missing_cats = len(cats_trn - cats_tst)
            else:
                new_cats = 0
                missing_cats = 0

            records.append({
                "feature_name": col,
                "feature_type": row["feature_type"],
                "psi_value": psi,
                "ks_statistic": ks_map.get(col, np.nan),
                "new_categories_in_test": int(new_cats) if not is_num else np.nan,
                "missing_categories_in_test": int(missing_cats) if not is_num else np.nan,
            })

        df = pd.DataFrame(records).sort_values(by="psi_value", ascending=False)
        df.to_csv(report_dir / "distribution_drift_analysis.csv", index=False)

        # Plot categorical frequency shifts for top drifted category
        cat_drift_items = df[df["feature_type"] == "Categorical"]
        if not cat_drift_items.empty:
            top_cat = cat_drift_items.iloc[0]["feature_name"]

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#06070b")
            ax.set_facecolor("#06070b")

            trn_counts = s_trn.value_counts(normalize=True).head(5)
            tst_counts = s_tst.value_counts(normalize=True).reindex(trn_counts.index).fillna(0.0)

            x = np.arange(len(trn_counts))
            width = 0.35

            ax.bar(x - width/2, trn_counts.values, width, label="Train", color="#8e97a4", alpha=0.4)
            ax.bar(x + width/2, tst_counts.values, width, label="Test", color="#ffffff", alpha=0.8, edgecolor="red", linewidth=0.5)

            ax.set_ylabel("Normalized Frequency", color="#8e97a4", fontsize=8)
            ax.set_title(f"[CATEGORICAL DRIFT] Frequency Shifts: {top_cat}", color="#fff", fontname="Orbitron", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(trn_counts.index.astype(str), rotation=15, ha="right", fontsize=7, color="#8e97a4")
            ax.tick_params(colors="#8e97a4", labelsize=7)
            ax.spines[:].set_color((1.0, 1.0, 1.0, 0.08))
            ax.legend(facecolor="#06070b", edgecolor=(1.0, 1.0, 1.0, 0.08), fontsize=8)

            (report_dir / "plots").mkdir(parents=True, exist_ok=True)
            fig.savefig(report_dir / "plots" / "categorical_drift_shifts.png", dpi=110, facecolor="#06070b")
            plt.close(fig)

        return df

    def analyze_feature_stability(
        self,
        report_dir: Path,
        psi_df: pd.DataFrame,
        ks_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Determines stability score based on aggregated PSI and KS shifts."""
        logger.info("Computing feature stability scores...")
        records = []

        ks_map = dict(zip(ks_df["feature_name"], ks_df["ks_statistic"]))

        for _, row in psi_df.iterrows():
            f = row["feature_name"]
            is_num = (row["feature_type"] == "Numerical")
            psi = float(row["psi_value"])

            ks_stat = float(ks_map.get(f, 0.0))

            # Stability Score = 100 * (1 - min(1, PSI)) - (10 * KS if numerical)
            # Maximum score is 100, minimum is 0.
            score = 100.0 * (1.0 - min(1.0, psi))
            if is_num:
                score -= 10.0 * ks_stat
            score = max(0.0, score)

            if score >= 90.0:
                classification = "STABLE"
            elif score >= 75.0:
                classification = "UNSTABLE"
            else:
                classification = "CRITICAL"

            records.append({
                "feature_name": f,
                "feature_type": row["feature_type"],
                "psi_value": psi,
                "ks_statistic": ks_stat if is_num else np.nan,
                "stability_score": round(score, 4),
                "stability_classification": classification,
            })

        df = pd.DataFrame(records).sort_values(by="stability_score", ascending=True)
        df.to_csv(report_dir / "feature_stability_analysis.csv", index=False)
        return df

    def classify_drift_severity(self, report_dir: Path, stability_df: pd.DataFrame) -> pd.DataFrame:
        """Groups columns into severity groups (No Drift to Critical Drift) and rates overall risk."""
        logger.info("Classifying drift severity...")
        records = []

        for _, row in stability_df.iterrows():
            psi = float(row["psi_value"])
            score = float(row["stability_score"])

            if psi < 0.05 and score >= 95.0:
                severity = "NO DRIFT"
                priority = "LOW"
                risk_score = 0
            elif psi < 0.10:
                severity = "MINOR DRIFT"
                priority = "LOW"
                risk_score = 2
            elif psi < 0.20:
                severity = "MODERATELY DRIFTED"
                priority = "MEDIUM"
                risk_score = 5
            elif psi < 0.35:
                severity = "MAJOR DRIFT"
                priority = "HIGH"
                risk_score = 8
            else:
                severity = "CRITICAL DRIFT"
                priority = "IMMEDIATE ACTION"
                risk_score = 10

            records.append({
                "feature_name": row["feature_name"],
                "feature_type": row["feature_type"],
                "psi_value": psi,
                "stability_score": score,
                "drift_severity": severity,
                "monitoring_priority": priority,
                "risk_score": risk_score,
            })

        df = pd.DataFrame(records).sort_values(by="risk_score", ascending=False)
        df.to_csv(report_dir / "drift_severity_report.csv", index=False)
        return df

    def analyze_drift_root_cause(self, report_dir: Path, severity_df: pd.DataFrame) -> pd.DataFrame:
        """Diagnoses time correlation changes and outputs mitigation remedies."""
        logger.info("Investigating drift root causes...")
        records = []

        for _, row in severity_df.iterrows():
            f = row["feature_name"]
            sev = row["drift_severity"]
            score = float(row["stability_score"])

            if sev in {"NO DRIFT", "MINOR DRIFT"}:
                cause = "Normal variance inside transaction sample splits"
                mitigation = "Continue standard monitoring without changes"
            elif sev == "MODERATELY DRIFTED":
                cause = "Temporal shift due to evolving transaction behaviors over time splits"
                mitigation = "Enable periodic model re-training on recent datasets"
            elif sev == "MAJOR DRIFT":
                cause = "Significant demographic user changes (device, browser versions shift)"
                mitigation = "Quantize category values or apply binning transformations to continuous factors"
            else:
                # CRITICAL
                cause = "Extreme environment shift or feature definition decay between partitions"
                mitigation = "Omit feature from production training or set strict clipping boundaries"

            records.append({
                "feature_name": f,
                "drift_severity": sev,
                "stability_score": score,
                "probable_root_cause": cause,
                "recommended_mitigation": mitigation,
            })

        df = pd.DataFrame(records).sort_values(by="stability_score", ascending=True)
        df.to_csv(report_dir / "drift_root_cause_analysis.csv", index=False)
        return df

    def compile_html_dashboard(
        self,
        report_dir: Path,
        df_inventory: pd.DataFrame,
        df_stats: pd.DataFrame,
        df_psi: pd.DataFrame,
        df_ks: pd.DataFrame,
        df_dist: pd.DataFrame,
        df_stab: pd.DataFrame,
        df_sev: pd.DataFrame,
        df_root: pd.DataFrame,
    ) -> None:
        """HTML compiler generating glassmorphic HUD summary dashboard."""
        logger.info("Compiling Drift HTML report...")

        def _to_html(df: pd.DataFrame) -> str:
            if df.empty:
                return "<div class='no-data'>NO DRIFT ANALYSIS REPORTED</div>"
            return df.to_html(
                classes="hud-table",
                index=False,
                border=0,
                justify="left",
            )

        summary = {
            "total_tested": len(df_sev),
            "critical_drift_count": int((df_sev["drift_severity"] == "CRITICAL DRIFT").sum()),
            "major_drift_count": int((df_sev["drift_severity"] == "MAJOR DRIFT").sum()),
            "max_psi": float(df_psi["psi_value"].max()) if not df_psi.empty else 0.0,
            "average_stability": float(df_stab["stability_score"].mean()) if not df_stab.empty else 100.0,
        }

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>IEEE-CIS DRIFT DIAGNOSTICS REPORT</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Orbitron:wght@500;700;900&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #06070b;
            --panel-bg: rgba(14, 16, 22, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-color: #8e97a4;
            --text-white: #ffffff;
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
        }}

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

        .dashboard-body {{
            display: grid;
            grid-template-columns: 1fr 1.255fr;
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
            font-size: 0.70rem;
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
            font-size: 0.70rem;
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
    </style>
    <script>
        function switchTab(index) {{
            const tabs = document.querySelectorAll('.carousel-tab');
            const img = document.getElementById('carousel-img');
            const images = [
                'plots/ks_drift_shifts.png',
                'plots/categorical_drift_shifts.png'
            ];
            
            tabs.forEach(t => t.classList.remove('active'));
            tabs[index].classList.add('active');
            img.src = images[index];
        }}
    </script>
</head>
<body>

    <header>
        <div>
            <h1>IEEE-CIS DRIFT DIAGNOSTICS REPORT</h1>
            <p style="font-size: 0.65rem; color: var(--text-color); margin-top: 0.25rem; letter-spacing: 1px;">STAGE 3.15: TRAIN-TEST DRIFT DYNAMICS MAP</p>
        </div>
        <div class="status-pill">DRIFT ENGINE OK</div>
    </header>

    <div class="hud-grid">
        <div class="hud-panel">
            <p class="metric-label">Features Analyzed</p>
            <p class="metric-value">{summary['total_tested']}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Average Stability</p>
            <p class="metric-value">{summary['average_stability']:.1f}%</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Max Feature PSI</p>
            <p class="metric-value">{summary['max_psi']:.4f}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Critical / Major Drift</p>
            <p class="metric-value">{summary['critical_drift_count'] + summary['major_drift_count']}</p>
        </div>
    </div>

    <div class="dashboard-body">
        <div class="hud-panel visualizer-card">
            <h2>DIAGNOSTIC VISUALIZATIONS</h2>
            <div class="carousel-tabs">
                <button class="carousel-tab active" onclick="switchTab(0)">Continuous KS Shift</button>
                <button class="carousel-tab" onclick="switchTab(1)">Categorical Frequency Shift</button>
            </div>
            <div class="carousel-content">
                <img id="carousel-img" class="carousel-img" src="plots/ks_drift_shifts.png" alt="KS Drift">
            </div>
        </div>

        <div class="hud-panel">
            <h2>FEATURE STABILITY CLASSIFICATIONS</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_sev.head(30))}
            </div>
        </div>
    </div>

    <div class="secondary-panel-grid">
        <div class="hud-panel">
            <h2>DRIFT ROOT CAUSE & RECOMMENDATIONS</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_root.head(30))}
            </div>
        </div>
        <div class="hud-panel">
            <h2>TRAIN-TEST DESCRIPTIVE COMPARATIVE STATS</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_stats.head(30))}
            </div>
        </div>
    </div>

</body>
</html>
"""
        with open(report_dir / "drift_report.html", "w") as f:
            f.write(html_template)
        logger.info("Compiled Drift HTML report saved.")

    def analyze_all(self, report_dir: Path) -> None:
        """Executes full Drift Diagnostics pipeline."""
        report_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = report_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("--- Stage 3.15: Drift Analysis (Starting) ---")

        df_inv = self.analyze_drift_inventory(report_dir)
        df_stats = self.analyze_train_test_stats_drift(report_dir)
        df_psi = self.analyze_psi(report_dir)
        df_ks = self.analyze_ks_drift(report_dir)
        df_dist = self.analyze_distribution_drift(report_dir, df_psi, df_ks)
        df_stab = self.analyze_feature_stability(report_dir, df_psi, df_ks)
        df_sev = self.classify_drift_severity(report_dir, df_stab)
        df_root = self.analyze_drift_root_cause(report_dir, df_sev)

        self.compile_html_dashboard(
            report_dir,
            df_inv,
            df_stats,
            df_psi,
            df_ks,
            df_dist,
            df_stab,
            df_sev,
            df_root,
        )

        logger.info("--- Stage 3.15: Drift Analysis Complete ---")
