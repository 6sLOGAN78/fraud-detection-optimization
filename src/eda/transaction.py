# ruff: noqa: E501
"""Transaction Feature Analysis engine — Part 3.8 IEEE-CIS Fraud Detection EDA."""

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

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seconds per unit
_SECS_HOUR = 3600
_SECS_DAY = 86400
_SECS_WEEK = 604800

_AMT_BINS = [0, 20, 100, 500, 2000, np.inf]
_AMT_LABELS = ["Very Low", "Low", "Medium", "High", "Very High"]


def _amt_bin(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=_AMT_BINS, labels=_AMT_LABELS, right=True)


def _dist_bin(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return pd.cut(series, bins=4, labels=["Near", "Medium", "Far", "Extreme"])
    q = [s.min() - 1, s.quantile(0.25), s.quantile(0.5), s.quantile(0.75), s.max() + 1]
    q = sorted(set(q))
    n = len(q) - 1
    labels = ["Near", "Medium", "Far", "Extreme"][:n]
    return pd.cut(series, bins=q, labels=labels)


class TransactionFeatureAnalyzer:
    """Comprehensive domain-focused EDA on IEEE-CIS transaction features.

    Sub-modules:
    - 3.8.4  TransactionAmt analysis
    - 3.8.5  ProductCD analysis
    - 3.8.6  Card feature analysis (card1-6)
    - 3.8.7  Address feature analysis (addr1, addr2)
    - 3.8.8  Distance feature analysis (dist1, dist2)
    - 3.8.9  Transaction timing analysis (TransactionDT)
    - 3.8.10 Cross-feature interaction analysis
    - 3.8.11 Transaction risk profiling
    - 3.8.12 Feature engineering recommendations
    """

    _CARD_COLS = ["card1", "card2", "card3", "card4", "card5", "card6"]
    _ADDR_COLS = ["addr1", "addr2"]
    _DIST_COLS = ["dist1", "dist2"]

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
        logger.info("TransactionFeatureAnalyzer initialized. Train rows: %d", len(df_train))

    # ------------------------------------------------------------------ #
    # Orchestrator                                                         #
    # ------------------------------------------------------------------ #

    def analyze_all(self, report_dir: Path) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = report_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("--- Stage 3.8: Transaction Feature Analysis ---")

        df_amt = self.analyze_transaction_amount(report_dir)
        df_prod = self.analyze_productcd(report_dir)
        df_card = self.analyze_card_features(report_dir)
        df_addr = self.analyze_address_features(report_dir)
        df_dist = self.analyze_distance_features(report_dir)
        df_time = self.analyze_transaction_timing(report_dir)
        df_xfeat = self.analyze_cross_features(report_dir)
        df_risk = self.analyze_risk_profiles(report_dir)
        df_recs = self.generate_feature_engineering_recommendations(report_dir)

        self.generate_plots(plots_dir)

        summary = self._build_summary(df_amt, df_prod, df_card, df_risk)

        self.compile_html_dashboard(
            report_dir=report_dir,
            summary=summary,
            df_amt=df_amt,
            df_prod=df_prod,
            df_card=df_card,
            df_dist=df_dist,
            df_time=df_time,
            df_xfeat=df_xfeat,
            df_risk=df_risk,
            df_recs=df_recs,
        )

        with (report_dir / "transaction_analysis.json").open("w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Transaction Feature Analysis complete. Reports → %s", report_dir)

    # ------------------------------------------------------------------ #
    # 3.8.4 Transaction Amount                                            #
    # ------------------------------------------------------------------ #

    def analyze_transaction_amount(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing TransactionAmt...")
        col = "TransactionAmt"
        if col not in self.df_train.columns:
            logger.warning("'%s' not found — skipping.", col)
            return pd.DataFrame()

        amt = self.df_train[col].dropna()
        q1, q3 = float(amt.quantile(0.25)), float(amt.quantile(0.75))
        iqr = q3 - q1

        stats = {
            "min": float(amt.min()), "max": float(amt.max()),
            "mean": float(amt.mean()), "median": float(amt.median()),
            "std": float(amt.std()), "q1": q1, "q3": q3,
            "iqr": iqr,
            "cv": float(amt.std() / amt.mean()) if amt.mean() > 0 else float("nan"),
            "skewness": float(amt.skew()), "kurtosis": float(amt.kurt()),
        }

        # Fraud by amount bin
        rows: list[dict[str, Any]] = []
        if self.target_col in self.df_train.columns:
            sub = self.df_train[[col, self.target_col]].dropna(subset=[col])
            sub = sub.copy()
            sub["amt_bin"] = _amt_bin(sub[col])
            grp = sub.groupby("amt_bin", observed=True)[self.target_col].agg(
                fraud_count="sum", total_count="count"
            ).reset_index()
            grp["fraud_rate"] = (grp["fraud_count"] / grp["total_count"] * 100).round(4)
            grp["mean_amt"] = sub.groupby("amt_bin", observed=True)[col].mean().values
            grp["relative_risk"] = (grp["fraud_rate"] / (self._global_fraud_rate * 100)).round(4)
            rows = grp.to_dict("records")

        df = pd.DataFrame(rows) if rows else pd.DataFrame(list(stats.items()), columns=["metric", "value"])
        df.to_csv(report_dir / "transaction_amount_analysis.csv", index=False)

        with (report_dir / "transaction_amount_stats.json").open("w") as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info("TransactionAmt analysis complete: mean=%.2f, median=%.2f", stats["mean"], stats["median"])
        return df

    # ------------------------------------------------------------------ #
    # 3.8.5 ProductCD                                                     #
    # ------------------------------------------------------------------ #

    def analyze_productcd(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing ProductCD...")
        col = "ProductCD"
        if col not in self.df_train.columns:
            return pd.DataFrame()

        vc = self.df_train[col].value_counts()
        n = len(self.df_train[col].dropna())

        records: list[dict[str, Any]] = []
        for cat, cnt in vc.items():
            mask = self.df_train[col] == cat
            fraud_rate = float(self.df_train.loc[mask, self.target_col].mean() * 100) if self.target_col in self.df_train.columns else float("nan")
            mean_amt = float(self.df_train.loc[mask, "TransactionAmt"].mean()) if "TransactionAmt" in self.df_train.columns else float("nan")
            records.append({
                "product": str(cat),
                "count": int(cnt),
                "percentage": round(cnt / n * 100, 4),
                "fraud_rate": round(fraud_rate, 4),
                "relative_risk": round(fraud_rate / (self._global_fraud_rate * 100), 4) if self._global_fraud_rate > 0 else float("nan"),
                "mean_amt": round(mean_amt, 4),
            })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "productcd_analysis.csv", index=False)
        logger.info("ProductCD analysis complete: %d products", len(df))
        return df

    # ------------------------------------------------------------------ #
    # 3.8.6 Card Features                                                 #
    # ------------------------------------------------------------------ #

    def analyze_card_features(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing card features...")
        avail = [c for c in self._CARD_COLS if c in self.df_train.columns]
        records: list[dict[str, Any]] = []

        for col in avail:
            series = self.df_train[col]
            n_total = len(series)
            n_miss = int(series.isna().sum())
            n_unique = int(series.nunique())
            vc = series.value_counts()

            # Overall fraud rate for this feature
            if self.target_col in self.df_train.columns:
                sub = self.df_train[[col, self.target_col]].dropna(subset=[col])
                grp = sub.groupby(col, observed=True)[self.target_col].agg(["sum", "count"])
                grp["fraud_rate"] = grp["sum"] / grp["count"] * 100
                max_fraud_cat = str(grp["fraud_rate"].idxmax())
                max_fraud_rate = round(float(grp["fraud_rate"].max()), 4)
                max_rr = round(max_fraud_rate / (self._global_fraud_rate * 100), 4)
            else:
                max_fraud_cat = "N/A"
                max_fraud_rate = float("nan")
                max_rr = float("nan")

            records.append({
                "feature": col,
                "n_unique": n_unique,
                "missing_pct": round(n_miss / n_total * 100, 4),
                "dominant_category": str(vc.index[0]) if len(vc) > 0 else "N/A",
                "dominant_pct": round(float(vc.iloc[0]) / (n_total - n_miss) * 100, 4) if len(vc) > 0 else 0.0,
                "highest_fraud_category": max_fraud_cat,
                "highest_fraud_rate": max_fraud_rate,
                "relative_risk": max_rr,
            })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "card_feature_analysis.csv", index=False)
        logger.info("Card feature analysis complete: %d card features", len(df))
        return df

    # ------------------------------------------------------------------ #
    # 3.8.7 Address Features                                              #
    # ------------------------------------------------------------------ #

    def analyze_address_features(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing address features...")
        avail = [c for c in self._ADDR_COLS if c in self.df_train.columns]
        records: list[dict[str, Any]] = []

        for col in avail:
            series = self.df_train[col].dropna()
            n_total = len(self.df_train[col])
            n_miss = int(self.df_train[col].isna().sum())
            vc = series.value_counts()

            if self.target_col in self.df_train.columns:
                sub = self.df_train[[col, self.target_col]].dropna(subset=[col])
                grp = sub.groupby(col, observed=True)[self.target_col].agg(["sum", "count"])
                grp["fraud_rate"] = grp["sum"] / grp["count"] * 100
                top5_high_risk = grp.nlargest(5, "fraud_rate")[["fraud_rate"]].reset_index()
                top5_str = ", ".join(f"{r[col]}({r['fraud_rate']:.1f}%)" for _, r in top5_high_risk.iterrows())
            else:
                top5_str = "N/A"

            records.append({
                "feature": col,
                "n_unique": int(series.nunique()),
                "missing_pct": round(n_miss / n_total * 100, 4),
                "top_address": str(vc.index[0]) if len(vc) > 0 else "N/A",
                "top_address_count": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "top5_high_risk_addresses": top5_str,
            })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "address_feature_analysis.csv", index=False)
        logger.info("Address feature analysis complete.")
        return df

    # ------------------------------------------------------------------ #
    # 3.8.8 Distance Features                                             #
    # ------------------------------------------------------------------ #

    def analyze_distance_features(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing distance features...")
        avail = [c for c in self._DIST_COLS if c in self.df_train.columns]
        records: list[dict[str, Any]] = []

        for col in avail:
            series = self.df_train[col].dropna()
            n_total = len(self.df_train[col])
            n_miss = int(self.df_train[col].isna().sum())

            if series.empty:
                continue

            bin_col = _dist_bin(series)
            if self.target_col in self.df_train.columns:
                sub = self.df_train[[col, self.target_col]].dropna(subset=[col]).copy()
                sub["dist_bin"] = _dist_bin(sub[col])
                grp = sub.groupby("dist_bin", observed=True)[self.target_col].agg(
                    fraud_count="sum", total="count"
                ).reset_index()
                grp["fraud_rate"] = (grp["fraud_count"] / grp["total"] * 100).round(4)
                for _, row in grp.iterrows():
                    records.append({
                        "feature": col,
                        "dist_bin": str(row["dist_bin"]),
                        "total": int(row["total"]),
                        "fraud_count": int(row["fraud_count"]),
                        "fraud_rate": float(row["fraud_rate"]),
                        "mean": round(float(series.mean()), 4),
                        "median": round(float(series.median()), 4),
                        "std": round(float(series.std()), 4),
                        "missing_pct": round(n_miss / n_total * 100, 4),
                    })
            else:
                records.append({
                    "feature": col, "dist_bin": "ALL",
                    "total": int(len(series)), "fraud_count": 0, "fraud_rate": 0.0,
                    "mean": round(float(series.mean()), 4),
                    "median": round(float(series.median()), 4),
                    "std": round(float(series.std()), 4),
                    "missing_pct": round(n_miss / n_total * 100, 4),
                })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "distance_feature_analysis.csv", index=False)
        logger.info("Distance feature analysis complete.")
        return df

    # ------------------------------------------------------------------ #
    # 3.8.9 Transaction Timing                                            #
    # ------------------------------------------------------------------ #

    def analyze_transaction_timing(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing transaction timing...")
        col = "TransactionDT"
        if col not in self.df_train.columns:
            return pd.DataFrame()

        dt = self.df_train[col].dropna()
        hour = (dt // _SECS_HOUR % 24).astype(int)
        dow = (dt // _SECS_DAY % 7).astype(int)

        train = self.df_train.copy()
        train["hour"] = (train[col] // _SECS_HOUR % 24).astype(int)
        train["dow"] = (train[col] // _SECS_DAY % 7).astype(int)

        records: list[dict[str, Any]] = []
        for h in range(24):
            mask = train["hour"] == h
            sub = train[mask]
            fraud_rate = float(sub[self.target_col].mean() * 100) if self.target_col in sub.columns and len(sub) > 0 else 0.0
            records.append({
                "hour": h,
                "transaction_count": int(len(sub)),
                "fraud_count": int(sub[self.target_col].sum()) if self.target_col in sub.columns else 0,
                "fraud_rate": round(fraud_rate, 4),
            })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "transaction_time_analysis.csv", index=False)
        logger.info("Timing analysis complete. Peak hour: %d", df.loc[df["transaction_count"].idxmax(), "hour"])
        return df

    # ------------------------------------------------------------------ #
    # 3.8.10 Cross-Feature Interactions                                   #
    # ------------------------------------------------------------------ #

    def analyze_cross_features(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing cross-feature interactions...")
        df = self.df_train.copy()
        if "TransactionAmt" in df.columns:
            df["amt_bin"] = _amt_bin(df["TransactionAmt"])

        pairs = [
            ("amt_bin", "ProductCD"),
            ("amt_bin", "card4"),
            ("ProductCD", "card4"),
            ("ProductCD", "addr1"),
            ("card4", "addr1"),
            ("amt_bin", "card6"),
        ]

        all_rows: list[dict[str, Any]] = []
        for a, b in pairs:
            if a not in df.columns or b not in df.columns:
                continue
            sub = df[[a, b, self.target_col]].dropna() if self.target_col in df.columns else df[[a, b]].dropna()
            if sub.empty:
                continue
            try:
                grp = sub.groupby([a, b], observed=True).agg(
                    transaction_count=(self.target_col if self.target_col in sub.columns else a, "count"),
                    fraud_count=(self.target_col, "sum") if self.target_col in sub.columns else (a, "count"),
                    mean_amt=("TransactionAmt", "mean") if "TransactionAmt" in sub.columns else (a, "count"),
                ).reset_index()
                grp["fraud_rate"] = (grp["fraud_count"] / grp["transaction_count"] * 100).round(4) if "fraud_count" in grp.columns else 0.0
                grp["relative_risk"] = (grp["fraud_rate"] / (self._global_fraud_rate * 100)).round(4)
                grp.insert(0, "interaction", f"{a}×{b}")
                all_rows.append(grp)
            except Exception as exc:
                logger.warning("Cross-feature %s×%s failed: %s", a, b, exc)

        df_out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        df_out.to_csv(report_dir / "transaction_feature_interactions.csv", index=False)
        logger.info("Cross-feature analysis complete: %d interaction records", len(df_out))
        return df_out

    # ------------------------------------------------------------------ #
    # 3.8.11 Risk Profiling                                               #
    # ------------------------------------------------------------------ #

    def analyze_risk_profiles(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Building transaction risk profiles...")
        df = self.df_train.copy()

        has_target = self.target_col in df.columns
        if "TransactionAmt" in df.columns:
            df["amt_bin"] = _amt_bin(df["TransactionAmt"])
        if "TransactionDT" in df.columns:
            df["hour"] = (df["TransactionDT"] // _SECS_HOUR % 24).astype(int)
            df["is_night"] = df["hour"].isin(range(0, 6)).astype(int)

        profile_cols = [c for c in ["amt_bin", "ProductCD", "card4", "card6"] if c in df.columns]
        if len(profile_cols) < 2:
            return pd.DataFrame()

        sub_cols = profile_cols + ([self.target_col] if has_target else [])
        sub = df[sub_cols].dropna()

        try:
            agg_dict: dict[str, Any] = {"transaction_count": (self.target_col if has_target else profile_cols[0], "count")}
            if has_target:
                agg_dict["fraud_count"] = (self.target_col, "sum")
            grp = sub.groupby(profile_cols, observed=True).agg(**agg_dict).reset_index()
            if has_target:
                grp["fraud_rate"] = (grp["fraud_count"] / grp["transaction_count"] * 100).round(4)
                grp["risk_label"] = pd.cut(
                    grp["fraud_rate"],
                    bins=[-1, 1, 5, 15, 101],
                    labels=["Low", "Moderate", "High", "Critical"],
                )
            grp = grp.sort_values("transaction_count", ascending=False).head(100).reset_index(drop=True)
        except Exception as exc:
            logger.warning("Risk profiling groupby failed: %s", exc)
            return pd.DataFrame()

        grp.to_csv(report_dir / "transaction_risk_profiles.csv", index=False)
        logger.info("Risk profiles complete: %d profiles", len(grp))
        return grp

    # ------------------------------------------------------------------ #
    # 3.8.12 Feature Engineering Recommendations                          #
    # ------------------------------------------------------------------ #

    def generate_feature_engineering_recommendations(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Generating feature engineering recommendations...")
        records = [
            {"feature_group": "TransactionAmt", "engineered_feature": "log1p_TransactionAmt",
             "method": "log1p transform", "rationale": "Highly right-skewed; log reduces tail and stabilizes variance.", "priority": "High"},
            {"feature_group": "TransactionAmt", "engineered_feature": "amt_bin",
             "method": "Quantile binning (5 bands)", "rationale": "Non-linear fraud signal varies by amount magnitude.", "priority": "High"},
            {"feature_group": "TransactionAmt", "engineered_feature": "amt_zscore",
             "method": "Z-score normalization", "rationale": "Flag extreme outlier amounts per-user or globally.", "priority": "Medium"},
            {"feature_group": "Card Features", "engineered_feature": "card_freq_encoding",
             "method": "Frequency encoding", "rationale": "High-cardinality card1/card2; frequency proxy for card volume.", "priority": "High"},
            {"feature_group": "Card Features", "engineered_feature": "card_risk_score",
             "method": "Target encoding", "rationale": "card4/card6 fraud rates encode network-level risk.", "priority": "High"},
            {"feature_group": "Card Features", "engineered_feature": "card1_card2_combo",
             "method": "Concatenation hash", "rationale": "Unique card combinations identify specific cards without raw IDs.", "priority": "Medium"},
            {"feature_group": "Address Features", "engineered_feature": "addr1_freq_encoding",
             "method": "Frequency encoding", "rationale": "Address frequency signals shipping volume — high freq = retail.", "priority": "Medium"},
            {"feature_group": "Address Features", "engineered_feature": "addr_fraud_rate_encoding",
             "method": "Target encoding", "rationale": "Capture geographical fraud hotspots.", "priority": "High"},
            {"feature_group": "Distance Features", "engineered_feature": "dist1_bin",
             "method": "Quantile binning (Near/Medium/Far/Extreme)", "rationale": "Extreme distances correlate with card-not-present fraud.", "priority": "Medium"},
            {"feature_group": "Distance Features", "engineered_feature": "dist1_missing_flag",
             "method": "Binary missingness indicator", "rationale": "dist1 missing often means card-present transaction.", "priority": "Medium"},
            {"feature_group": "Timing", "engineered_feature": "hour_of_day",
             "method": "Modular arithmetic on TransactionDT", "rationale": "Fraud spikes at specific hours (e.g., late night).", "priority": "High"},
            {"feature_group": "Timing", "engineered_feature": "day_of_week",
             "method": "Modular arithmetic on TransactionDT", "rationale": "Weekend vs. weekday fraud patterns differ significantly.", "priority": "High"},
            {"feature_group": "Cross Features", "engineered_feature": "product_card_interaction",
             "method": "Label(ProductCD) × card4 frequency encoding", "rationale": "Product-card combination reveals specific risk patterns.", "priority": "High"},
            {"feature_group": "Cross Features", "engineered_feature": "amt_product_ratio",
             "method": "TransactionAmt / ProductCD mean amt", "rationale": "Deviation from product average amount flags anomalies.", "priority": "Medium"},
        ]

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "transaction_feature_recommendations.csv", index=False)
        logger.info("Feature engineering recommendations generated: %d items", len(df))
        return df

    # ------------------------------------------------------------------ #
    # Plot generation                                                      #
    # ------------------------------------------------------------------ #

    def generate_plots(self, plots_dir: Path) -> None:
        logger.info("Generating transaction feature plots...")
        plt.style.use("dark_background")
        _bg = "#06070b"
        _fg = "#8e97a4"
        _red = "#d63031"

        def _save(fig: plt.Figure, name: str) -> None:
            fig.savefig(plots_dir / name, dpi=110, bbox_inches="tight", facecolor=_bg)
            plt.close(fig)

        # 1. TransactionAmt histogram
        if "TransactionAmt" in self.df_train.columns:
            try:
                amt = self.df_train["TransactionAmt"].dropna()
                fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=_bg)
                for ax in axes:
                    ax.set_facecolor(_bg)
                axes[0].hist(amt, bins=60, color=_fg, edgecolor="#ffffff11")
                axes[0].set_title("[HIST] TransactionAmt", color="#fff", fontsize=10)
                axes[1].hist(np.log1p(amt), bins=60, color=_fg, edgecolor="#ffffff11")
                axes[1].set_title("[LOG] log1p(TransactionAmt)", color="#fff", fontsize=10)
                for ax in axes:
                    ax.tick_params(colors=_fg)
                    ax.spines[:].set_color("#ffffff11")
                plt.tight_layout()
                _save(fig, "transaction_amt_hist.png")
            except Exception as exc:
                logger.warning("Amt hist plot failed: %s", exc)

        # 2. Fraud rate by ProductCD
        if "ProductCD" in self.df_train.columns and self.target_col in self.df_train.columns:
            try:
                grp = self.df_train.groupby("ProductCD", observed=True)[self.target_col].mean() * 100
                fig, ax = plt.subplots(figsize=(8, 4), facecolor=_bg)
                ax.set_facecolor(_bg)
                colors = [_red if v > grp.mean() else _fg for v in grp.values]
                ax.bar(grp.index, grp.values, color=colors, edgecolor="#ffffff11")
                ax.axhline(grp.mean(), color="#fff", linewidth=0.8, linestyle="--", alpha=0.5)
                ax.set_title("[FRAUD RATE] by ProductCD", color=_red, fontsize=10)
                ax.tick_params(colors=_fg)
                ax.spines[:].set_color("#ffffff11")
                plt.tight_layout()
                _save(fig, "productcd_fraud_rate.png")
            except Exception as exc:
                logger.warning("ProductCD plot failed: %s", exc)

        # 3. Hourly transaction volume
        if "TransactionDT" in self.df_train.columns:
            try:
                hour = (self.df_train["TransactionDT"] // _SECS_HOUR % 24).astype(int)
                hc = hour.value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(10, 4), facecolor=_bg)
                ax.set_facecolor(_bg)
                ax.bar(hc.index, hc.values, color=_fg, edgecolor="#ffffff11")
                ax.set_xlabel("Hour of Day", color=_fg)
                ax.set_title("[TIMING] Hourly Transaction Volume", color="#fff", fontsize=10)
                ax.tick_params(colors=_fg)
                ax.spines[:].set_color("#ffffff11")
                plt.tight_layout()
                _save(fig, "hourly_volume.png")
            except Exception as exc:
                logger.warning("Timing plot failed: %s", exc)

        # 4. Fraud rate by hour
        if "TransactionDT" in self.df_train.columns and self.target_col in self.df_train.columns:
            try:
                df2 = self.df_train[["TransactionDT", self.target_col]].dropna().copy()
                df2["hour"] = (df2["TransactionDT"] // _SECS_HOUR % 24).astype(int)
                fr = df2.groupby("hour")[self.target_col].mean() * 100
                fig, ax = plt.subplots(figsize=(10, 4), facecolor=_bg)
                ax.set_facecolor(_bg)
                bar_colors = [_red if v > fr.mean() else _fg for v in fr.values]
                ax.bar(fr.index, fr.values, color=bar_colors, edgecolor="#ffffff11")
                ax.axhline(fr.mean(), color="#fff", linewidth=0.8, linestyle="--", alpha=0.5)
                ax.set_title("[FRAUD RATE] by Hour of Day", color=_red, fontsize=10)
                ax.tick_params(colors=_fg)
                ax.spines[:].set_color("#ffffff11")
                plt.tight_layout()
                _save(fig, "hourly_fraud_rate.png")
            except Exception as exc:
                logger.warning("Hourly fraud rate plot failed: %s", exc)

        logger.info("Transaction plots saved to %s", plots_dir)

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    def _build_summary(
        self,
        df_amt: pd.DataFrame,
        df_prod: pd.DataFrame,
        df_card: pd.DataFrame,
        df_risk: pd.DataFrame,
    ) -> dict[str, Any]:
        amt_col = "TransactionAmt"
        amt = self.df_train[amt_col] if amt_col in self.df_train.columns else pd.Series(dtype="float64")

        high_risk_profiles = 0
        if not df_risk.empty and "risk_label" in df_risk.columns:
            high_risk_profiles = int((df_risk["risk_label"].isin(["High", "Critical"])).sum())

        return {
            "total_transactions": int(len(self.df_train)),
            "fraud_transactions": int(self.df_train[self.target_col].sum()) if self.target_col in self.df_train.columns else 0,
            "global_fraud_rate_pct": round(self._global_fraud_rate * 100, 4),
            "amt_mean": round(float(amt.mean()), 4) if not amt.empty else 0.0,
            "amt_median": round(float(amt.median()), 4) if not amt.empty else 0.0,
            "amt_max": round(float(amt.max()), 4) if not amt.empty else 0.0,
            "product_count": int(len(df_prod)) if not df_prod.empty else 0,
            "card_features_analyzed": int(len(df_card)) if not df_card.empty else 0,
            "high_risk_profiles": high_risk_profiles,
        }

    # ------------------------------------------------------------------ #
    # HTML Compilation                                                     #
    # ------------------------------------------------------------------ #

    def compile_html_dashboard(
        self,
        report_dir: Path,
        summary: dict[str, Any],
        df_amt: pd.DataFrame,
        df_prod: pd.DataFrame,
        df_card: pd.DataFrame,
        df_dist: pd.DataFrame,
        df_time: pd.DataFrame,
        df_xfeat: pd.DataFrame,
        df_risk: pd.DataFrame,
        df_recs: pd.DataFrame,
    ) -> None:
        logger.info("Compiling Transaction HTML Dashboard...")

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
    <title>IEEE-CIS TRANSACTION DIAGNOSTICS</title>
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

        /* Futuristic Grid Overlay & Scanlines */
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
            background: rgba(46, 213, 115, 0.1);
            color: var(--alert-green);
            border: 1px solid var(--alert-green);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* Metric Grid */
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin-bottom: 2.5rem;
            position: relative;
            z-index: 10;
        }}

        .metric-card {{
            background: var(--panel-bg);
            backdrop-filter: blur(16px) saturate(120%);
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            border-radius: 4px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}

        .metric-title {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-color);
            margin-bottom: 0.8rem;
        }}

        .metric-value {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-white);
        }}

        .metric-desc {{
            font-size: 0.65rem;
            margin-top: 0.5rem;
            opacity: 0.6;
        }}

        /* Section Layout */
        .hud-section {{
            margin-bottom: 3.5rem;
            position: relative;
            z-index: 10;
        }}

        .section-header {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
            color: var(--text-white);
            letter-spacing: 2px;
            margin-bottom: 1.5rem;
            text-transform: uppercase;
            border-left: 3px solid var(--text-white);
            padding-left: 0.8rem;
        }}

        /* Grid for contents */
        .grid-split {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }}

        .hud-panel {{
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            backdrop-filter: blur(16px);
            padding: 1.5rem;
            border-radius: 4px;
        }}

        .hud-panel-title {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.85rem;
            color: var(--text-white);
            margin-bottom: 1.2rem;
            letter-spacing: 1px;
            text-transform: uppercase;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }}

        /* Visuals Carousel Section */
        .carousel-container {{
            margin-bottom: 2.5rem;
        }}

        .carousel-controls {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}

        .carousel-btn {{
            background: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-color);
            padding: 0.5rem 1rem;
            cursor: pointer;
            font-family: 'Orbitron', sans-serif;
            font-weight: 500;
            font-size: 0.75rem;
            text-transform: uppercase;
            transition: all 0.2s ease-in-out;
        }}

        .carousel-btn.active, .carousel-btn:hover {{
            background: var(--text-white);
            color: var(--bg-color);
            border-color: var(--text-white);
        }}

        .carousel-slide {{
            display: none;
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            text-align: center;
        }}

        .carousel-slide.active {{
            display: block;
        }}

        .carousel-slide img {{
            max-width: 100%;
            height: auto;
            border: 1px solid var(--border-color);
        }}

        /* Tables Styling */
        .hud-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.75rem;
            text-align: left;
        }}

        .hud-table th {{
            font-family: 'Orbitron', sans-serif;
            font-weight: 500;
            padding: 0.8rem;
            color: var(--text-white);
            border-bottom: 1px solid var(--text-white);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .hud-table td {{
            padding: 0.8rem;
            border-bottom: 1px solid var(--border-color);
        }}

        .hud-table tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
            color: var(--text-white);
        }}

        .no-data {{
            padding: 2rem;
            text-align: center;
            font-size: 0.8rem;
            letter-spacing: 1px;
            color: var(--alert-red);
            border: 1px dashed var(--border-color);
        }}
    </style>
</head>
<body>

    <header>
        <h1>IEEE-CIS TRANSACTION DIAGNOSTICS</h1>
        <div class="status-pill">TX ANALYZER OK</div>
    </header>

    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-title">TOTAL TRANSACTIONS</div>
            <div class="metric-value">{summary['total_transactions']:,}</div>
            <div class="metric-desc">Dataset size analyzed</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">GLOBAL FRAUD RATE</div>
            <div class="metric-value">{summary['global_fraud_rate_pct']:.4f}%</div>
            <div class="metric-desc">Overall baseline fraud rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">MEAN TRANSACTION AMT</div>
            <div class="metric-value">${summary['amt_mean']:.2f}</div>
            <div class="metric-desc">Average monetary value</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">HIGH-RISK PROFILES</div>
            <div class="metric-value" style="color: var(--alert-red);">{summary['high_risk_profiles']}</div>
            <div class="metric-desc">Profiles with critical fraud rate</div>
        </div>
    </div>

    <!-- Section 1: Visuals Carousel -->
    <div class="hud-section">
        <div class="section-header">Diagnostic Visualizations</div>
        <div class="carousel-container">
            <div class="carousel-controls">
                <button class="carousel-btn active" onclick="switchSlide(0)">Amount Distribution</button>
                <button class="carousel-btn" onclick="switchSlide(1)">ProductCD Fraud Risk</button>
                <button class="carousel-btn" onclick="switchSlide(2)">Hourly Volume</button>
                <button class="carousel-btn" onclick="switchSlide(3)">Hourly Fraud Rate</button>
            </div>
            <div id="slide-0" class="carousel-slide active">
                <img src="plots/transaction_amt_hist.png" alt="TransactionAmt distribution histograms">
            </div>
            <div id="slide-1" class="carousel-slide">
                <img src="plots/productcd_fraud_rate.png" alt="Fraud rate by ProductCD category">
            </div>
            <div id="slide-2" class="carousel-slide">
                <img src="plots/hourly_volume.png" alt="Hourly transaction volume histogram">
            </div>
            <div id="slide-3" class="carousel-slide">
                <img src="plots/hourly_fraud_rate.png" alt="Hourly fraud rate analysis">
            </div>
        </div>
    </div>

    <!-- Section 2: Distribution & Product Summary -->
    <div class="hud-section">
        <div class="section-header">Core Distributions</div>
        <div class="grid-split">
            <div class="hud-panel">
                <div class="hud-panel-title">Transaction Amount Binned Stats</div>
                {_to_html(df_amt)}
            </div>
            <div class="hud-panel">
                <div class="hud-panel-title">ProductCD Fraud & Amount Profile</div>
                {_to_html(df_prod)}
            </div>
        </div>
    </div>

    <!-- Section 3: Cards & Distances -->
    <div class="hud-section">
        <div class="section-header">Card & Distance Features</div>
        <div class="grid-split">
            <div class="hud-panel">
                <div class="hud-panel-title">Card Features (card1-6) Analysis</div>
                {_to_html(df_card)}
            </div>
            <div class="hud-panel">
                <div class="hud-panel-title">Distance Features (dist1-2) Bins</div>
                {_to_html(df_dist)}
            </div>
        </div>
    </div>

    <!-- Section 4: Risk Profiling & Cross Features -->
    <div class="hud-section">
        <div class="section-header">Cross-Feature Interactions & Risk Profiles</div>
        <div class="grid-split">
            <div class="hud-panel">
                <div class="hud-panel-title">Top 15 Cross-Feature Interactions</div>
                {_to_html(df_xfeat.head(15) if not df_xfeat.empty else df_xfeat)}
            </div>
            <div class="hud-panel">
                <div class="hud-panel-title">Top 15 Transaction Risk Profiles</div>
                {_to_html(df_risk.head(15) if not df_risk.empty else df_risk)}
            </div>
        </div>
    </div>

    <!-- Section 5: Engineering Recommendations -->
    <div class="hud-section">
        <div class="section-header">Feature Engineering Opportunities</div>
        <div class="hud-panel">
            <div class="hud-panel-title">Recommended Feature Pipeline Transformations</div>
            {_to_html(df_recs)}
        </div>
    </div>

    <script>
        function switchSlide(idx) {{
            const slides = document.querySelectorAll('.carousel-slide');
            const btns = document.querySelectorAll('.carousel-btn');
            
            slides.forEach((slide, i) => {{
                if (i === idx) {{
                    slide.classList.add('active');
                }} else {{
                    slide.classList.remove('active');
                }}
            }});
            
            btns.forEach((btn, i) => {{
                if (i === idx) {{
                    btn.classList.add('active');
                }} else {{
                    btn.classList.remove('active');
                }}
            }});
        }}
    </script>
</body>
</html>
"""
        dashboard_path = report_dir / "transaction_analysis_report.html"
        dashboard_path.write_text(html_template)
        logger.info("Compiled Transaction HTML dashboard saved to: %s", dashboard_path)

