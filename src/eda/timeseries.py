# ruff: noqa: E501
"""Time Series Feature Analysis engine — Part 3.10 IEEE-CIS Fraud Detection EDA."""

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

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesFeatureAnalyzer:
    """Comprehensive analysis engine for IEEE-CIS Time Series Features via TransactionDT.

    Sub-modules:
    - 3.10.4  TransactionDT Analysis
    - 3.10.5  Hourly Analysis
    - 3.10.6  Daily Analysis
    - 3.10.7  Weekly Analysis
    - 3.10.8  Monthly Trend Analysis
    - 3.10.9  Fraud Trend Analysis
    - 3.10.10 Seasonality Analysis
    - 3.10.11 Temporal Drift Analysis
    - 3.10.12 Temporal Anomaly Detection
    - 3.10.13 Time-Based Feature Recommendations
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
        logger.info("TimeSeriesFeatureAnalyzer initialized. Train rows: %d, Test rows: %d", len(df_train), len(df_test))

        # Add derived date/time columns to internal dataframes for vectorized speed
        for df in [self.df_train, self.df_test]:
            if "TransactionDT" in df.columns:
                # Convert TransactionDT (seconds) to hours, days, weeks, months
                df["dt_hour"] = (df["TransactionDT"] // 3600) % 24
                df["dt_day"] = df["TransactionDT"] // 86400
                df["dt_day_of_week"] = df["dt_day"] % 7
                df["dt_week"] = df["TransactionDT"] // (86400 * 7)
                df["dt_month"] = df["TransactionDT"] // (86400 * 30)

    # ------------------------------------------------------------------ #
    # Orchestrator                                                         #
    # ------------------------------------------------------------------ #

    def analyze_all(self, report_dir: Path) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = report_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("--- Stage 3.10: Time Series Feature Analysis ---")

        df_dt = self.analyze_transaction_dt(report_dir)
        df_hourly = self.analyze_hourly(report_dir)
        df_daily = self.analyze_daily(report_dir)
        df_weekly = self.analyze_weekly(report_dir)
        df_monthly = self.analyze_monthly(report_dir)
        df_fraud = self.analyze_fraud_trends(report_dir)
        df_season = self.analyze_seasonality(report_dir)
        df_drift = self.analyze_temporal_drift(report_dir)
        df_anomaly = self.detect_temporal_anomalies(report_dir)
        df_recs = self.generate_time_feature_recommendations(report_dir)

        self.generate_plots(plots_dir)

        summary = self._build_summary(df_dt, df_hourly, df_drift, df_anomaly)

        self.compile_html_dashboard(
            report_dir=report_dir,
            summary=summary,
            df_dt=df_dt,
            df_hourly=df_hourly,
            df_daily=df_daily,
            df_weekly=df_weekly,
            df_monthly=df_monthly,
            df_fraud=df_fraud,
            df_season=df_season,
            df_drift=df_drift,
            df_anomaly=df_anomaly,
            df_recs=df_recs,
        )

        with (report_dir / "timeseries_analysis.json").open("w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Time Series Feature Analysis complete. Reports → %s", report_dir)

    # ------------------------------------------------------------------ #
    # 3.10.4 TransactionDT Analysis                                      #
    # ------------------------------------------------------------------ #

    def analyze_transaction_dt(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing TransactionDT baseline metrics...")
        col = "TransactionDT"
        if col not in self.df_train.columns:
            return pd.DataFrame()

        t_min = float(self.df_train[col].min())
        t_max = float(self.df_train[col].max())
        t_span = t_max - t_min
        days_span = t_span / 86400.0
        n_tx = len(self.df_train)

        density = n_tx / days_span if days_span > 0 else 0.0
        avg_hourly = n_tx / (days_span * 24.0) if days_span > 0 else 0.0

        records = [
            {"metric": "Minimum TransactionDT", "value": t_min, "description": "Start timestamp of the dataset (seconds)"},
            {"metric": "Maximum TransactionDT", "value": t_max, "description": "End timestamp of the dataset (seconds)"},
            {"metric": "Time Span (Seconds)", "value": t_span, "description": "Delta between maximum and minimum timestamps"},
            {"metric": "Time Span (Days)", "value": round(days_span, 4), "description": "Delta converted to days"},
            {"metric": "Transaction Density (per Day)", "value": round(density, 4), "description": "Average transaction throughput per day"},
            {"metric": "Transaction Density (per Hour)", "value": round(avg_hourly, 4), "description": "Average transaction throughput per hour"},
        ]

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "transactiondt_analysis.csv", index=False)

        summary = {
            "min_dt": t_min,
            "max_dt": t_max,
            "span_seconds": t_span,
            "span_days": round(days_span, 4),
            "density_per_day": round(density, 4),
            "density_per_hour": round(avg_hourly, 4),
        }
        with (report_dir / "transactiondt_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        return df

    # ------------------------------------------------------------------ #
    # 3.10.5 Hourly Analysis                                             #
    # ------------------------------------------------------------------ #

    def analyze_hourly(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing Hourly transaction metrics...")
        col = "dt_hour"
        if col not in self.df_train.columns:
            return pd.DataFrame()

        has_target = self.target_col in self.df_train.columns
        agg_cols = {
            "transaction_count": (col, "count"),
            "total_amount": ("TransactionAmt", "sum") if "TransactionAmt" in self.df_train.columns else (col, "count"),
        }
        if has_target:
            agg_cols["fraud_count"] = (self.target_col, "sum")

        grp = self.df_train.groupby(col).agg(**agg_cols).reset_index()
        grp["percentage"] = (grp["transaction_count"] / len(self.df_train) * 100).round(4)
        grp["average_amount"] = (grp["total_amount"] / grp["transaction_count"]).round(4)

        if has_target:
            grp["fraud_rate"] = (grp["fraud_count"] / grp["transaction_count"] * 100).round(4)
            grp["relative_risk"] = (grp["fraud_rate"] / (self._global_fraud_rate * 100)).round(4)
        else:
            grp["fraud_rate"] = 0.0
            grp["relative_risk"] = 1.0

        grp.to_csv(report_dir / "hourly_analysis.csv", index=False)
        return grp

    # ------------------------------------------------------------------ #
    # 3.10.6 Daily Analysis                                              #
    # ------------------------------------------------------------------ #

    def analyze_daily(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing Daily transaction metrics...")
        col = "dt_day"
        if col not in self.df_train.columns:
            return pd.DataFrame()

        has_target = self.target_col in self.df_train.columns
        agg_cols = {
            "transaction_count": (col, "count"),
            "total_amount": ("TransactionAmt", "sum") if "TransactionAmt" in self.df_train.columns else (col, "count"),
        }
        if has_target:
            agg_cols["fraud_count"] = (self.target_col, "sum")

        grp = self.df_train.groupby(col).agg(**agg_cols).reset_index()
        grp["percentage"] = (grp["transaction_count"] / len(self.df_train) * 100).round(4)
        grp["average_amount"] = (grp["total_amount"] / grp["transaction_count"]).round(4)

        if has_target:
            grp["fraud_rate"] = (grp["fraud_count"] / grp["transaction_count"] * 100).round(4)
            grp["relative_risk"] = (grp["fraud_rate"] / (self._global_fraud_rate * 100)).round(4)
        else:
            grp["fraud_rate"] = 0.0
            grp["relative_risk"] = 1.0

        grp.to_csv(report_dir / "daily_analysis.csv", index=False)
        return grp

    # ------------------------------------------------------------------ #
    # 3.10.7 Weekly Analysis                                             #
    # ------------------------------------------------------------------ #

    def analyze_weekly(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing Weekly transaction metrics...")
        col = "dt_week"
        if col not in self.df_train.columns:
            return pd.DataFrame()

        has_target = self.target_col in self.df_train.columns
        agg_cols = {
            "transaction_count": (col, "count"),
            "total_amount": ("TransactionAmt", "sum") if "TransactionAmt" in self.df_train.columns else (col, "count"),
        }
        if has_target:
            agg_cols["fraud_count"] = (self.target_col, "sum")

        grp = self.df_train.groupby(col).agg(**agg_cols).reset_index()
        grp["percentage"] = (grp["transaction_count"] / len(self.df_train) * 100).round(4)
        grp["average_amount"] = (grp["total_amount"] / grp["transaction_count"]).round(4)

        if has_target:
            grp["fraud_rate"] = (grp["fraud_count"] / grp["transaction_count"] * 100).round(4)
            grp["relative_risk"] = (grp["fraud_rate"] / (self._global_fraud_rate * 100)).round(4)
        else:
            grp["fraud_rate"] = 0.0
            grp["relative_risk"] = 1.0

        # Calculate WoW Growth
        grp["volume_growth"] = grp["transaction_count"].pct_change().round(4)

        grp.to_csv(report_dir / "weekly_analysis.csv", index=False)
        return grp

    # ------------------------------------------------------------------ #
    # 3.10.8 Monthly Trend Analysis                                      #
    # ------------------------------------------------------------------ #

    def analyze_monthly(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing Monthly transaction metrics...")
        col = "dt_month"
        if col not in self.df_train.columns:
            return pd.DataFrame()

        has_target = self.target_col in self.df_train.columns
        agg_cols = {
            "transaction_count": (col, "count"),
            "total_amount": ("TransactionAmt", "sum") if "TransactionAmt" in self.df_train.columns else (col, "count"),
        }
        if has_target:
            agg_cols["fraud_count"] = (self.target_col, "sum")

        grp = self.df_train.groupby(col).agg(**agg_cols).reset_index()
        grp["average_amount"] = (grp["total_amount"] / grp["transaction_count"]).round(4)

        if has_target:
            grp["fraud_rate"] = (grp["fraud_count"] / grp["transaction_count"] * 100).round(4)
            grp["relative_risk"] = (grp["fraud_rate"] / (self._global_fraud_rate * 100)).round(4)
        else:
            grp["fraud_rate"] = 0.0
            grp["relative_risk"] = 1.0

        grp["monthly_growth"] = grp["transaction_count"].pct_change().round(4)

        grp.to_csv(report_dir / "monthly_analysis.csv", index=False)
        return grp

    # ------------------------------------------------------------------ #
    # 3.10.9 Fraud Trend Analysis                                        #
    # ------------------------------------------------------------------ #

    def analyze_fraud_trends(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing Fraud temporal trends...")
        if "dt_day" not in self.df_train.columns or self.target_col not in self.df_train.columns:
            return pd.DataFrame()

        # Day-level count of transactions and fraud
        grp = self.df_train.groupby("dt_day").agg(
            transactions=("isFraud", "count"),
            fraud_count=(self.target_col, "sum"),
        ).reset_index()

        grp["fraud_rate"] = (grp["fraud_count"] / grp["transactions"] * 100).round(4)
        
        # Calculate Rolling metrics (7-day window)
        grp["rolling_fraud_count"] = grp["fraud_count"].rolling(window=7, min_periods=1).mean().round(4)
        grp["rolling_fraud_rate"] = grp["fraud_rate"].rolling(window=7, min_periods=1).mean().round(4)
        grp["rolling_std_fraud_count"] = grp["fraud_count"].rolling(window=7, min_periods=1).std().fillna(0).round(4)

        # Growth Rate (Day over Day)
        grp["fraud_count_dod_growth"] = grp["fraud_count"].pct_change().round(4)

        # Spike detection (z-score on fraud count above 2.0 std dev)
        grp["fraud_count_rolling_diff"] = grp["fraud_count"] - grp["rolling_fraud_count"]
        grp["fraud_spike"] = (grp["fraud_count_rolling_diff"] > (2.0 * grp["rolling_std_fraud_count"])).astype(int)

        grp.to_csv(report_dir / "fraud_trend_analysis.csv", index=False)
        return grp

    # ------------------------------------------------------------------ #
    # 3.10.10 Seasonality Analysis                                       #
    # ------------------------------------------------------------------ #

    def analyze_seasonality(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing transaction seasonality indices...")
        
        # We compute seasonal indices for Hour of the day and Day of the week.
        # Seasonal Index = Group Average / Series Average
        records: list[dict[str, Any]] = []

        # 1. Hourly Seasonality
        if "dt_hour" in self.df_train.columns:
            grand_mean = len(self.df_train) / 24.0
            hourly_counts = self.df_train["dt_hour"].value_counts().reindex(range(24), fill_value=0)
            for hr, count in hourly_counts.items():
                idx = float(count / grand_mean)
                records.append({
                    "seasonality_dimension": "Hour of Day",
                    "dimension_value": int(hr),
                    "frequency_count": int(count),
                    "seasonal_index": round(idx, 4),
                    "periodicity_strength": "Strong" if abs(idx - 1.0) > 0.15 else "Weak"
                })

        # 2. Day of Week Seasonality
        if "dt_day_of_week" in self.df_train.columns:
            grand_mean = len(self.df_train) / 7.0
            dow_counts = self.df_train["dt_day_of_week"].value_counts().reindex(range(7), fill_value=0)
            for dow, count in dow_counts.items():
                idx = float(count / grand_mean)
                records.append({
                    "seasonality_dimension": "Day of Week",
                    "dimension_value": int(dow),
                    "frequency_count": int(count),
                    "seasonal_index": round(idx, 4),
                    "periodicity_strength": "Strong" if abs(idx - 1.0) > 0.05 else "Weak"
                })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "seasonality_analysis.csv", index=False)
        return df

    # ------------------------------------------------------------------ #
    # 3.10.11 Temporal Drift Analysis                                    #
    # ------------------------------------------------------------------ #

    def analyze_temporal_drift(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing Temporal Drift (PSI & JSD) between Train and Test...")
        
        # Since absolute TransactionDT does not overlap, we calculate drift on 'hour' of the day
        if "dt_hour" not in self.df_train.columns or "dt_hour" not in self.df_test.columns:
            return pd.DataFrame()

        # Distribution of hours in train
        train_counts = self.df_train["dt_hour"].value_counts(normalize=True).reindex(range(24), fill_value=0.0)
        # Distribution of hours in test
        test_counts = self.df_test["dt_hour"].value_counts(normalize=True).reindex(range(24), fill_value=0.0)

        records: list[dict[str, Any]] = []
        psi_total = 0.0

        for hr in range(24):
            q_i = train_counts[hr]
            p_i = test_counts[hr]

            # Avoid division by zero
            q_i_safe = max(q_i, 1e-5)
            p_i_safe = max(p_i, 1e-5)

            # PSI contribution
            psi_i = (p_i - q_i) * np.log(p_i_safe / q_i_safe)
            psi_total += psi_i

            # Jensen-Shannon Divergence contribution
            m_i = 0.5 * (p_i + q_i)
            m_i_safe = max(m_i, 1e-5)
            jsd_i = 0.5 * (p_i * np.log(p_i_safe / m_i_safe) + q_i * np.log(q_i_safe / m_i_safe))

            records.append({
                "hour": hr,
                "train_pct": round(float(q_i * 100), 4),
                "test_pct": round(float(p_i * 100), 4),
                "psi_contribution": round(float(psi_i), 6),
                "jsd_contribution": round(float(jsd_i), 6),
            })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "temporal_drift_analysis.csv", index=False)

        drift_report = {
            "psi_total": round(psi_total, 6),
            "jsd_total": round(float(df["jsd_contribution"].sum()), 6),
            "drift_status": "Significant Drift" if psi_total > 0.2 else "Moderate Drift" if psi_total > 0.1 else "Stable",
        }
        with (report_dir / "temporal_drift_report.json").open("w") as f:
            json.dump(drift_report, f, indent=2)

        return df

    # ------------------------------------------------------------------ #
    # 3.10.12 Temporal Anomaly Detection                                 #
    # ------------------------------------------------------------------ #

    def detect_temporal_anomalies(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Detecting daily transaction anomalies...")
        if "dt_day" not in self.df_train.columns:
            return pd.DataFrame()

        grp = self.df_train.groupby("dt_day").size().reset_index(name="transaction_count")
        
        # Compute rolling window statistics
        grp["rolling_mean"] = grp["transaction_count"].rolling(window=7, min_periods=1).mean()
        grp["rolling_std"] = grp["transaction_count"].rolling(window=7, min_periods=1).std().fillna(0)

        # Detect anomalies where count is 2.5 std devs away from rolling mean
        grp["z_score"] = np.where(
            grp["rolling_std"] > 0, 
            (grp["transaction_count"] - grp["rolling_mean"]) / grp["rolling_std"], 
            0.0
        )
        grp["anomaly_flag"] = (np.abs(grp["z_score"]) > 2.5).astype(int)

        grp.to_csv(report_dir / "temporal_anomalies.csv", index=False)
        return grp

    # ------------------------------------------------------------------ #
    # 3.10.13 Time Recommendations                                       #
    # ------------------------------------------------------------------ #

    def generate_time_feature_recommendations(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Generating feature engineering recommendations...")
        records = [
            {"category": "Basic Features", "feature": "dt_hour", "formula_description": "(TransactionDT // 3600) % 24", "rationale": "Extract diurnal behavior patterns."},
            {"category": "Basic Features", "feature": "dt_day_of_week", "formula_description": "(TransactionDT // 86400) % 7", "rationale": "Capture weekday vs weekend behavior changes."},
            {"category": "Basic Features", "feature": "is_night_transaction", "formula_description": "dt_hour in [23, 0, 1, 2, 3, 4, 5]", "rationale": "High-risk time bounds of abnormal user sessions."},
            {"category": "Cyclical Features", "feature": "dt_hour_sin / dt_hour_cos", "formula_description": "sin/cos(2 * pi * dt_hour / 24)", "rationale": "Maintain structural continuity across midnight transition (23h -> 0h)."},
            {"category": "Rolling Features", "feature": "user_transactions_last_hour", "formula_description": "Rolling count using identity keys", "rationale": "Identify high-velocity card grooming attempts."},
            {"category": "Trend Features", "feature": "time_since_last_txn", "formula_description": "TransactionDT - prev_TransactionDT", "rationale": "Identify rapid sequential transaction surges."},
        ]
        df = pd.DataFrame(records)
        df.to_csv(report_dir / "time_feature_recommendations.csv", index=False)
        return df

    # ------------------------------------------------------------------ #
    # Core Plots                                                         #
    # ------------------------------------------------------------------ #

    def generate_plots(self, plots_dir: Path) -> None:
        logger.info("Generating plot assets...")
        plt.style.use("dark_background")
        _bg = "#06070b"
        _fg = "#8e97a4"
        _red = "#d63031"

        def _save(fig: plt.Figure, name: str) -> None:
            fig.savefig(plots_dir / name, dpi=110, bbox_inches="tight", facecolor=_bg)
            plt.close(fig)

        # 1. Daily transaction counts and rolling averages
        if "dt_day" in self.df_train.columns:
            try:
                grp = self.df_train.groupby("dt_day").size().reset_index(name="cnt")
                grp["roll"] = grp["cnt"].rolling(window=7, min_periods=1).mean()
                fig, ax = plt.subplots(figsize=(8, 4), facecolor=_bg)
                ax.set_facecolor(_bg)
                ax.plot(grp["dt_day"], grp["cnt"], color=_fg, alpha=0.4, label="Daily Counts")
                ax.plot(grp["dt_day"], grp["roll"], color="#fff", linewidth=1.5, label="7-Day Rolling Mean")
                ax.set_title("[TIMELINE] Daily Transaction volume", color="#fff", fontsize=10)
                ax.tick_params(colors=_fg)
                ax.spines[:].set_color("#ffffff11")
                ax.legend(facecolor=_bg, edgecolor="#ffffff11", fontsize=8)
                plt.tight_layout()
                _save(fig, "transaction_timeline.png")
            except Exception as exc:
                logger.warning("Timeline plot failed: %s", exc)

        # 2. Hourly transaction volume and fraud rate
        if "dt_hour" in self.df_train.columns:
            try:
                grp = self.df_train.groupby("dt_hour").agg(
                    cnt=("dt_hour", "count"),
                    fraud=(self.target_col, "mean") if self.target_col in self.df_train.columns else ("dt_hour", "mean"),
                ).reset_index()
                grp["fraud_rate"] = grp["fraud"] * 100

                fig, ax1 = plt.subplots(figsize=(8, 4), facecolor=_bg)
                ax1.set_facecolor(_bg)
                ax2 = ax1.twinx()

                ax1.bar(grp["dt_hour"], grp["cnt"], color=_fg, alpha=0.3, edgecolor="#ffffff11", label="Tx Volume")
                ax2.plot(grp["dt_hour"], grp["fraud_rate"], color=_red, marker="o", markersize=4, linewidth=1.5, label="Fraud Rate (%)")

                ax1.set_title("[DIURNAL] Hourly Transaction Volume vs Fraud Rate", color="#fff", fontsize=10)
                ax1.tick_params(colors=_fg)
                ax2.tick_params(colors=_red)
                ax1.spines[:].set_color("#ffffff11")
                ax2.spines[:].set_color("#ffffff11")
                plt.tight_layout()
                _save(fig, "hourly_distribution.png")
            except Exception as exc:
                logger.warning("Hourly distribution plot failed: %s", exc)

        # 3. Weekly trend
        if "dt_week" in self.df_train.columns:
            try:
                grp = self.df_train.groupby("dt_week").size().reset_index(name="cnt")
                fig, ax = plt.subplots(figsize=(8, 4), facecolor=_bg)
                ax.set_facecolor(_bg)
                ax.bar(grp["dt_week"], grp["cnt"], color=_fg, edgecolor="#ffffff11")
                ax.set_title("[WEEKLY] Weekly Transaction Count", color="#fff", fontsize=10)
                ax.tick_params(colors=_fg)
                ax.spines[:].set_color("#ffffff11")
                plt.tight_layout()
                _save(fig, "weekly_trend.png")
            except Exception as exc:
                logger.warning("Weekly trend plot failed: %s", exc)

        # 4. Train vs Test Hourly Drift
        if "dt_hour" in self.df_train.columns and "dt_hour" in self.df_test.columns:
            try:
                train_dist = self.df_train["dt_hour"].value_counts(normalize=True).reindex(range(24), fill_value=0.0)
                test_dist = self.df_test["dt_hour"].value_counts(normalize=True).reindex(range(24), fill_value=0.0)

                x = np.arange(24)
                width = 0.35

                fig, ax = plt.subplots(figsize=(8, 4), facecolor=_bg)
                ax.set_facecolor(_bg)
                ax.bar(x - width/2, train_dist.values, width, label="Train Distribution", color=_fg)
                ax.bar(x + width/2, test_dist.values, width, label="Test Distribution", color="#fff", alpha=0.7)

                ax.set_title("[TEMPORAL DRIFT] Hourly Distribution Comparison", color="#fff", fontsize=10)
                ax.tick_params(colors=_fg)
                ax.spines[:].set_color("#ffffff11")
                ax.legend(facecolor=_bg, edgecolor="#ffffff11", fontsize=8)
                plt.tight_layout()
                _save(fig, "drift_analysis.png")
            except Exception as exc:
                logger.warning("Drift plot failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Summary Builder                                                    #
    # ------------------------------------------------------------------ #

    def _build_summary(
        self,
        df_dt: pd.DataFrame,
        df_hourly: pd.DataFrame,
        df_drift: pd.DataFrame,
        df_anomaly: pd.DataFrame,
    ) -> dict[str, Any]:
        
        # Max hourly risk
        peak_hour = -1
        max_hour_fr = 0.0
        if not df_hourly.empty and "fraud_rate" in df_hourly.columns:
            idx = df_hourly["fraud_rate"].idxmax()
            peak_hour = int(df_hourly.loc[idx, "dt_hour"])
            max_hour_fr = float(df_hourly.loc[idx, "fraud_rate"])

        # Temporal anomalies count
        anomalies_count = 0
        if not df_anomaly.empty and "anomaly_flag" in df_anomaly.columns:
            anomalies_count = int(df_anomaly["anomaly_flag"].sum())

        # Drift score
        drift_total = 0.0
        if not df_drift.empty and "psi_contribution" in df_drift.columns:
            drift_total = float(df_drift["psi_contribution"].sum())

        t_min = float(self.df_train["TransactionDT"].min()) if "TransactionDT" in self.df_train.columns else 0.0
        t_max = float(self.df_train["TransactionDT"].max()) if "TransactionDT" in self.df_train.columns else 0.0
        days_span = float((t_max - t_min) / 86400.0)

        return {
            "total_transactions": int(len(self.df_train)),
            "days_covered": round(days_span, 4),
            "peak_risk_hour": peak_hour,
            "peak_risk_hour_fraud_rate_pct": round(max_hour_fr, 4),
            "temporal_anomalies_detected": anomalies_count,
            "hourly_distribution_drift_psi": round(drift_total, 4),
        }

    # ------------------------------------------------------------------ #
    # HTML compilation                                                   #
    # ------------------------------------------------------------------ #

    def compile_html_dashboard(
        self,
        report_dir: Path,
        summary: dict[str, Any],
        df_dt: pd.DataFrame,
        df_hourly: pd.DataFrame,
        df_daily: pd.DataFrame,
        df_weekly: pd.DataFrame,
        df_monthly: pd.DataFrame,
        df_fraud: pd.DataFrame,
        df_season: pd.DataFrame,
        df_drift: pd.DataFrame,
        df_anomaly: pd.DataFrame,
        df_recs: pd.DataFrame,
    ) -> None:
        logger.info("Compiling Timeseries HTML Dashboard...")

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
    <title>IEEE-CIS TIME SERIES DIAGNOSTICS</title>
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
            grid-template-columns: 1.5fr 1fr;
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
            padding: 0.5rem 1rem;
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
            max-height: 480px;
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
            <h1>IEEE-CIS TIME SERIES DIAGNOSTICS</h1>
            <p style="font-size: 0.65rem; color: var(--text-color); margin-top: 0.25rem; letter-spacing: 1px;">STAGE 3.10: SYSTEM TIME PROFILE</p>
        </div>
        <div class="status-pill">TIME ANALYZER OK</div>
    </header>

    <div class="hud-grid">
        <div class="hud-panel">
            <p class="metric-label">Total Transactions</p>
            <p class="metric-value">{summary['total_transactions']:,}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Timeline Coverage</p>
            <p class="metric-value">{summary['days_covered']} Days</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Hourly Drift (PSI)</p>
            <p class="metric-value">{summary['hourly_distribution_drift_psi']}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Temporal Anomalies</p>
            <p class="metric-value metric-val-red">{summary['temporal_anomalies_detected']}</p>
        </div>
    </div>

    <div class="dashboard-body">
        <div class="hud-panel visualizer-card">
            <h2>DIAGNOSTIC VISUALIZATIONS</h2>
            <div class="carousel-tabs">
                <button class="carousel-tab active" onclick="switchTab(0)">Transaction Timeline</button>
                <button class="carousel-tab" onclick="switchTab(1)">Hourly Distribution</button>
                <button class="carousel-tab" onclick="switchTab(2)">Weekly Trend</button>
                <button class="carousel-tab" onclick="switchTab(3)">Drift Analysis</button>
            </div>
            <div class="carousel-content">
                <img id="carousel-img" class="carousel-img" src="plots/transaction_timeline.png" alt="Timeline Plot">
            </div>
        </div>

        <div class="hud-panel">
            <h2>HOURLY ANALYSIS SUMMARY</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_hourly)}
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
                        <th>Feature Name</th>
                        <th>Formula / Logic</th>
                        <th>ML Rationale</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f"<tr><td><span class='tag-pill tag-pill-high'>{row['category']}</span></td><td><strong>{row['feature']}</strong></td><td><code>{row['formula_description']}</code></td><td>{row['rationale']}</td></tr>" for idx, row in df_recs.iterrows())}
                </tbody>
            </table>
        </div>
    </div>

    <div class="secondary-panel-grid">
        <div class="hud-panel">
            <h2>TRANSACTIONDT SUMMARY METRICS</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_dt)}
            </div>
        </div>
        <div class="hud-panel">
            <h2>TEMPORAL DRIFT ESTIMATIONS (TRAIN VS TEST)</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_drift.head(15))}
            </div>
        </div>
    </div>

    <div class="secondary-panel-grid">
        <div class="hud-panel" style="grid-column: span 2;">
            <h2>DAILY ANOMALIES</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_anomaly[df_anomaly['anomaly_flag'] == 1])}
            </div>
        </div>
    </div>

    <script>
        const imagePaths = [
            "plots/transaction_timeline.png",
            "plots/hourly_distribution.png",
            "plots/weekly_trend.png",
            "plots/drift_analysis.png"
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

        with (report_dir / "timeseries_analysis_report.html").open("w") as f:
            f.write(html_template)
        logger.info("Compiled Timeseries HTML dashboard saved to: %s", report_dir / "timeseries_analysis_report.html")
