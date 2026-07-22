# ruff: noqa: E501
"""Identity Feature Analysis engine — Part 3.9 IEEE-CIS Fraud Detection EDA."""

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


def _parse_browser(val: Any) -> str:
    """Helper to parse raw browser names into clean families."""
    if not isinstance(val, str):
        return "Unknown"
    v = val.lower()
    if "chrome" in v:
        return "Chrome"
    elif "safari" in v:
        return "Safari"
    elif "ie" in v or "trident" in v:
        return "IE"
    elif "firefox" in v:
        return "Firefox"
    elif "edge" in v:
        return "Edge"
    elif "opera" in v:
        return "Opera"
    return "Other"


def _parse_os(val: Any) -> str:
    """Helper to parse raw OS names into clean families."""
    if not isinstance(val, str):
        return "Unknown"
    v = val.lower()
    if "windows" in v:
        return "Windows"
    elif "mac" in v:
        return "macOS"
    elif "ios" in v or "ipad" in v or "iphone" in v:
        return "iOS"
    elif "android" in v:
        return "Android"
    elif "linux" in v:
        return "Linux"
    return "Other"


class IdentityFeatureAnalyzer:
    """Comprehensive analysis engine for IEEE-CIS Identity Features.

    Sub-modules:
    - 3.9.4  Identity Feature Inventory
    - 3.9.5  id_01 - id_38 Analysis
    - 3.9.6  DeviceType Analysis
    - 3.9.7  DeviceInfo Analysis
    - 3.9.8  Browser Analysis
    - 3.9.9  Operating System Analysis
    - 3.9.10 Identity Availability Analysis
    - 3.9.11 Identity Missingness & Fraud Analysis
    - 3.9.12 Identity Interaction Analysis
    - 3.9.13 Identity Risk Profiling
    - 3.9.14 Feature Engineering Opportunities
    """

    _NUM_IDS = [f"id_{i:02d}" for i in range(1, 12)]
    _CAT_IDS = [f"id_{i:02d}" for i in range(12, 39)]

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
        logger.info("IdentityFeatureAnalyzer initialized. Train rows: %d", len(df_train))

    # ------------------------------------------------------------------ #
    # Orchestrator                                                         #
    # ------------------------------------------------------------------ #

    def analyze_all(self, report_dir: Path) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = report_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("--- Stage 3.9: Identity Feature Analysis ---")

        df_inv = self.analyze_identity_inventory(report_dir)
        df_feats = self.analyze_id_features(report_dir)
        df_dev_type = self.analyze_device_type(report_dir)
        df_dev_info = self.analyze_device_info(report_dir)
        df_browser = self.analyze_browsers(report_dir)
        df_os = self.analyze_operating_systems(report_dir)
        df_avail = self.analyze_identity_availability(report_dir)
        df_missing = self.analyze_missingness_fraud(report_dir)
        df_xfeat = self.analyze_interactions(report_dir)
        df_risk = self.analyze_risk_profiles(report_dir)
        df_recs = self.generate_feature_engineering_recommendations(report_dir)

        self.generate_plots(plots_dir)

        summary = self._build_summary(df_inv, df_dev_type, df_browser, df_risk)

        self.compile_html_dashboard(
            report_dir=report_dir,
            summary=summary,
            df_inv=df_inv,
            df_feats=df_feats,
            df_dev_type=df_dev_type,
            df_dev_info=df_dev_info,
            df_browser=df_browser,
            df_os=df_os,
            df_avail=df_avail,
            df_missing=df_missing,
            df_xfeat=df_xfeat,
            df_risk=df_risk,
            df_recs=df_recs,
        )

        with (report_dir / "identity_analysis.json").open("w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Identity Feature Analysis complete. Reports → %s", report_dir)

    # ------------------------------------------------------------------ #
    # 3.9.4 Identity Feature Inventory                                    #
    # ------------------------------------------------------------------ #

    def analyze_identity_inventory(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing Identity Feature Inventory...")
        all_cols = self._NUM_IDS + self._CAT_IDS + ["DeviceType", "DeviceInfo"]
        records: list[dict[str, Any]] = []

        for col in all_cols:
            if col not in self.df_train.columns:
                continue
            series = self.df_train[col]
            missing_pct = float(series.isna().mean() * 100)
            cardinality = int(series.nunique(dropna=True))
            dtype = str(series.dtype)

            if col in self._NUM_IDS:
                family = "Numerical"
            elif col in self._CAT_IDS:
                family = "Categorical"
            elif col in ["DeviceType", "DeviceInfo"]:
                family = "Device"
            else:
                family = "Other"

            records.append({
                "feature": col,
                "data_type": dtype,
                "missing_pct": round(missing_pct, 4),
                "cardinality": cardinality,
                "feature_family": family,
                "availability": "Present" if missing_pct < 100 else "Absent"
            })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "identity_feature_inventory.csv", index=False)

        # Write metadata JSON
        meta = {
            "numerical_count": len([r for r in records if r["feature_family"] == "Numerical"]),
            "categorical_count": len([r for r in records if r["feature_family"] == "Categorical"]),
            "device_count": len([r for r in records if r["feature_family"] == "Device"]),
            "total_identity_features": len(records),
        }
        with (report_dir / "identity_metadata.json").open("w") as f:
            json.dump(meta, f, indent=2)

        return df

    # ------------------------------------------------------------------ #
    # 3.9.5 id_01 - id_38 Feature Analysis                               #
    # ------------------------------------------------------------------ #

    def analyze_id_features(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing individual id_01 to id_38 features...")
        records: list[dict[str, Any]] = []

        # Analyze Numerical id_01 - id_11
        for col in self._NUM_IDS:
            if col not in self.df_train.columns:
                continue
            s = self.df_train[col].dropna()
            if s.empty:
                continue

            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = int(((s < lower_bound) | (s > upper_bound)).sum())
            outlier_pct = float(outliers / len(s) * 100) if len(s) > 0 else 0.0

            # Fraud rate where this id exists vs missing
            if self.target_col in self.df_train.columns:
                f_present = float(self.df_train.loc[self.df_train[col].notna(), self.target_col].mean() * 100)
                f_missing = float(self.df_train.loc[self.df_train[col].isna(), self.target_col].mean() * 100)
            else:
                f_present = f_missing = float("nan")

            records.append({
                "feature": col,
                "family": "Numerical",
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "skewness": round(float(s.skew()), 4) if len(s) > 2 else 0.0,
                "outlier_pct": round(outlier_pct, 4),
                "fraud_rate_present": round(f_present, 4),
                "fraud_rate_missing": round(f_missing, 4),
            })

        # Analyze Categorical id_12 - id_38
        for col in self._CAT_IDS:
            if col not in self.df_train.columns:
                continue
            s = self.df_train[col].dropna()
            if s.empty:
                continue

            vc = s.value_counts()
            dom_pct = float(vc.iloc[0] / len(s) * 100) if len(vc) > 0 else 0.0
            rare_cats = len(vc[vc / len(s) < 0.01])

            # Fraud rate where this id exists vs missing
            if self.target_col in self.df_train.columns:
                f_present = float(self.df_train.loc[self.df_train[col].notna(), self.target_col].mean() * 100)
                f_missing = float(self.df_train.loc[self.df_train[col].isna(), self.target_col].mean() * 100)
            else:
                f_present = f_missing = float("nan")

            records.append({
                "feature": col,
                "family": "Categorical",
                "mean": float("nan"),
                "std": float("nan"),
                "skewness": float("nan"),
                "outlier_pct": float("nan"),
                "fraud_rate_present": round(f_present, 4),
                "fraud_rate_missing": round(f_missing, 4),
            })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "identity_feature_analysis.csv", index=False)
        return df

    # ------------------------------------------------------------------ #
    # 3.9.6 DeviceType Analysis                                           #
    # ------------------------------------------------------------------ #

    def analyze_device_type(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing DeviceType...")
        col = "DeviceType"
        if col not in self.df_train.columns:
            return pd.DataFrame()

        df = self.df_train.copy()
        df[col] = df[col].astype(str).replace("nan", "Missing")

        grp = df.groupby(col).agg(
            count=(self.target_col if self.target_col in df.columns else col, "count"),
            fraud_count=(self.target_col, "sum") if self.target_col in df.columns else (col, "count"),
        ).reset_index()

        grp["percentage"] = (grp["count"] / len(df) * 100).round(4)
        if self.target_col in df.columns:
            grp["fraud_rate"] = (grp["fraud_count"] / grp["count"] * 100).round(4)
            grp["relative_risk"] = (grp["fraud_rate"] / (self._global_fraud_rate * 100)).round(4)
        else:
            grp["fraud_rate"] = 0.0
            grp["relative_risk"] = 1.0

        grp.to_csv(report_dir / "device_type_analysis.csv", index=False)
        return grp

    # ------------------------------------------------------------------ #
    # 3.9.7 DeviceInfo Analysis                                           #
    # ------------------------------------------------------------------ #

    def analyze_device_info(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing DeviceInfo...")
        col = "DeviceInfo"
        if col not in self.df_train.columns:
            return pd.DataFrame()

        df = self.df_train.copy()
        df[col] = df[col].astype(str).replace("nan", "Missing")

        grp = df.groupby(col).agg(
            count=(self.target_col if self.target_col in df.columns else col, "count"),
            fraud_count=(self.target_col, "sum") if self.target_col in df.columns else (col, "count"),
        ).reset_index()

        grp["percentage"] = (grp["count"] / len(df) * 100).round(4)
        if self.target_col in df.columns:
            grp["fraud_rate"] = (grp["fraud_count"] / grp["count"] * 100).round(4)
            grp["relative_risk"] = (grp["fraud_rate"] / (self._global_fraud_rate * 100)).round(4)
        else:
            grp["fraud_rate"] = 0.0
            grp["relative_risk"] = 1.0

        # Sort by frequency but capture high device variability
        grp = grp.sort_values(by="count", ascending=False).reset_index(drop=True)
        grp.to_csv(report_dir / "device_info_analysis.csv", index=False)
        return grp

    # ------------------------------------------------------------------ #
    # 3.9.8 Browser Analysis                                              #
    # ------------------------------------------------------------------ #

    def analyze_browsers(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing Browser (id_31)...")
        col = "id_31"
        if col not in self.df_train.columns:
            return pd.DataFrame()

        df = self.df_train.copy()
        df[col] = df[col].astype(str).replace("nan", "Unknown")
        df["browser_family"] = df[col].apply(_parse_browser)

        grp = df.groupby("browser_family").agg(
            count=(self.target_col if self.target_col in df.columns else "browser_family", "count"),
            fraud_count=(self.target_col, "sum") if self.target_col in df.columns else ("browser_family", "count"),
        ).reset_index()

        grp["percentage"] = (grp["count"] / len(df) * 100).round(4)
        if self.target_col in df.columns:
            grp["fraud_rate"] = (grp["fraud_count"] / grp["count"] * 100).round(4)
            grp["relative_risk"] = (grp["fraud_rate"] / (self._global_fraud_rate * 100)).round(4)
        else:
            grp["fraud_rate"] = 0.0
            grp["relative_risk"] = 1.0

        grp = grp.sort_values(by="count", ascending=False).reset_index(drop=True)
        grp.to_csv(report_dir / "browser_analysis.csv", index=False)
        return grp

    # ------------------------------------------------------------------ #
    # 3.9.9 Operating System Analysis                                     #
    # ------------------------------------------------------------------ #

    def analyze_operating_systems(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing OS (id_30)...")
        col = "id_30"
        if col not in self.df_train.columns:
            return pd.DataFrame()

        df = self.df_train.copy()
        df[col] = df[col].astype(str).replace("nan", "Unknown")
        df["os_family"] = df[col].apply(_parse_os)

        grp = df.groupby("os_family").agg(
            count=(self.target_col if self.target_col in df.columns else "os_family", "count"),
            fraud_count=(self.target_col, "sum") if self.target_col in df.columns else ("os_family", "count"),
        ).reset_index()

        grp["percentage"] = (grp["count"] / len(df) * 100).round(4)
        if self.target_col in df.columns:
            grp["fraud_rate"] = (grp["fraud_count"] / grp["count"] * 100).round(4)
            grp["relative_risk"] = (grp["fraud_rate"] / (self._global_fraud_rate * 100)).round(4)
        else:
            grp["fraud_rate"] = 0.0
            grp["relative_risk"] = 1.0

        grp = grp.sort_values(by="count", ascending=False).reset_index(drop=True)
        grp.to_csv(report_dir / "os_analysis.csv", index=False)
        return grp

    # ------------------------------------------------------------------ #
    # 3.9.10 Identity Availability Analysis                               #
    # ------------------------------------------------------------------ #

    def analyze_identity_availability(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing Identity Availability...")
        all_cols = self._NUM_IDS + self._CAT_IDS + ["DeviceType", "DeviceInfo"]
        avail_cols = [c for c in all_cols if c in self.df_train.columns]

        if not avail_cols:
            return pd.DataFrame()

        df = self.df_train[avail_cols].copy()
        present_count = df.notna().sum(axis=1)
        pct_present = (present_count / len(avail_cols) * 100).round(4)

        # Categorize
        cats = pd.cut(
            pct_present,
            bins=[-0.1, 0, 30, 70, 100.1],
            labels=["None", "Low", "Medium", "High"],
        )

        df_avail = pd.DataFrame({
            "present_features": present_count,
            "completeness_pct": pct_present,
            "avail_category": cats,
        })
        if self.target_col in self.df_train.columns:
            df_avail[self.target_col] = self.df_train[self.target_col]

        grp = df_avail.groupby("avail_category", observed=True).agg(
            count=(self.target_col if self.target_col in df_avail.columns else "avail_category", "count"),
            fraud_count=(self.target_col, "sum") if self.target_col in df_avail.columns else ("avail_category", "count"),
        ).reset_index()

        grp["percentage"] = (grp["count"] / len(df_avail) * 100).round(4)
        if self.target_col in df_avail.columns:
            grp["fraud_rate"] = (grp["fraud_count"] / grp["count"] * 100).round(4)
            grp["relative_risk"] = (grp["fraud_rate"] / (self._global_fraud_rate * 100)).round(4)
        else:
            grp["fraud_rate"] = 0.0
            grp["relative_risk"] = 1.0

        grp.to_csv(report_dir / "identity_availability.csv", index=False)
        return grp

    # ------------------------------------------------------------------ #
    # 3.9.11 Identity Missingness & Fraud Analysis                        #
    # ------------------------------------------------------------------ #

    def analyze_missingness_fraud(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing Identity Missingness vs Fraud...")
        all_cols = self._NUM_IDS + self._CAT_IDS + ["DeviceType", "DeviceInfo"]
        avail_cols = [c for c in all_cols if c in self.df_train.columns]

        records: list[dict[str, Any]] = []
        for col in avail_cols:
            series = self.df_train[col]
            missing_pct = float(series.isna().mean() * 100)

            if self.target_col in self.df_train.columns:
                f_present = float(self.df_train.loc[series.notna(), self.target_col].mean() * 100)
                f_missing = float(self.df_train.loc[series.isna(), self.target_col].mean() * 100)
                relative_risk_diff = round(f_missing / f_present, 4) if f_present > 0 else 1.0
            else:
                f_present = f_missing = float("nan")
                relative_risk_diff = 1.0

            records.append({
                "feature": col,
                "missing_pct": round(missing_pct, 4),
                "fraud_rate_present": round(f_present, 4),
                "fraud_rate_missing": round(f_missing, 4),
                "missingness_relative_risk": relative_risk_diff,
            })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "identity_missing_analysis.csv", index=False)
        return df

    # ------------------------------------------------------------------ #
    # 3.9.12 Identity Interaction Analysis                                #
    # ------------------------------------------------------------------ #

    def analyze_interactions(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Analyzing identity cross-feature interactions...")
        df = self.df_train.copy()
        if "id_31" in df.columns:
            df["id_31"] = df["id_31"].astype(str).replace("nan", "Unknown")
        if "id_30" in df.columns:
            df["id_30"] = df["id_30"].astype(str).replace("nan", "Unknown")
        if "DeviceType" in df.columns:
            df["DeviceType"] = df["DeviceType"].astype(str).replace("nan", "Unknown")

        df["browser_family"] = df["id_31"].apply(_parse_browser) if "id_31" in df.columns else "Unknown"
        df["os_family"] = df["id_30"].apply(_parse_os) if "id_30" in df.columns else "Unknown"


        pairs = [
            ("DeviceType", "browser_family"),
            ("DeviceType", "os_family"),
            ("browser_family", "os_family"),
        ]

        all_rows: list[dict[str, Any]] = []
        for a, b in pairs:
            if a not in df.columns or b not in df.columns:
                continue
            sub = df[[a, b] + ([self.target_col] if self.target_col in df.columns else [])].dropna()
            if sub.empty:
                continue

            grp = sub.groupby([a, b], observed=True).agg(
                transaction_count=(self.target_col if self.target_col in sub.columns else a, "count"),
                fraud_count=(self.target_col, "sum") if self.target_col in sub.columns else (a, "count"),
            ).reset_index()

            if self.target_col in sub.columns:
                grp["fraud_rate"] = (grp["fraud_count"] / grp["transaction_count"] * 100).round(4)
                grp["relative_risk"] = (grp["fraud_rate"] / (self._global_fraud_rate * 100)).round(4)
            else:
                grp["fraud_rate"] = 0.0
                grp["relative_risk"] = 1.0

            grp.insert(0, "interaction", f"{a}×{b}")
            all_rows.append(grp)

        df_out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        df_out = df_out.sort_values("transaction_count", ascending=False).reset_index(drop=True)
        df_out.to_csv(report_dir / "identity_interactions.csv", index=False)
        return df_out

    # ------------------------------------------------------------------ #
    # 3.9.13 Identity Risk Profiling                                      #
    # ------------------------------------------------------------------ #

    def analyze_risk_profiles(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Building identity risk profiles...")
        df = self.df_train.copy()
        if "id_31" in df.columns:
            df["id_31"] = df["id_31"].astype(str).replace("nan", "Unknown")
        if "id_30" in df.columns:
            df["id_30"] = df["id_30"].astype(str).replace("nan", "Unknown")
        if "DeviceType" in df.columns:
            df["DeviceType"] = df["DeviceType"].astype(str).replace("nan", "Unknown")

        df["browser_family"] = df["id_31"].apply(_parse_browser) if "id_31" in df.columns else "Unknown"
        df["os_family"] = df["id_30"].apply(_parse_os) if "id_30" in df.columns else "Unknown"


        profile_cols = ["DeviceType", "browser_family", "os_family"]
        has_target = self.target_col in df.columns

        sub = df[profile_cols + ([self.target_col] if has_target else [])]

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
        grp.to_csv(report_dir / "identity_risk_profiles.csv", index=False)
        return grp

    # ------------------------------------------------------------------ #
    # 3.9.14 Feature Recommendations                                       #
    # ------------------------------------------------------------------ #

    def generate_feature_engineering_recommendations(self, report_dir: Path) -> pd.DataFrame:
        logger.info("Generating feature engineering recommendations...")
        records = [
            {"feature_group": "Identity Completeness", "engineered_feature": "identity_completeness_score",
             "method": "Count non-null identity fields", "rationale": "High missingness tracks transactions where cards are present or absent; completeness score acts as validation proxy.", "priority": "High"},
            {"feature_group": "Identity Completeness", "engineered_feature": "identity_missing_indicator_id_01_id_11",
             "method": "Vector OR-pooling", "rationale": "Flag transactions missing critical customer attributes entirely.", "priority": "High"},
            {"feature_group": "Device Features", "engineered_feature": "device_info_freq_encoding",
             "method": "Frequency encoding (DeviceInfo)", "rationale": "DeviceInfo has huge cardinality (>1000 brands). Frequency encodes market popularity.", "priority": "High"},
            {"feature_group": "Device Features", "engineered_feature": "device_info_risk_score",
             "method": "Target encoding (DeviceInfo)", "rationale": "Capture risk values for specialized models.", "priority": "High"},
            {"feature_group": "Device Features", "engineered_feature": "is_desktop_device",
             "method": "Binary indicator (DeviceType == desktop)", "rationale": "Distinct usage rates and fraud rate levels between desktops and mobile devices.", "priority": "Medium"},
            {"feature_group": "Browser Info", "engineered_feature": "browser_family",
             "method": "Keyword lookup extraction", "rationale": "Group high-cardinality version lines (e.g., chrome 63.0.x) to a browser family.", "priority": "High"},
            {"feature_group": "Browser Info", "engineered_feature": "browser_rare_indicator",
             "method": "Frequency cut-off flag (<0.1%)", "rationale": "Attackers often spoof obscure user agents. Flagging rare browsers maps anomaly signals.", "priority": "Medium"},
            {"feature_group": "Operating System", "engineered_feature": "os_family",
             "method": "Keyword lookup extraction", "rationale": "Cluster OS names into Windows, macOS, OS X, iOS, Android, Linux.", "priority": "High"},
            {"feature_group": "Operating System", "engineered_feature": "os_risk_score",
             "method": "Target encoding (id_30)", "rationale": "Older platform versions exhibit higher vulnerability rates.", "priority": "Medium"},
            {"feature_group": "Cross-Features", "engineered_feature": "devtype_browser_combo",
             "method": "Concatenate strings", "rationale": "Flag contradictory combinations (e.g. desktop + mobile safari) which signal browser spoofing.", "priority": "High"},
            {"feature_group": "Cross-Features", "engineered_feature": "browser_os_combo",
             "method": "Concatenate strings", "rationale": "Interaction encodes detailed customer device environment profile.", "priority": "High"},
            {"feature_group": "Cross-Features", "engineered_feature": "id_01_amt_ratio",
             "method": "Numerical id_01 / TransactionAmt", "rationale": "Ratio of numeric identity metrics to amount highlights abnormal values.", "priority": "Medium"},
        ]

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "identity_feature_recommendations.csv", index=False)
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

        # 1. DeviceType distribution
        if "DeviceType" in self.df_train.columns:
            try:
                vc = self.df_train["DeviceType"].astype(str).replace("nan", "Missing").value_counts()
                fig, ax = plt.subplots(figsize=(8, 4), facecolor=_bg)
                ax.set_facecolor(_bg)
                ax.bar(vc.index, vc.values, color=_fg, edgecolor="#ffffff11")
                ax.set_title("[DEVICE] DeviceType Share Including Missing", color="#fff", fontsize=10)
                ax.tick_params(colors=_fg)
                ax.spines[:].set_color("#ffffff11")
                plt.tight_layout()
                _save(fig, "devicetype_distribution.png")
            except Exception as exc:
                logger.warning("DeviceType plot failed: %s", exc)

        # 2. Browser distribution
        if "id_31" in self.df_train.columns:
            try:
                browsers = self.df_train["id_31"].apply(_parse_browser).value_counts()
                fig, ax = plt.subplots(figsize=(8, 4), facecolor=_bg)
                ax.set_facecolor(_bg)
                ax.bar(browsers.index, browsers.values, color=_fg, edgecolor="#ffffff11")
                ax.set_title("[BROWSER] Major Browser Families", color="#fff", fontsize=10)
                ax.tick_params(colors=_fg)
                ax.spines[:].set_color("#ffffff11")
                plt.tight_layout()
                _save(fig, "browser_family_distribution.png")
            except Exception as exc:
                logger.warning("Browser plot failed: %s", exc)

        # 3. OS distribution
        if "id_30" in self.df_train.columns:
            try:
                os = self.df_train["id_30"].apply(_parse_os).value_counts()
                fig, ax = plt.subplots(figsize=(8, 4), facecolor=_bg)
                ax.set_facecolor(_bg)
                ax.bar(os.index, os.values, color=_fg, edgecolor="#ffffff11")
                ax.set_title("[OS] Major Operating Systems", color="#fff", fontsize=10)
                ax.tick_params(colors=_fg)
                ax.spines[:].set_color("#ffffff11")
                plt.tight_layout()
                _save(fig, "os_family_distribution.png")
            except Exception as exc:
                logger.warning("OS plot failed: %s", exc)

        # 4. Identity Availability vs Fraud Rate
        if "DeviceType" in self.df_train.columns and self.target_col in self.df_train.columns:
            try:
                df = self.df_train.copy()
                df["has_identity"] = df["DeviceType"].notna().map({True: "Identity Present", False: "Identity Missing"})
                fr = df.groupby("has_identity")[self.target_col].mean() * 100
                fig, ax = plt.subplots(figsize=(6, 4), facecolor=_bg)
                ax.set_facecolor(_bg)
                ax.bar(fr.index, fr.values, color=[_red if "Missing" in i else _fg for i in fr.index], edgecolor="#ffffff11")
                ax.set_title("[FRAUD] Fraud Rate: Identity Present vs Missing", color=_red, fontsize=10)
                ax.tick_params(colors=_fg)
                ax.spines[:].set_color("#ffffff11")
                plt.tight_layout()
                _save(fig, "availability_vs_fraud.png")
            except Exception as exc:
                logger.warning("Availability vs Fraud plot failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Summary Builder                                                    #
    # ------------------------------------------------------------------ #

    def _build_summary(
        self,
        df_inv: pd.DataFrame,
        df_dev_type: pd.DataFrame,
        df_browser: pd.DataFrame,
        df_risk: pd.DataFrame,
    ) -> dict[str, Any]:
        has_id_count = int(self.df_train["DeviceType"].notna().sum())
        missing_id_count = len(self.df_train) - has_id_count

        high_risk_profiles = 0
        if not df_risk.empty and "risk_label" in df_risk.columns:
            high_risk_profiles = int((df_risk["risk_label"].isin(["High", "Critical"])).sum())

        return {
            "total_transactions": int(len(self.df_train)),
            "global_fraud_rate_pct": round(self._global_fraud_rate * 100, 4),
            "identity_present_count": has_id_count,
            "identity_missing_count": missing_id_count,
            "identity_coverage_pct": round(has_id_count / len(self.df_train) * 100, 4) if len(self.df_train) > 0 else 0.0,
            "unique_browsers": int(df_browser["browser_family"].nunique()) if not df_browser.empty else 0,
            "unique_devices": int(self.df_train["DeviceInfo"].nunique()) if "DeviceInfo" in self.df_train.columns else 0,
            "high_risk_profiles": high_risk_profiles,
        }

    # ------------------------------------------------------------------ #
    # HTML compilation                                                   #
    # ------------------------------------------------------------------ #

    def compile_html_dashboard(
        self,
        report_dir: Path,
        summary: dict[str, Any],
        df_inv: pd.DataFrame,
        df_feats: pd.DataFrame,
        df_dev_type: pd.DataFrame,
        df_dev_info: pd.DataFrame,
        df_browser: pd.DataFrame,
        df_os: pd.DataFrame,
        df_avail: pd.DataFrame,
        df_missing: pd.DataFrame,
        df_xfeat: pd.DataFrame,
        df_risk: pd.DataFrame,
        df_recs: pd.DataFrame,
    ) -> None:
        logger.info("Compiling Identity HTML Dashboard...")

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
    <title>IEEE-CIS IDENTITY DIAGNOSTICS</title>
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
        <h1>IEEE-CIS IDENTITY DIAGNOSTICS</h1>
        <div class="status-pill">IDENTITY ANALYZER OK</div>
    </header>

    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-title">TOTAL TRANSACTIONS</div>
            <div class="metric-value">{summary['total_transactions']:,}</div>
            <div class="metric-desc">Dataset size analyzed</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">IDENTITY COVERAGE</div>
            <div class="metric-value">{summary['identity_coverage_pct']:.4f}%</div>
            <div class="metric-desc">Transactions with identity records</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">UNIQUE DEVICE DESIGNS</div>
            <div class="metric-value">{summary['unique_devices']:,}</div>
            <div class="metric-desc">Cardinality of DeviceInfo</div>
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
                <button class="carousel-btn active" onclick="switchSlide(0)">DeviceType Distribution</button>
                <button class="carousel-btn" onclick="switchSlide(1)">Browser Families</button>
                <button class="carousel-btn" onclick="switchSlide(2)">Operating Systems</button>
                <button class="carousel-btn" onclick="switchSlide(3)">Availability vs Fraud</button>
            </div>
            <div id="slide-0" class="carousel-slide active">
                <img src="plots/devicetype_distribution.png" alt="DeviceType distribution histograms">
            </div>
            <div id="slide-1" class="carousel-slide">
                <img src="plots/browser_family_distribution.png" alt="Major Browser Families">
            </div>
            <div id="slide-2" class="carousel-slide">
                <img src="plots/os_family_distribution.png" alt="Major Operating Systems">
            </div>
            <div id="slide-3" class="carousel-slide">
                <img src="plots/availability_vs_fraud.png" alt="Fraud rates by identity availability">
            </div>
        </div>
    </div>

    <!-- Section 2: Identification Inventory -->
    <div class="hud-section">
        <div class="section-header">Identity Variable Inventory</div>
        <div class="grid-split">
            <div class="hud-panel">
                <div class="hud-panel-title">Identity Features Inventory Listing</div>
                {_to_html(df_inv)}
            </div>
            <div class="hud-panel">
                <div class="hud-panel-title">Feature-wise Coverage & Statistics</div>
                {_to_html(df_feats)}
            </div>
        </div>
    </div>

    <!-- Section 3: Hardware & Browser Platforms -->
    <div class="hud-section">
        <div class="section-header">Device & Platform Characteristics</div>
        <div class="grid-split">
            <div class="hud-panel">
                <div class="hud-panel-title">DeviceType Analysis</div>
                {_to_html(df_dev_type)}
            </div>
            <div class="hud-panel">
                <div class="hud-panel-title">Browser Family (id_31) Analysis</div>
                {_to_html(df_browser)}
            </div>
        </div>
    </div>

    <!-- Section 4: Operating System & DeviceInfo Info -->
    <div class="hud-section">
        <div class="section-header">OS & Card Hardware Profiles</div>
        <div class="grid-split">
            <div class="hud-panel">
                <div class="hud-panel-title">Operating System Family (id_30) Analysis</div>
                {_to_html(df_os)}
            </div>
            <div class="hud-panel">
                <div class="hud-panel-title">DeviceInfo Frequencies (Top 15 Categories)</div>
                {_to_html(df_dev_info.head(15) if not df_dev_info.empty else df_dev_info)}
            </div>
        </div>
    </div>

    <!-- Section 5: Availability & Missingness Analysis -->
    <div class="hud-section">
        <div class="section-header">Completeness & Informative Missingness</div>
        <div class="grid-split">
            <div class="hud-panel">
                <div class="hud-panel-title">Transaction Identity Availability Classes</div>
                {_to_html(df_avail)}
            </div>
            <div class="hud-panel">
                <div class="hud-panel-title">Informative Missingness Relative Fraud Risks</div>
                {_to_html(df_missing)}
            </div>
        </div>
    </div>

    <!-- Section 6: Interactions & Profiles -->
    <div class="hud-section">
        <div class="section-header">Multi-variable Cross-Feature Interactions & Profiles</div>
        <div class="grid-split">
            <div class="hud-panel">
                <div class="hud-panel-title">Multi-variable Identity Level Interactions</div>
                {_to_html(df_xfeat.head(20) if not df_xfeat.empty else df_xfeat)}
            </div>
            <div class="hud-panel">
                <div class="hud-panel-title">Composite Identity Risk Behavioral Profiles</div>
                {_to_html(df_risk.head(20) if not df_risk.empty else df_risk)}
            </div>
        </div>
    </div>

    <!-- Section 7: Engineering Recommendations -->
    <div class="hud-section">
        <div class="section-header">Recommended Engineering Pipe Modifications</div>
        <div class="hud-panel">
            <div class="hud-panel-title">Feature Engineering Recommendations</div>
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
        dashboard_path = report_dir / "identity_analysis_report.html"
        dashboard_path.write_text(html_template)
        logger.info("Compiled Identity HTML dashboard saved to: %s", dashboard_path)
