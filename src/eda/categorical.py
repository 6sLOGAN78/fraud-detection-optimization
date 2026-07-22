# ruff: noqa: E501
"""Categorical Feature Analysis engine — Part 3.7 IEEE-CIS Fraud Detection EDA."""

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
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def classify_cardinality(n_unique: int) -> str:
    """Classifies feature cardinality into bands.

    Args:
        n_unique: Number of unique categories.

    Returns:
        Cardinality classification label.
    """
    if n_unique <= 2:
        return "Binary"
    if n_unique <= 10:
        return "Low"
    if n_unique <= 100:
        return "Medium"
    if n_unique <= 1000:
        return "High"
    return "Very High"


def classify_dominance(dominant_pct: float) -> str:
    """Classifies category dominance ratio.

    Args:
        dominant_pct: Percentage held by the most frequent category.

    Returns:
        Dominance classification string.
    """
    if dominant_pct >= 95.0:
        return "Near-Constant"
    if dominant_pct >= 80.0:
        return "Highly Dominant"
    if dominant_pct >= 60.0:
        return "Moderately Dominant"
    return "Balanced"


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Computes Total Variation (TV) distance between two distributions.

    Args:
        p: Probability vector from train.
        q: Probability vector from test.

    Returns:
        TV distance in [0, 1].
    """
    try:
        all_keys = set(np.concatenate([p.index.to_numpy(), q.index.to_numpy()]))
        p_aligned = p.reindex(list(all_keys), fill_value=0.0)
        q_aligned = q.reindex(list(all_keys), fill_value=0.0)
        return float(0.5 * np.sum(np.abs(p_aligned.to_numpy() - q_aligned.to_numpy())))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Main Analyzer Class
# ---------------------------------------------------------------------------


class CategoricalFeatureAnalyzer:
    """Performs comprehensive categorical diagnostics for IEEE-CIS Fraud Detection.

    Implements Part 3.7 analysis sub-modules:
    - 3.7.4 Categorical Feature Identification
    - 3.7.5 Frequency Analysis
    - 3.7.6 Cardinality Analysis
    - 3.7.7 Rare Category Analysis
    - 3.7.8 Category Imbalance Analysis
    - 3.7.9 Fraud Rate by Category
    - 3.7.10 Target Distribution Analysis
    - 3.7.11 Category Stability Analysis
    """

    #: Known IEEE-CIS categorical columns per domain specification
    _KNOWN_CAT_KEYWORDS = (
        "ProductCD", "card4", "card6",
        "P_emaildomain", "R_emaildomain",
        "DeviceType", "DeviceInfo",
        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    )

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_col: str = "isFraud",
        rare_threshold: float = 0.01,
    ) -> None:
        """Initializes CategoricalFeatureAnalyzer.

        Args:
            df_train: Merged training DataFrame.
            df_test: Merged test DataFrame.
            target_col: Target label column name.
            rare_threshold: Minimum frequency fraction to avoid rare classification.
        """
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.target_col = target_col
        self.rare_threshold = rare_threshold

        _exclude = {"TransactionID", "TransactionDT", self.target_col}

        # Auto-detect categorical columns using dtype heuristics + domain knowledge
        dtype_cats = {
            col for col in df_train.columns
            if col not in _exclude and df_train[col].dtype in ("object", "category")
        }

        # Also pick up integer id_xx columns that are semantically categorical
        id_cats = {
            col for col in df_train.columns
            if col not in _exclude
            and col.startswith("id_")
            and col not in dtype_cats
            and df_train[col].nunique() <= 50
        }

        m_cols = {
            col for col in df_train.columns
            if col not in _exclude
            and col.startswith("M")
            and df_train[col].dtype == "object"
        }

        self.categorical_cols: list[str] = sorted(dtype_cats | id_cats | m_cols)
        logger.info(
            "Identified %d categorical features for analysis.",
            len(self.categorical_cols),
        )

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def analyze_all(self, report_dir: Path) -> None:
        """Executes the full categorical analysis pipeline.

        Args:
            report_dir: Root output directory for all reports and assets.
        """
        report_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = report_dir / "freq_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("--- Stage 3.7: Categorical Feature Analysis ---")

        df_features = self.identify_categorical_features(report_dir)
        df_freq = self.analyze_frequency(report_dir)
        df_card = self.analyze_cardinality(report_dir)
        df_rare = self.analyze_rare_categories(report_dir)
        df_imb = self.analyze_category_imbalance(report_dir)
        df_fraud = self.analyze_fraud_rates(report_dir)
        df_target = self.analyze_target_distribution(report_dir)
        df_stab = self.analyze_category_stability(report_dir)
        df_enc = self.generate_encoding_recommendations(
            report_dir, df_card, df_rare, df_imb, df_fraud
        )

        self.generate_plots(report_dir, plots_dir)

        summary = self._build_summary(df_features, df_rare, df_card, df_stab)

        self.compile_html_dashboard(
            report_dir=report_dir,
            summary=summary,
            df_features=df_features,
            df_freq=df_freq,
            df_card=df_card,
            df_imb=df_imb,
            df_fraud=df_fraud,
            df_stab=df_stab,
            df_enc=df_enc,
        )

        summary_path = report_dir / "categorical_analysis.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Categorical Feature Analysis completed. Reports saved to %s", report_dir)

    # ------------------------------------------------------------------
    # 3.7.4 — Categorical Feature Identification
    # ------------------------------------------------------------------

    def identify_categorical_features(self, report_dir: Path) -> pd.DataFrame:
        """Profiles all categorical features with structural metadata.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame with per-feature identification metrics.
        """
        logger.info("Identifying categorical features...")
        records = []
        for col in self.categorical_cols:
            series = self.df_train[col]
            n_total = len(series)
            n_missing = int(series.isna().sum())
            missing_pct = round(n_missing / n_total * 100, 4) if n_total > 0 else 0.0
            n_unique = int(series.nunique())
            vc = series.value_counts(dropna=True)
            dominant = str(vc.index[0]) if len(vc) > 0 else "N/A"
            dominant_pct = round(vc.iloc[0] / (n_total - n_missing) * 100, 4) if len(vc) > 0 and (n_total - n_missing) > 0 else 0.0
            records.append({
                "feature": col,
                "dtype": str(series.dtype),
                "n_categories": n_unique,
                "missing_count": n_missing,
                "missing_pct": missing_pct,
                "dominant_category": dominant,
                "dominant_pct": dominant_pct,
            })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "categorical_features.csv", index=False)
        summary = {
            "total_categorical_features": int(len(df)),
            "features": df["feature"].tolist(),
        }
        with (report_dir / "categorical_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Categorical feature identification saved: %d features", len(df))
        return df

    # ------------------------------------------------------------------
    # 3.7.5 — Frequency Analysis
    # ------------------------------------------------------------------

    def analyze_frequency(self, report_dir: Path) -> pd.DataFrame:
        """Computes category occurrence frequencies for every categorical feature.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame with category-level frequency metrics.
        """
        logger.info("Running frequency analysis...")
        all_rows: list[dict[str, Any]] = []
        freq_summary: dict[str, Any] = {}

        for col in self.categorical_cols:
            series = self.df_train[col].dropna()
            n = len(series)
            if n == 0:
                continue
            vc = series.value_counts()
            vc_pct = (vc / n * 100).round(4)
            cum_pct = vc_pct.cumsum()

            freq_summary[col] = {
                "most_frequent": str(vc.index[0]),
                "most_frequent_count": int(vc.iloc[0]),
                "most_frequent_pct": float(vc_pct.iloc[0]),
                "least_frequent": str(vc.index[-1]),
                "least_frequent_count": int(vc.iloc[-1]),
                "least_frequent_pct": float(vc_pct.iloc[-1]),
            }

            for rank, (cat, cnt) in enumerate(vc.items(), start=1):
                all_rows.append({
                    "feature": col,
                    "category": str(cat),
                    "count": int(cnt),
                    "percentage": float(vc_pct[cat]),
                    "cumulative_pct": float(cum_pct[cat]),
                    "rank": rank,
                })

        df = pd.DataFrame(all_rows)
        df.to_csv(report_dir / "category_frequency.csv", index=False)
        with (report_dir / "frequency_summary.json").open("w") as f:
            json.dump(freq_summary, f, indent=2, default=str)
        logger.info("Frequency analysis complete: %d category records", len(df))
        return df

    # ------------------------------------------------------------------
    # 3.7.6 — Cardinality Analysis
    # ------------------------------------------------------------------

    def analyze_cardinality(self, report_dir: Path) -> pd.DataFrame:
        """Measures category uniqueness and classifies cardinality bands.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame with per-feature cardinality metrics.
        """
        logger.info("Running cardinality analysis...")
        records = []
        for col in self.categorical_cols:
            series = self.df_train[col].dropna()
            n = len(series)
            if n == 0:
                records.append({"feature": col, "n_unique": 0, "cardinality_ratio": 0.0,
                                 "max_freq": 0, "min_freq": 0, "cardinality_class": "N/A"})
                continue
            vc = series.value_counts()
            n_unique = int(vc.shape[0])
            records.append({
                "feature": col,
                "n_unique": n_unique,
                "cardinality_ratio": round(n_unique / n, 6),
                "max_freq": int(vc.iloc[0]),
                "min_freq": int(vc.iloc[-1]),
                "cardinality_class": classify_cardinality(n_unique),
            })

        df = pd.DataFrame(records).sort_values("n_unique", ascending=False).reset_index(drop=True)
        df.to_csv(report_dir / "cardinality_report.csv", index=False)
        logger.info("Cardinality analysis complete.")
        return df

    # ------------------------------------------------------------------
    # 3.7.7 — Rare Category Analysis
    # ------------------------------------------------------------------

    def analyze_rare_categories(self, report_dir: Path) -> pd.DataFrame:
        """Identifies categories below the rare_threshold frequency.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame with per-feature rare category counts and details.
        """
        logger.info("Running rare category analysis (threshold=%.2f%%)...", self.rare_threshold * 100)
        feature_rows: list[dict[str, Any]] = []
        rare_summary: dict[str, Any] = {}

        for col in self.categorical_cols:
            series = self.df_train[col].dropna()
            n = len(series)
            if n == 0:
                continue
            vc = series.value_counts()
            vc_pct = vc / n

            rare_mask = vc_pct < self.rare_threshold
            n_rare = int(rare_mask.sum())
            rare_count = int(vc[rare_mask].sum())
            rare_pct = round(rare_count / n * 100, 4)

            rare_summary[col] = {
                "n_rare_categories": n_rare,
                "rare_sample_count": rare_count,
                "rare_pct_of_total": rare_pct,
            }

            feature_rows.append({
                "feature": col,
                "n_rare_categories": n_rare,
                "rare_sample_count": rare_count,
                "rare_pct_of_total": rare_pct,
                "rare_categories": ", ".join(str(v) for v in vc[rare_mask].index[:10].tolist()),
            })

        df = pd.DataFrame(feature_rows).sort_values("n_rare_categories", ascending=False).reset_index(drop=True)
        df.to_csv(report_dir / "rare_categories.csv", index=False)
        with (report_dir / "rare_category_summary.json").open("w") as f:
            json.dump(rare_summary, f, indent=2, default=str)
        logger.info("Rare category analysis complete: %d features analyzed.", len(df))
        return df

    # ------------------------------------------------------------------
    # 3.7.8 — Category Imbalance Analysis
    # ------------------------------------------------------------------

    def analyze_category_imbalance(self, report_dir: Path) -> pd.DataFrame:
        """Evaluates intra-feature category imbalance using entropy and dominance.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame with imbalance metrics per feature.
        """
        logger.info("Running category imbalance analysis...")
        records = []
        for col in self.categorical_cols:
            series = self.df_train[col].dropna()
            n = len(series)
            if n == 0:
                continue
            vc = series.value_counts()
            probs = (vc / n).to_numpy()
            n_unique = len(probs)

            dominant_pct = round(float(probs[0] * 100), 4)
            smallest_pct = round(float(probs[-1] * 100), 4)
            dominance_ratio = round(float(probs[0] / probs[-1]), 4) if probs[-1] > 0 else float("inf")
            ent = round(float(scipy_entropy(probs, base=2)), 6) if n_unique > 1 else 0.0
            max_entropy = np.log2(n_unique) if n_unique > 1 else 1.0
            diversity_index = round(ent / max_entropy, 6) if max_entropy > 0 else 0.0

            records.append({
                "feature": col,
                "dominant_pct": dominant_pct,
                "smallest_pct": smallest_pct,
                "dominance_ratio": dominance_ratio,
                "entropy_bits": ent,
                "diversity_index": diversity_index,
                "dominance_class": classify_dominance(dominant_pct),
            })

        df = pd.DataFrame(records).sort_values("diversity_index").reset_index(drop=True)
        df.to_csv(report_dir / "category_imbalance.csv", index=False)
        logger.info("Category imbalance analysis complete.")
        return df

    # ------------------------------------------------------------------
    # 3.7.9 — Fraud Rate by Category
    # ------------------------------------------------------------------

    def analyze_fraud_rates(self, report_dir: Path) -> pd.DataFrame:
        """Computes fraud and legitimate rates for every category.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame with per-category fraud rate metrics.
        """
        if self.target_col not in self.df_train.columns:
            logger.warning("Target column '%s' missing. Skipping fraud rate analysis.", self.target_col)
            return pd.DataFrame()

        logger.info("Computing fraud rates by category...")
        global_fraud_rate = self.df_train[self.target_col].mean()
        all_rows: list[dict[str, Any]] = []

        for col in self.categorical_cols:
            sub = self.df_train[[col, self.target_col]].dropna(subset=[col])
            if sub.empty:
                continue
            grp = sub.groupby(col, observed=True)[self.target_col].agg(["sum", "count"]).reset_index()
            grp.columns = ["category", "fraud_count", "total_count"]
            grp["legit_count"] = grp["total_count"] - grp["fraud_count"]
            grp["fraud_rate"] = (grp["fraud_count"] / grp["total_count"] * 100).round(4)
            grp["relative_risk"] = (grp["fraud_rate"] / (global_fraud_rate * 100)).round(4)
            grp.insert(0, "feature", col)
            all_rows.append(grp)

        if not all_rows:
            return pd.DataFrame()

        df = pd.concat(all_rows, ignore_index=True)
        df = df.sort_values(["feature", "fraud_rate"], ascending=[True, False]).reset_index(drop=True)
        df.to_csv(report_dir / "fraud_rate_by_category.csv", index=False)
        logger.info("Fraud rate analysis complete: %d category records", len(df))
        return df

    # ------------------------------------------------------------------
    # 3.7.10 — Target Distribution Analysis
    # ------------------------------------------------------------------

    def analyze_target_distribution(self, report_dir: Path) -> pd.DataFrame:
        """Analyzes isFraud distribution across each categorical feature.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame with lift scores and fraud contribution per category.
        """
        if self.target_col not in self.df_train.columns:
            logger.warning("Target column missing. Skipping target distribution analysis.")
            return pd.DataFrame()

        logger.info("Computing target distributions by category...")
        total_fraud = self.df_train[self.target_col].sum()
        all_rows: list[dict[str, Any]] = []

        for col in self.categorical_cols:
            sub = self.df_train[[col, self.target_col]].dropna(subset=[col])
            if sub.empty:
                continue

            grp = (
                sub.groupby(col, observed=True)[self.target_col]
                .agg(fraud_pct=lambda x: x.mean() * 100,
                     fraud_count="sum",
                     total_count="count")
                .reset_index()
            )
            grp.columns = ["category", "fraud_pct", "fraud_count", "total_count"]
            grp["legit_pct"] = 100.0 - grp["fraud_pct"]
            grp["fraud_to_legit_ratio"] = (grp["fraud_pct"] / grp["legit_pct"].replace(0, np.nan)).round(4)
            global_rate_pct = self.df_train[self.target_col].mean() * 100
            grp["lift_score"] = (grp["fraud_pct"] / global_rate_pct).round(4)
            grp["contribution_to_total_fraud_pct"] = (grp["fraud_count"] / total_fraud * 100).round(4)
            grp.insert(0, "feature", col)
            grp["fraud_pct"] = grp["fraud_pct"].round(4)
            grp["legit_pct"] = grp["legit_pct"].round(4)
            all_rows.append(grp)

        if not all_rows:
            return pd.DataFrame()

        df = pd.concat(all_rows, ignore_index=True)
        df.to_csv(report_dir / "target_distribution_by_category.csv", index=False)
        logger.info("Target distribution analysis complete.")
        return df

    # ------------------------------------------------------------------
    # 3.7.11 — Category Stability Analysis
    # ------------------------------------------------------------------

    def analyze_category_stability(self, report_dir: Path) -> pd.DataFrame:
        """Compares category distributions between train and test datasets.

        Args:
            report_dir: Output base directory.

        Returns:
            DataFrame with per-feature train/test distribution drift metrics.
        """
        logger.info("Running category stability analysis...")
        records = []
        drift_report: dict[str, Any] = {}

        for col in self.categorical_cols:
            tr = self.df_train[col].dropna()
            te = self.df_test[col].dropna() if col in self.df_test.columns else pd.Series([], dtype="object")

            tr_cats = set(tr.unique())
            te_cats = set(te.unique())

            new_in_test = te_cats - tr_cats
            missing_in_test = tr_cats - te_cats

            tr_dist = tr.value_counts(normalize=True)
            te_dist = te.value_counts(normalize=True) if len(te) > 0 else pd.Series(dtype="float64")
            tv = total_variation_distance(tr_dist, te_dist)

            records.append({
                "feature": col,
                "train_categories": len(tr_cats),
                "test_categories": len(te_cats),
                "new_categories_in_test": len(new_in_test),
                "missing_categories_in_test": len(missing_in_test),
                "tv_distance": round(tv, 6),
                "drift_level": "High" if tv > 0.15 else ("Moderate" if tv > 0.05 else "Low"),
                "new_category_examples": ", ".join(str(c) for c in list(new_in_test)[:5]),
            })
            drift_report[col] = {
                "tv_distance": round(tv, 6) if not np.isnan(tv) else None,
                "new_in_test": [str(c) for c in list(new_in_test)[:10]],
                "missing_in_test": [str(c) for c in list(missing_in_test)[:10]],
            }

        df = pd.DataFrame(records).sort_values("tv_distance", ascending=False).reset_index(drop=True)
        df.to_csv(report_dir / "category_stability.csv", index=False)
        with (report_dir / "category_drift_report.json").open("w") as f:
            json.dump(drift_report, f, indent=2, default=str)
        logger.info("Category stability analysis complete.")
        return df

    # ------------------------------------------------------------------
    # Encoding Recommendations
    # ------------------------------------------------------------------

    def generate_encoding_recommendations(
        self,
        report_dir: Path,
        df_card: pd.DataFrame,
        df_rare: pd.DataFrame,
        df_imb: pd.DataFrame,
        df_fraud: pd.DataFrame,
    ) -> pd.DataFrame:
        """Derives encoding strategy per feature based on diagnostic findings.

        Args:
            report_dir: Output base directory.
            df_card: Cardinality report DataFrame.
            df_rare: Rare category report DataFrame.
            df_imb: Category imbalance report DataFrame.
            df_fraud: Fraud rate by category DataFrame.

        Returns:
            DataFrame with encoding recommendation per feature.
        """
        logger.info("Generating encoding recommendations...")

        card_map = df_card.set_index("feature")["cardinality_class"].to_dict() if not df_card.empty else {}
        rare_map = df_rare.set_index("feature")["n_rare_categories"].to_dict() if not df_rare.empty else {}
        imb_map = df_imb.set_index("feature")["diversity_index"].to_dict() if not df_imb.empty else {}

        # High-risk features (max lift > 3) get target encoding signal
        high_risk_feats: set[str] = set()
        if not df_fraud.empty and "lift_score" not in df_fraud.columns and "fraud_rate" in df_fraud.columns:
            for feat, grp in df_fraud.groupby("feature"):
                if grp["relative_risk"].max() > 3.0:
                    high_risk_feats.add(str(feat))
        elif not df_fraud.empty and "relative_risk" in df_fraud.columns:
            for feat, grp in df_fraud.groupby("feature"):
                if grp["relative_risk"].max() > 3.0:
                    high_risk_feats.add(str(feat))

        records = []
        for col in self.categorical_cols:
            card_cls = card_map.get(col, "Unknown")
            n_rare = rare_map.get(col, 0)
            diversity = imb_map.get(col, 1.0)

            if card_cls == "Binary":
                strategy = "Label Encoding (Binary)"
                reason = "Only 2 categories; direct binary mapping."
            elif card_cls == "Low" and col in high_risk_feats:
                strategy = "Target Encoding"
                reason = "Low cardinality + high fraud signal; target encoding retains risk info."
            elif card_cls in ("Low", "Medium") and n_rare == 0:
                strategy = "One-Hot Encoding"
                reason = "Low/medium cardinality with no rare categories; OHE is safe."
            elif card_cls in ("Low", "Medium") and n_rare > 0:
                strategy = "One-Hot Encoding with Rare Grouping"
                reason = f"{n_rare} rare categories; group to 'Other' then OHE."
            elif card_cls == "High":
                strategy = "Frequency Encoding"
                reason = "High cardinality; frequency encoding avoids dimensionality explosion."
            elif card_cls == "Very High":
                strategy = "Target Encoding / Hashing"
                reason = "Very high cardinality; hashing trick or target encoding required."
            else:
                strategy = "Frequency Encoding"
                reason = "Fallback; frequency encoding is memory-safe."

            if diversity < 0.1:
                strategy += " + Near-Constant Flag"
                reason += " Near-constant — consider dropping."

            records.append({
                "feature": col,
                "cardinality_class": card_cls,
                "n_rare_categories": int(n_rare),
                "diversity_index": round(diversity, 4),
                "high_fraud_signal": col in high_risk_feats,
                "recommended_encoding": strategy,
                "reason": reason,
            })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "encoding_recommendations.csv", index=False)
        logger.info("Encoding recommendations generated for %d features.", len(df))
        return df

    # ------------------------------------------------------------------
    # Plot Generation
    # ------------------------------------------------------------------

    def generate_plots(self, report_dir: Path, plots_dir: Path) -> None:
        """Generates frequency bar charts and fraud-rate plots for key features.

        Args:
            report_dir: Root report directory.
            plots_dir: Plot output sub-directory.
        """
        logger.info("Generating categorical visualizations...")

        plot_cols = [
            c for c in self.categorical_cols
            if self.df_train[c].nunique() <= 30
        ][:12]  # Cap at 12 features for visual clarity

        has_target = self.target_col in self.df_train.columns
        plt.style.use("dark_background")

        for col in plot_cols:
            try:
                vc = self.df_train[col].value_counts().head(15)
                fig, ax = plt.subplots(figsize=(9, 4), facecolor="#06070b")
                ax.set_facecolor("#06070b")
                bars = ax.bar(
                    range(len(vc)),
                    vc.values,
                    color="#8e97a4",
                    edgecolor="#ffffff22",
                    linewidth=0.5,
                )
                ax.set_xticks(range(len(vc)))
                ax.set_xticklabels(
                    [str(v)[:18] for v in vc.index],
                    rotation=40,
                    ha="right",
                    fontsize=7,
                    color="#8e97a4",
                )
                ax.set_ylabel("Count", color="#8e97a4", fontsize=9)
                ax.set_title(
                    f"[FREQ] {col} — Top{len(vc)} Categories",
                    color="#ffffff",
                    fontsize=10,
                    fontweight="bold",
                    pad=12,
                )
                ax.tick_params(axis="y", colors="#8e97a4")
                ax.spines[:].set_color("#ffffff11")
                plt.tight_layout()
                fig.savefig(
                    plots_dir / f"{col}_freq.png",
                    dpi=110,
                    bbox_inches="tight",
                    facecolor="#06070b",
                )
                plt.close(fig)

                # Fraud-rate overlay if target available
                if has_target:
                    sub = self.df_train[[col, self.target_col]].dropna(subset=[col])
                    grp = sub.groupby(col, observed=True)[self.target_col].mean() * 100
                    grp = grp.sort_values(ascending=False).head(15)
                    fig2, ax2 = plt.subplots(figsize=(9, 4), facecolor="#06070b")
                    ax2.set_facecolor("#06070b")
                    bar_colors = [
                        "#d63031" if v > grp.mean() else "#8e97a4"
                        for v in grp.values
                    ]
                    ax2.bar(
                        range(len(grp)),
                        grp.values,
                        color=bar_colors,
                        edgecolor="#ffffff22",
                        linewidth=0.5,
                    )
                    ax2.axhline(
                        grp.mean(), color="#ffffff", linewidth=0.8, linestyle="--", alpha=0.5
                    )
                    ax2.set_xticks(range(len(grp)))
                    ax2.set_xticklabels(
                        [str(v)[:18] for v in grp.index],
                        rotation=40,
                        ha="right",
                        fontsize=7,
                        color="#8e97a4",
                    )
                    ax2.set_ylabel("Fraud Rate (%)", color="#8e97a4", fontsize=9)
                    ax2.set_title(
                        f"[FRAUD RATE] {col}",
                        color="#d63031",
                        fontsize=10,
                        fontweight="bold",
                        pad=12,
                    )
                    ax2.tick_params(axis="y", colors="#8e97a4")
                    ax2.spines[:].set_color("#ffffff11")
                    plt.tight_layout()
                    fig2.savefig(
                        plots_dir / f"{col}_fraud_rate.png",
                        dpi=110,
                        bbox_inches="tight",
                        facecolor="#06070b",
                    )
                    plt.close(fig2)

            except Exception as exc:  # noqa: BLE001
                logger.warning("Plot failed for column '%s': %s", col, exc)

        logger.info("Categorical plots saved to %s", plots_dir)

    # ------------------------------------------------------------------
    # Summary Builder
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        df_features: pd.DataFrame,
        df_rare: pd.DataFrame,
        df_card: pd.DataFrame,
        df_stab: pd.DataFrame,
    ) -> dict[str, Any]:
        """Builds top-level summary dict for the HTML report and JSON.

        Args:
            df_features: Feature identification DataFrame.
            df_rare: Rare categories DataFrame.
            df_card: Cardinality DataFrame.
            df_stab: Stability DataFrame.

        Returns:
            Summary metrics dictionary.
        """
        high_card = int(df_card["cardinality_class"].isin(["High", "Very High"]).sum()) if not df_card.empty else 0
        features_with_rare = int((df_rare["n_rare_categories"] > 0).sum()) if not df_rare.empty else 0
        high_drift = int((df_stab["drift_level"] == "High").sum()) if not df_stab.empty else 0

        return {
            "total_categorical_features": int(len(self.categorical_cols)),
            "high_cardinality_features": high_card,
            "features_with_rare_categories": features_with_rare,
            "high_drift_features": high_drift,
            "rare_threshold_pct": round(self.rare_threshold * 100, 2),
        }

    # ------------------------------------------------------------------
    # 3.7.12 — HTML Dashboard Compiler
    # ------------------------------------------------------------------

    def compile_html_dashboard(
        self,
        report_dir: Path,
        summary: dict[str, Any],
        df_features: pd.DataFrame,
        df_freq: pd.DataFrame,
        df_card: pd.DataFrame,
        df_imb: pd.DataFrame,
        df_fraud: pd.DataFrame,
        df_stab: pd.DataFrame,
        df_enc: pd.DataFrame,
    ) -> None:
        """Assembles the monochromatic HUD-style HTML dashboard report.

        Args:
            report_dir: Output directory.
            summary: Aggregated summary metrics.
            df_features: Categorical features identification DataFrame.
            df_freq: Frequency analysis DataFrame.
            df_card: Cardinality analysis DataFrame.
            df_imb: Imbalance analysis DataFrame.
            df_fraud: Fraud rates DataFrame.
            df_stab: Stability analysis DataFrame.
            df_enc: Encoding recommendations DataFrame.
        """
        # Build table rows
        def _rows(df: pd.DataFrame, cols: list[str], max_rows: int = 30) -> str:
            rows_html = ""
            for _, row in df.head(max_rows).iterrows():
                tds = "".join(
                    f"<td>{row.get(c, '')}</td>" for c in cols
                )
                rows_html += f"<tr>{tds}</tr>\n"
            return rows_html

        def _th(cols: list[str]) -> str:
            return "".join(f"<th>{c.replace('_', ' ').title()}</th>" for c in cols)

        feat_cols = ["feature", "dtype", "n_categories", "missing_pct", "dominant_category", "dominant_pct"]
        card_cols = ["feature", "n_unique", "cardinality_ratio", "max_freq", "min_freq", "cardinality_class"]
        imb_cols = ["feature", "dominant_pct", "smallest_pct", "entropy_bits", "diversity_index", "dominance_class"]
        stab_cols = ["feature", "train_categories", "test_categories", "new_categories_in_test", "tv_distance", "drift_level"]
        enc_cols = ["feature", "cardinality_class", "n_rare_categories", "recommended_encoding", "reason"]

        freq_top = (
            df_freq[df_freq["rank"] == 1][["feature", "category", "count", "percentage"]]
            if not df_freq.empty else pd.DataFrame()
        )
        fraud_top = (
            df_fraud.groupby("feature", group_keys=False).apply(
                lambda g: g.nlargest(1, "fraud_rate")
            ).reset_index(drop=True)[["feature", "category", "fraud_rate", "relative_risk"]]
            if not df_fraud.empty and "fraud_rate" in df_fraud.columns else pd.DataFrame()
        )

        # Plot image carousel slides
        plots_dir = report_dir / "freq_plots"
        plot_cols_available = sorted(plots_dir.glob("*_freq.png")) if plots_dir.exists() else []
        slides = ""
        nav_btns = ""
        for i, img_path in enumerate(plot_cols_available[:8]):
            col_name = img_path.stem.replace("_freq", "")
            active = "active" if i == 0 else ""
            fraud_img = plots_dir / f"{col_name}_fraud_rate.png"
            fraud_img_tag = (
                f'<div class="visual-card"><div class="v-title">[FRAUD RATE] {col_name}</div>'
                f'<img src="freq_plots/{fraud_img.name}" alt="{col_name} fraud"></div>'
                if fraud_img.exists() else ""
            )
            slides += f"""
            <div class="carousel-slide {active}" id="slide-{col_name}">
                <h3>Feature: {col_name}</h3>
                <div class="grid-2" style="margin-top:15px;">
                    <div class="visual-card">
                        <div class="v-title">[FREQUENCY] {col_name}</div>
                        <img src="freq_plots/{img_path.name}" alt="{col_name} freq">
                    </div>
                    {fraud_img_tag}
                </div>
            </div>"""
            nav_btns += f'<button onclick="showSlide(\'{col_name}\')">{col_name}</button>\n'

        font_url = (
            "https://fonts.googleapis.com/css2?"
            "family=Orbitron:wght@400;600;800;900&"
            "family=JetBrains+Mono:wght@400;700&"
            "family=Inter:wght@400;600&"
            "display=swap"
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IEEE-CIS Categorical Feature Diagnostics</title>
    <link href="{font_url}" rel="stylesheet">
    <style>
        :root {{
            --bg: #06070b;
            --card-bg: rgba(18, 22, 32, 0.45);
            --border: rgba(255,255,255,0.08);
            --text: #ffffff;
            --muted: #8e97a4;
            --danger: #d63031;
            --pass: #13b981;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        @keyframes scanline {{
            0% {{ transform: translateY(-100%); }}
            100% {{ transform: translateY(100%); }}
        }}
        @keyframes pulse-grey {{
            0%, 100% {{ opacity: 0.4; }}
            50% {{ opacity: 1.0; }}
        }}
        .scanline-overlay {{
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: linear-gradient(rgba(255,255,255,0), rgba(255,255,255,0.012) 50%, rgba(255,255,255,0) 100%);
            background-size: 100% 4px;
            animation: scanline 12s linear infinite;
            pointer-events: none; z-index: 9999;
        }}
        body {{
            background-color: var(--bg); color: var(--text);
            font-family: 'Inter', sans-serif;
            min-height: 100vh; overflow-x: hidden;
        }}
        .hud-grid-bg {{
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background-image:
                linear-gradient(rgba(255,255,255,0.01) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.01) 1px, transparent 1px);
            background-size: 40px 40px;
            pointer-events: none; z-index: 0;
        }}
        header {{
            background: rgba(6,7,11,0.9);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 20px 40px;
            display: flex; justify-content: space-between; align-items: center;
            position: sticky; top: 0; z-index: 100;
        }}
        .logo {{
            font-family: 'Orbitron', sans-serif; font-size: 1.4rem; font-weight: 900;
            letter-spacing: 0.05em; text-transform: uppercase; color: var(--text);
        }}
        .logo span {{ color: var(--muted); font-weight: 400; }}
        .nav {{ display: flex; gap: 20px; }}
        .nav a {{
            color: var(--muted); text-decoration: none;
            font-family: 'Orbitron', sans-serif; font-size: 0.75rem;
            letter-spacing: 0.05em; text-transform: uppercase; transition: color 0.2s;
        }}
        .nav a:hover {{ color: var(--text); }}
        .hud-status {{
            font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--muted);
            display: flex; align-items: center; gap: 8px;
            background: rgba(255,255,255,0.03); padding: 6px 12px;
            border: 1px solid var(--border); border-radius: 4px;
        }}
        .pulse-dot {{
            width: 8px; height: 8px; background-color: var(--muted);
            border-radius: 50%; animation: pulse-grey 2s infinite;
        }}
        main {{
            max-width: 1400px; margin: 40px auto; padding: 0 24px;
            position: relative; z-index: 1;
        }}
        .grid-4 {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr));
            gap: 20px; margin-bottom: 40px;
        }}
        .grid-2 {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(500px,1fr));
            gap: 25px; margin-bottom: 40px;
        }}
        @media(max-width:768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
        .glass-card {{
            background: var(--card-bg); border: 1px solid var(--border);
            border-radius: 4px; padding: 24px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            position: relative; transition: all 0.3s ease;
        }}
        .glass-card::before {{
            content: ""; position: absolute; top: 0; left: 0;
            width: 6px; height: 6px;
            border-top: 1px solid var(--muted); border-left: 1px solid var(--muted);
        }}
        .glass-card:hover {{
            border-color: rgba(255,255,255,0.2);
            box-shadow: 0 0 15px rgba(255,255,255,0.08);
        }}
        .metric-title {{
            font-family: 'Orbitron', sans-serif; font-size: 0.7rem;
            text-transform: uppercase; letter-spacing: 0.08em;
            color: var(--muted); margin-bottom: 8px;
        }}
        .metric-value {{
            font-family: 'JetBrains Mono', monospace; font-size: 2rem;
            font-weight: 700; color: var(--text); margin-bottom: 4px;
        }}
        .metric-desc {{ font-size: 0.72rem; color: var(--muted); }}
        .sect-title {{
            font-family: 'Orbitron', sans-serif; font-size: 1rem;
            text-transform: uppercase; letter-spacing: 0.08em;
            color: var(--text); border-bottom: 1px solid var(--border);
            padding-bottom: 8px; margin-bottom: 16px;
            display: flex; justify-content: space-between; align-items: center;
        }}
        .sect-title::after {{ font-size: 0.65rem; color: var(--muted); content: "[SECTION.CAT]"; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
        th {{
            text-align: left; padding: 8px 10px;
            border-bottom: 2px solid var(--border);
            font-family: 'Orbitron', sans-serif; font-size: 0.65rem;
            letter-spacing: 0.05em; text-transform: uppercase; color: var(--text);
        }}
        td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); color: var(--muted); }}
        td:first-child {{ color: var(--text); font-weight: 600; }}
        tr:hover td {{ background: rgba(255,255,255,0.03); }}
        .badge {{
            display: inline-block; padding: 3px 7px;
            border-radius: 0; font-size: 0.65rem; font-weight: 600;
            font-family: 'Orbitron', sans-serif; letter-spacing: 0.05em; text-transform: uppercase;
        }}
        .badge-high {{ background: rgba(214,48,49,0.08); color: var(--danger); border: 1px solid rgba(214,48,49,0.2); }}
        .badge-low  {{ background: rgba(255,255,255,0.05); color: var(--text); border: 1px solid var(--border); }}
        .badge-pass {{ background: rgba(19,185,129,0.08); color: var(--pass); border: 1px solid rgba(19,185,129,0.2); }}
        .carousel-tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
        .carousel-tabs button {{
            background: rgba(255,255,255,0.03); border: 1px solid var(--border);
            color: var(--muted); padding: 7px 14px; cursor: pointer;
            font-family: 'Orbitron', sans-serif; font-size: 0.65rem;
            letter-spacing: 0.05em; transition: all 0.2s;
        }}
        .carousel-tabs button:hover, .carousel-tabs button.active {{
            background: rgba(255,255,255,0.08); color: var(--text);
            border-color: rgba(255,255,255,0.25);
        }}
        .carousel-slide {{ display: none; }}
        .carousel-slide.active {{ display: block; }}
        .carousel-slide h3 {{
            font-family: 'Orbitron', sans-serif; font-size: 0.9rem;
            letter-spacing: 0.06em; text-transform: uppercase; color: var(--text);
            margin-bottom: 12px;
        }}
        .visual-card {{
            background: rgba(255,255,255,0.01); border: 1px solid var(--border);
            padding: 14px; text-align: center;
        }}
        .v-title {{
            font-family: 'Orbitron', sans-serif; font-size: 0.65rem;
            text-transform: uppercase; letter-spacing: 0.05em;
            color: var(--muted); margin-bottom: 10px;
        }}
        .visual-card img {{
            max-width: 100%; height: auto;
            border: 1px solid var(--border);
            filter: grayscale(100%) contrast(1.1) brightness(0.9);
        }}
    </style>
    <script>
        function showSlide(n) {{
            document.querySelectorAll('.carousel-slide').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.carousel-tabs button').forEach(b => b.classList.remove('active'));
            const el = document.getElementById('slide-' + n);
            if (el) el.classList.add('active');
            for (let btn of document.querySelectorAll('.carousel-tabs button')) {{
                if (btn.textContent.trim() === n) {{ btn.classList.add('active'); break; }}
            }}
        }}
    </script>
</head>
<body>
    <div class="scanline-overlay"></div>
    <div class="hud-grid-bg"></div>
    <header>
        <div class="logo">IEEE-CIS <span>Categorical Diagnostics</span></div>
        <div class="nav">
            <a href="#summary">Summary</a>
            <a href="#visuals">Visuals</a>
            <a href="#tables">Tables</a>
            <a href="#encoding">Encoding</a>
        </div>
        <div class="hud-status">
            <span class="pulse-dot"></span>
            <span>CAT ANALYZER OK</span>
        </div>
    </header>
    <main>
        <div id="summary" class="grid-4">
            <div class="glass-card">
                <div class="metric-title">Total Categorical Features</div>
                <div class="metric-value">{summary["total_categorical_features"]}</div>
                <div class="metric-desc">All identified categorical columns</div>
            </div>
            <div class="glass-card">
                <div class="metric-title">High Cardinality Features</div>
                <div class="metric-value" style="color:var(--danger);">{summary["high_cardinality_features"]}</div>
                <div class="metric-desc">Unique count &gt; 100</div>
            </div>
            <div class="glass-card">
                <div class="metric-title">Features With Rare Cats</div>
                <div class="metric-value" style="color:var(--muted);">{summary["features_with_rare_categories"]}</div>
                <div class="metric-desc">Below {summary["rare_threshold_pct"]}% frequency threshold</div>
            </div>
            <div class="glass-card">
                <div class="metric-title">High Drift Features</div>
                <div class="metric-value" style="color:var(--danger);">{summary["high_drift_features"]}</div>
                <div class="metric-desc">TV distance &gt; 0.15 train vs test</div>
            </div>
        </div>

        <div id="visuals" class="glass-card" style="margin-bottom:40px;">
            <div class="sect-title">Feature Frequency & Fraud Rate Plots</div>
            <div class="carousel-tabs">
                {nav_btns if nav_btns else '<span style="color:var(--muted);font-size:0.8rem;">No plots available</span>'}
            </div>
            <div class="carousel-slides">
                {slides if slides else ''}
            </div>
        </div>

        <div id="tables" class="grid-2">
            <div class="glass-card">
                <div class="sect-title">Feature Identification (Top 25)</div>
                <div style="overflow-x:auto;">
                    <table>
                        <thead><tr>{_th(feat_cols)}</tr></thead>
                        <tbody>{_rows(df_features, feat_cols)}</tbody>
                    </table>
                </div>
            </div>
            <div class="glass-card">
                <div class="sect-title">Cardinality Analysis</div>
                <div style="overflow-x:auto;">
                    <table>
                        <thead><tr>{_th(card_cols)}</tr></thead>
                        <tbody>{_rows(df_card, card_cols)}</tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="grid-2">
            <div class="glass-card">
                <div class="sect-title">Category Imbalance</div>
                <div style="overflow-x:auto;">
                    <table>
                        <thead><tr>{_th(imb_cols)}</tr></thead>
                        <tbody>{_rows(df_imb, imb_cols)}</tbody>
                    </table>
                </div>
            </div>
            <div class="glass-card">
                <div class="sect-title">Train vs Test Stability (Top 25 by Drift)</div>
                <div style="overflow-x:auto;">
                    <table>
                        <thead><tr>{_th(stab_cols)}</tr></thead>
                        <tbody>{_rows(df_stab, stab_cols)}</tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="grid-2">
            <div class="glass-card">
                <div class="sect-title">Top Dominant Category per Feature</div>
                <div style="overflow-x:auto;">
                    <table>
                        <thead><tr><th>Feature</th><th>Category</th><th>Count</th><th>Percentage</th></tr></thead>
                        <tbody>{_rows(freq_top, ["feature","category","count","percentage"]) if not freq_top.empty else "<tr><td colspan='4'>No data</td></tr>"}</tbody>
                    </table>
                </div>
            </div>
            <div class="glass-card">
                <div class="sect-title">Highest Fraud Rate per Feature</div>
                <div style="overflow-x:auto;">
                    <table>
                        <thead><tr><th>Feature</th><th>Category</th><th>Fraud Rate %</th><th>Relative Risk</th></tr></thead>
                        <tbody>{_rows(fraud_top, ["feature","category","fraud_rate","relative_risk"]) if not fraud_top.empty else "<tr><td colspan='4'>Target not available</td></tr>"}</tbody>
                    </table>
                </div>
            </div>
        </div>

        <div id="encoding" class="glass-card" style="margin-bottom:40px;">
            <div class="sect-title">Encoding Recommendations</div>
            <div style="overflow-x:auto;">
                <table>
                    <thead><tr>{_th(enc_cols)}</tr></thead>
                    <tbody>{_rows(df_enc, enc_cols, max_rows=50)}</tbody>
                </table>
            </div>
        </div>
    </main>
    <script>
        const firstBtn = document.querySelector('.carousel-tabs button');
        if (firstBtn) firstBtn.click();
    </script>
</body>
</html>"""

        report_path = report_dir / "categorical_analysis_report.html"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Compiled Categorical HTML dashboard to: %s", report_path)
