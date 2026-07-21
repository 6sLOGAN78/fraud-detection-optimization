"""Dataset profiling module for IEEE-CIS datasets statistical & structural analysis."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy, kurtosis, skew

from src.utils.logging import setup_logger

logger = setup_logger("dataset_profiling")


def classify_column_sdd(col: str) -> str:
    """Classifies column name into families according to Part 3.2 SDD.

    Args:
        col: Column name.

    Returns:
        Feature family name.
    """
    col_lower = col.lower()
    if col == "isFraud":
        return "Target"
    if col == "TransactionID":
        return "Identifier"
    if col == "TransactionDT":
        return "Time"

    # Card block
    if col.startswith("card"):
        return "Card"

    # Address block
    if col.startswith("addr"):
        return "Address"

    # Distance block
    if col.startswith("dist"):
        return "Distance"

    # Email domains
    if "emaildomain" in col_lower:
        return "Email"

    # Device features
    if col in ["DeviceInfo", "DeviceType"] or col.startswith("device"):
        return "Device"

    # C, D, M, V families
    if col.startswith("C") and col[1:].isdigit():
        return "Count"
    if col.startswith("D") and col[1:].isdigit():
        return "Delta"
    if col.startswith("M") and col[1:].isdigit():
        return "Match"
    if col.startswith("V") and col[1:].isdigit():
        return "Anonymous"

    # id_01 - id_38
    if col.startswith("id_"):
        return "Identity"

    # Transaction column fallbacks
    if col in ["ProductCD", "TransactionAmt"]:
        return "Transaction"

    return "Transaction"


class DatasetProfiler:
    """Performs dataset structural and statistical profiling on train and test folds."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_col: str = "isFraud",
    ) -> None:
        """Initializes the DatasetProfiler with training and testing datasets."""
        self.df_train = df_train
        # Align test columns (e.g. id-01 -> id_01) to match training schema
        self.df_test = df_test.rename(columns=lambda x: x.replace("-", "_"))
        self.target_col = target_col

        # Common columns (ignoring target in test)
        self.common_cols = [c for c in df_train.columns if c != target_col]

    def profile_inventory(self) -> dict[str, Any]:
        """Generates dataset size, dimensions, and type groupings."""

        def get_counts(df: pd.DataFrame) -> dict[str, int]:
            num_cols = 0
            cat_cols = 0
            bool_cols = 0
            for col in df.columns:
                t = df[col].dtype
                if isinstance(t, pd.CategoricalDtype) or t is object:
                    cat_cols += 1
                elif t is bool:
                    bool_cols += 1
                else:
                    num_cols += 1
            return {
                "numeric": num_cols,
                "categorical": cat_cols,
                "boolean": bool_cols,
            }

        train_types = get_counts(self.df_train)
        test_types = get_counts(self.df_test)

        train_mem = float(self.df_train.memory_usage(deep=True).sum())
        test_mem = float(self.df_test.memory_usage(deep=True).sum())

        # Identity coverage (density of records having non-null device/id)
        # We define identity attributes as features belonging to identity/device
        id_cols = [
            c
            for c in self.common_cols
            if classify_column_sdd(c) in ["Identity", "Device"]
        ]
        if id_cols:
            train_has_id = int((~self.df_train[id_cols].isna().all(axis=1)).sum())
            test_has_id = int((~self.df_test[id_cols].isna().all(axis=1)).sum())
        else:
            train_has_id = 0
            test_has_id = 0

        fraud_samples = 0
        non_fraud_samples = 0
        fraud_pct = 0.0

        if self.target_col in self.df_train.columns:
            fraud_samples = int((self.df_train[self.target_col] == 1).sum())
            non_fraud_samples = int((self.df_train[self.target_col] == 0).sum())
            if len(self.df_train) > 0:
                fraud_pct = float((fraud_samples / len(self.df_train)) * 100)

        return {
            "train_rows": len(self.df_train),
            "train_cols": len(self.df_train.columns),
            "train_mem_bytes": train_mem,
            "train_numeric": train_types["numeric"],
            "train_categorical": train_types["categorical"],
            "train_boolean": train_types["boolean"],
            "test_rows": len(self.df_test),
            "test_cols": len(self.df_test.columns),
            "test_mem_bytes": test_mem,
            "test_numeric": test_types["numeric"],
            "test_categorical": test_types["categorical"],
            "test_boolean": test_types["boolean"],
            "fraud_samples": fraud_samples,
            "non_fraud_samples": non_fraud_samples,
            "fraud_pct": fraud_pct,
            "train_identity_count": train_has_id,
            "train_identity_pct": (
                float(train_has_id / len(self.df_train) * 100)
                if len(self.df_train) > 0
                else 0.0
            ),
            "test_identity_count": test_has_id,
            "test_identity_pct": (
                float(test_has_id / len(self.df_test) * 100)
                if len(self.df_test) > 0
                else 0.0
            ),
        }

    def profile_memory(self, report_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Analyzes memory per column and exports charts."""
        mem_train = self.df_train.memory_usage(deep=True)
        col_mem_bytes = mem_train.to_frame(name="bytes_train")
        col_mem_bytes["bytes_test"] = self.df_test.memory_usage(deep=True)
        col_mem_bytes = col_mem_bytes.fillna(0)

        col_mem_bytes["family"] = [
            classify_column_sdd(c) if c in self.common_cols else "Target"
            for c in col_mem_bytes.index
        ]
        col_mem_bytes["mb_train"] = col_mem_bytes["bytes_train"] / (1024 * 1024)

        # Sort columns to identify top memory consumers
        top_consumers = col_mem_bytes.sort_values(by="mb_train", ascending=False).head(
            20
        )

        # Plot memory usage
        report_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=top_consumers.reset_index(),
            x="mb_train",
            y="index",
            hue="index",
            palette="viridis",
            legend=False,
        )
        plt.title("Top 20 Largest Memory-Consuming Columns (Train)")
        plt.xlabel("Memory Usage (MB)")
        plt.ylabel("Column")
        plt.tight_layout()
        plt.savefig(report_dir / "memory_usage.png", dpi=100)
        plt.close()

        # Compile summary stats
        total_mem_train = float(mem_train.sum())
        total_mem_test = float(self.df_test.memory_usage(deep=True).sum())
        n_train = len(self.df_train)
        avg_row_train = float(total_mem_train / n_train) if n_train > 0 else 0.0

        # Memory profile DataFrame index contains Index itself
        col_mem_bytes = col_mem_bytes.reset_index().rename(columns={"index": "column"})
        col_mem_bytes.to_csv(report_dir / "memory_profile.csv", index=False)

        summary = {
            "total_mem_train_bytes": total_mem_train,
            "total_mem_test_bytes": total_mem_test,
            "avg_mem_per_row_train_bytes": avg_row_train,
            "top_consumers": top_consumers["mb_train"].to_dict(),
        }

        # Keep output files clean
        with (report_dir / "memory_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        return col_mem_bytes, summary

    def analyze_cardinality(self, report_dir: Path) -> pd.DataFrame:
        """Classifies column cardinality and flags outliers."""
        card_records = []
        n_rows = len(self.df_train)

        for col in self.df_train.columns:
            u_count = int(self.df_train[col].nunique(dropna=True))
            u_pct = float((u_count / n_rows) * 100) if n_rows > 0 else 0.0

            if u_count == 1:
                class_type = "Constant"
            elif u_count == 2:
                class_type = "Binary"
            elif u_count <= 10:
                class_type = "Low Cardinality"
            elif u_count <= 100:
                class_type = "Medium Cardinality"
            else:
                class_type = "High Cardinality"

            val_counts = self.df_train[col].value_counts(dropna=True)
            most_common = val_counts.index[0] if not val_counts.empty else None
            least_common = val_counts.index[-1] if not val_counts.empty else None

            card_records.append(
                {
                    "column": col,
                    "family": classify_column_sdd(col),
                    "unique_values": u_count,
                    "unique_pct": u_pct,
                    "classification": class_type,
                    "most_common_value": str(most_common),
                    "least_common_value": str(least_common),
                }
            )

        df_card = pd.DataFrame(card_records)
        df_card.to_csv(report_dir / "cardinality_report.csv", index=False)

        # Output high-cardinality list
        high_card = df_card[df_card["classification"] == "High Cardinality"]
        high_card.to_csv(report_dir / "high_cardinality_features.csv", index=False)

        # Plot cardinality classes
        plt.figure(figsize=(8, 5))
        sns.countplot(
            data=df_card,
            x="classification",
            hue="classification",
            palette="coolwarm",
            legend=False,
        )
        plt.title("Feature Multiplicity Distribution")
        plt.xlabel("Cardinality Class")
        plt.ylabel("Number of Columns")
        plt.tight_layout()
        plt.savefig(report_dir / "cardinality_histogram.png", dpi=100)
        plt.close()

        return df_card

    def profile_statistics(self, report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generates statistical metrics for both numerical and categorical fields."""
        num_stats = []
        cat_stats = []

        for col in self.df_train.columns:
            series = self.df_train[col]
            dtype = series.dtype

            # Categorical checks
            if isinstance(dtype, pd.CategoricalDtype) or dtype is object:
                # Basic Categorical Stats
                u_cats = int(series.nunique(dropna=True))
                val_counts = series.value_counts(dropna=True)
                dominant_pct = (
                    float((val_counts.iloc[0] / len(series)) * 100)
                    if not val_counts.empty
                    else 0.0
                )
                ent_val = (
                    float(entropy(val_counts.values)) if not val_counts.empty else 0.0
                )

                cat_stats.append(
                    {
                        "column": col,
                        "num_categories": u_cats,
                        "most_frequent": (
                            str(val_counts.index[0]) if not val_counts.empty else None
                        ),
                        "least_frequent": (
                            str(val_counts.index[-1]) if not val_counts.empty else None
                        ),
                        "dominant_category_pct": dominant_pct,
                        "entropy": ent_val,
                    }
                )
            else:
                # Numerical stats
                clean_s = series.dropna()
                mean_val = float(clean_s.mean()) if not clean_s.empty else np.nan
                median_val = float(clean_s.median()) if not clean_s.empty else np.nan
                mode_val = np.nan
                if not clean_s.empty and not clean_s.mode().empty:
                    mode_val = float(clean_s.mode().iloc[0])
                min_val = float(clean_s.min()) if not clean_s.empty else np.nan
                max_val = float(clean_s.max()) if not clean_s.empty else np.nan
                range_val = float(max_val - min_val) if not clean_s.empty else np.nan
                var_val = float(clean_s.var()) if not clean_s.empty else np.nan
                std_val = float(clean_s.std()) if not clean_s.empty else np.nan
                q1 = float(clean_s.quantile(0.25)) if not clean_s.empty else np.nan
                q3 = float(clean_s.quantile(0.75)) if not clean_s.empty else np.nan
                iqr = float(q3 - q1) if not clean_s.empty else np.nan
                skew_val = (
                    float(skew(clean_s))
                    if not clean_s.empty and len(clean_s) > 2
                    else np.nan
                )
                kurt_val = (
                    float(kurtosis(clean_s))
                    if not clean_s.empty and len(clean_s) > 2
                    else np.nan
                )

                num_stats.append(
                    {
                        "column": col,
                        "mean": mean_val,
                        "median": median_val,
                        "mode": mode_val,
                        "min": min_val,
                        "max": max_val,
                        "range": range_val,
                        "variance": var_val,
                        "std_dev": std_val,
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                        "skewness": skew_val,
                        "kurtosis": kurt_val,
                    }
                )

        df_num = pd.DataFrame(num_stats)
        df_cat = pd.DataFrame(cat_stats)

        df_num.to_csv(report_dir / "numerical_statistics.csv", index=False)
        df_cat.to_csv(report_dir / "categorical_statistics.csv", index=False)

        return df_num, df_cat

    def profile_completeness(
        self, report_dir: Path
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Calculates row, column, and family completeness profiles."""
        n_rows = len(self.df_train)
        completeness_records = []

        # Feature completeness
        for col in self.df_train.columns:
            nulls = int(self.df_train[col].isna().sum())
            comp_pct = float(((n_rows - nulls) / n_rows) * 100) if n_rows > 0 else 100.0
            completeness_records.append(
                {
                    "column": col,
                    "family": classify_column_sdd(col),
                    "null_count": nulls,
                    "completeness_pct": comp_pct,
                }
            )

        df_comp = pd.DataFrame(completeness_records)
        df_comp.to_csv(report_dir / "completeness_report.csv", index=False)

        # Family-wise completeness
        family_comp = df_comp.groupby("family")["completeness_pct"].mean().to_dict()

        # Overall row completeness properties
        row_nulls = self.df_train.isna().sum(axis=1)
        n_cols = len(self.df_train.columns)
        row_comp_pct = ((n_cols - row_nulls) / n_cols) * 100
        avg_row_comp = float(row_comp_pct.mean())

        overall_null_cells = int(self.df_train.isna().sum().sum())
        total_cells = len(self.df_train) * n_cols
        overall_comp = (
            float(((total_cells - overall_null_cells) / total_cells) * 100)
            if total_cells > 0
            else 100.0
        )

        summary = {
            "overall_completeness_pct": overall_comp,
            "average_row_completeness_pct": avg_row_comp,
            "family_completeness_pct": family_comp,
        }

        # Plot completeness per family
        plt.figure(figsize=(9, 5))
        sns.barplot(
            x=list(family_comp.values()),
            y=list(family_comp.keys()),
            hue=list(family_comp.keys()),
            palette="mako",
            legend=False,
        )
        plt.title("Average Completeness Percentage by Feature Family")
        plt.xlabel("Completeness (%)")
        plt.ylabel("Feature Family")
        plt.xlim(0, 105)
        plt.tight_layout()
        plt.savefig(report_dir / "completeness_families.png", dpi=100)
        plt.close()

        comp_summary_path = report_dir / "completeness_summary.json"
        with comp_summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        return df_comp, summary

    def generate_recommendations(
        self,
        df_card: pd.DataFrame,
        df_comp: pd.DataFrame,
    ) -> list[dict[str, str]]:
        """Generates analytical recommendations based on metadata constraints."""
        recs = []

        # 1. Identify columns requiring drop or investigation (excessive nulls)
        empty_cols = df_comp[df_comp["completeness_pct"] == 0.0]["column"].tolist()
        if empty_cols:
            recs.append(
                {
                    "category": "Potential Removal",
                    "target": ", ".join(empty_cols[:5])
                    + ("..." if len(empty_cols) > 5 else ""),
                    "recommendation": (
                        "Completely empty features. Recommended for immediate removal."
                    ),
                }
            )

        high_missing = df_comp[
            (df_comp["completeness_pct"] < 10.0) & (df_comp["completeness_pct"] > 0.0)
        ]["column"].tolist()
        if high_missing:
            recs.append(
                {
                    "category": "High Missingness Warning",
                    "target": ", ".join(high_missing[:5])
                    + ("..." if len(high_missing) > 5 else ""),
                    "recommendation": (
                        "Features have >90% missing values. Keep only "
                        "if informative for tree models, otherwise drop."
                    ),
                }
            )

        # 2. Constant columns
        constants = df_card[df_card["classification"] == "Constant"]["column"].tolist()
        if constants:
            recs.append(
                {
                    "category": "Potential Removal",
                    "target": ", ".join(constants[:5])
                    + ("..." if len(constants) > 5 else ""),
                    "recommendation": (
                        "Constant values. Provide zero variance; drop before training."
                    ),
                }
            )

        # 3. High-cardinality features requiring target encoding
        high_cat_cols = df_card[
            (df_card["classification"] == "High Cardinality")
            & (df_card["unique_values"] > 30)
        ]["column"].tolist()
        # Keep device info / ids only
        cat_high = [
            c
            for c in high_cat_cols
            if c in self.df_train.columns
            and (
                isinstance(self.df_train[c].dtype, pd.CategoricalDtype)
                or self.df_train[c].dtype == object
            )
        ]
        if cat_high:
            recs.append(
                {
                    "category": "Encoding Strategy",
                    "target": ", ".join(cat_high[:5])
                    + ("..." if len(cat_high) > 5 else ""),
                    "recommendation": (
                        "High cardinality categorical features. Consider target "
                        "encoding on train fold, frequency encoding, or "
                        "grouping rare categories."
                    ),
                }
            )

        # 4. Low/Medium cardinality categoricals
        low_cat_cols = df_card[
            (df_card["classification"] == "Binary")
            | (df_card["classification"] == "Low Cardinality")
        ]["column"].tolist()
        cat_low = [
            c
            for c in low_cat_cols
            if c in self.df_train.columns
            and (
                isinstance(self.df_train[c].dtype, pd.CategoricalDtype)
                or self.df_train[c].dtype == object
            )
        ]
        if cat_low:
            recs.append(
                {
                    "category": "Encoding Strategy",
                    "target": ", ".join(cat_low[:5])
                    + ("..." if len(cat_low) > 5 else ""),
                    "recommendation": (
                        "Low cardinality categoricals. Suitable for default "
                        "One-Hot Encoding or standard label encoding."
                    ),
                }
            )

        # 5. Suggested Interaction Features
        recs.append(
            {
                "category": "Feature Engineering",
                "target": "card1 + addr1, card1 + TransactionAmt",
                "recommendation": (
                    "High-importance cross-product interaction "
                    "domains recommended for tree-based models."
                ),
            }
        )

        return recs


def generate_feature_dictionary_reports(
    df_card: pd.DataFrame,
    df_comp: pd.DataFrame,
    recs: list[dict[str, str]],
    report_dir: Path,
) -> pd.DataFrame:
    """Combines metadata facets into a unified Feature Dictionary table."""
    feat_dict = pd.merge(df_card, df_comp, on=["column", "family"])

    # Map recommendation recommendation texts to columns
    rec_map = {}
    for r in recs:
        targets = [t.strip() for t in r["target"].split(",") if t.strip()]
        for t in targets:
            # handle ellipses
            if t != "...":
                rec_map[t] = r["recommendation"]

    # Assign recommended processing
    rec_texts = []
    for col in feat_dict["column"]:
        row_cls = feat_dict.loc[feat_dict["column"] == col, "classification"].values[0]
        if col in rec_map:
            rec_texts.append(rec_map[col])
        elif row_cls == "High Cardinality":
            rec_texts.append("Numerical scaling or robust target encoding.")
        else:
            rec_texts.append("Pass through as baseline feature.")

    feat_dict["recommended_processing"] = rec_texts
    feat_dict.to_csv(report_dir / "feature_dictionary.csv", index=False)

    # Save to json dictionary
    json_dict = {}
    for _, row in feat_dict.iterrows():
        json_dict[row["column"]] = {
            "family": row["family"],
            "unique_values": int(row["unique_values"]),
            "classification": row["classification"],
            "completeness_pct": float(row["completeness_pct"]),
            "recommended_processing": row["recommended_processing"],
        }
    with (report_dir / "feature_dictionary.json").open("w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=4)

    return feat_dict


def write_markdown_summary(
    inventory: dict[str, Any],
    mem_summary: dict[str, Any],
    comp_summary: dict[str, Any],
    recs: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Generates the Markdown profiling report summary."""
    lines = [
        "# IEEE-CIS Fraud Detection - Pre-Training Dataset Summary Profile",
        "",
        "## 1. Dataset Dimensions & Basic Topology",
        f"- **Training Samples**: {inventory['train_rows']:,}",
        f"- **Testing Samples**: {inventory['test_rows']:,}",
        f"- **Training Features**: {inventory['train_cols']:,}",
        f"- **Testing Features**: {inventory['test_cols']:,}",
        (
            "- **Target Variable (`isFraud`) Class Balance (Train)**: "
            f"{inventory['fraud_pct']:.3f}% Fraudulent (Total: "
            f"{inventory['fraud_samples']:,} samples)"
        ),
        (
            f"- **Identity Match Coverage (Train)**: "
            f"{inventory['train_identity_pct']:.2f}% (Total: "
            f"{inventory['train_identity_count']:,} rows matched)"
        ),
        "",
        "## 2. Memory Analysis",
        (
            "- **Total Training RAM footprint**: "
            f"{mem_summary['total_mem_train_bytes'] / (1024 * 1024):.2f} MB"
        ),
        (
            "- **Total Testing RAM footprint**: "
            f"{mem_summary['total_mem_test_bytes'] / (1024 * 1024):.2f} MB"
        ),
        (
            "- **Average Memory per Row (Train)**: "
            f"{mem_summary['avg_mem_per_row_train_bytes']:.2f} Bytes"
        ),
        "",
        "## 3. Data Completeness",
        (
            "- **Overall Matrix Data Density**: "
            f"{comp_summary['overall_completeness_pct']:.2f}% "
            "of cells populated (non-null)"
        ),
        (
            "- **Average Row Completeness**: "
            f"{comp_summary['average_row_completeness_pct']:.2f}%"
        ),
        "",
        "### average cell completeness by Feature Family:",
    ]

    for fam, pct in comp_summary["family_completeness_pct"].items():
        lines.append(f"- **{fam}**: {pct:.2f}% completeness")

    lines.extend(
        [
            "",
            "## 4. Preprocessing Quality Recommendations",
        ]
    )

    for r in recs:
        lines.extend(
            [
                f"### [{r['category']}] (Targets: `{r['target']}`)",
                f"- **Recommendation**: {r['recommendation']}",
                "",
            ]
        )

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Saved markdown summary profile to %s", output_path)


def write_html_report(
    inventory: dict[str, Any],
    mem_summary: dict[str, Any],
    comp_summary: dict[str, Any],
    recs: list[dict[str, str]],
    feat_dict: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generates a premium glassmorphic HTML reporting dashboard."""
    font_link = (
        "https://fonts.googleapis.com/css2?"
        "family=Orbitron:wght@400;600;800;900&"
        "family=JetBrains+Mono:wght@400;700&"
        "family=Inter:wght@400;600&"
        "display=swap"
    )

    raw_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IEEE-CIS Tabular Dataset Profile</title>
    <link href="{font_link}" rel="stylesheet">
    <style>
        :root {{
            --bg-base: #06070b;
            --panel-bg: rgba(18, 22, 32, 0.45);
            --border-primary: rgba(255, 255, 255, 0.08);
            --text-main: #ffffff;
            --text-secondary: #8e97a4;
            --accent-glow: rgba(255, 255, 255, 0.12);
            --color-fault: #d63031;
            --color-pass: #13b981;
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
            background-color: var(--bg-base);
            font-family: 'Inter', sans-serif;
            color: var(--text-main);
            line-height: 1.6;
            padding: 2.5rem;
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
        .header {{
            position: relative;
            z-index: 1;
            margin-bottom: 2.5rem;
            border-bottom: 1px solid var(--border-primary);
            padding-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        .header-title-box {{
            display: flex;
            flex-direction: column;
        }}
        .header h1 {{
            font-family: 'Orbitron', sans-serif;
            font-size: 2.2rem;
            font-weight: 900;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #ffffff;
            text-shadow: 0 0 15px rgba(255, 255, 255, 0.15);
        }}
        .header p {{
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-top: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .hud-status {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.03);
            padding: 6px 12px;
            border: 1px solid var(--border-primary);
            border-radius: 4px;
        }}
        .pulse-dot {{
            width: 8px;
            height: 8px;
            background-color: var(--text-secondary);
            border-radius: 50%;
            animation: pulse-grey 2s infinite;
        }}
        .grid {{
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }}
        .card {{
            background: var(--panel-bg);
            border: 1px solid var(--border-primary);
            border-radius: 4px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
            transition: all 0.3s ease;
            position: relative;
        }}
        .card::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; width: 6px; height: 6px;
            border-top: 1px solid var(--text-secondary);
            border-left: 1px solid var(--text-secondary);
        }}
        .card:hover {{
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.08);
            transform: translateY(-2px);
        }}
        .card h2 {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.95rem;
            letter-spacing: 0.08em;
            margin-bottom: 1.2rem;
            color: #ffffff;
            text-transform: uppercase;
            border-bottom: 1px solid var(--border-primary);
            padding-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .card h2::after {{
            content: "[SYS.LOC]";
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            color: var(--text-secondary);
        }}
        .card-stat {{
            font-size: 2rem;
            font-weight: 700;
            margin: 0.5rem 0;
            color: #ffffff;
            font-family: 'JetBrains Mono', monospace;
        }}
        .card-label {{
            color: var(--text-secondary);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .flex-list {{
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}
        .flex-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px dashed rgba(255, 255, 255, 0.05);
            padding-bottom: 4px;
        }}
        .flex-item span:first-child {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .flex-item strong {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
        }}
        .recs-container {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
            max-height: 480px;
            overflow-y: auto;
            padding-right: 5px;
        }}
        .rec-item {{
            border-left: 2px solid #ffffff;
            background: rgba(255, 255, 255, 0.02);
            padding: 1rem;
            border-radius: 0;
        }}
        .rec-title {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #ffffff;
        }}
        .rec-targets {{
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
            font-size: 0.75rem;
            margin-top: 0.25rem;
        }}
        .rec-body {{
            margin-top: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        table.profile-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            font-size: 0.85rem;
        }}
        table.profile-table th, table.profile-table td {{
            padding: 0.8rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-primary);
        }}
        table.profile-table th {{
            background: rgba(255, 255, 255, 0.02);
            color: #ffffff;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-size: 0.8rem;
        }}
        table.profile-table tr:hover {{
            background: rgba(255, 255, 255, 0.03);
        }}
        table.profile-table td {{
            font-family: 'Inter', sans-serif;
        }}
        table.profile-table td strong {{
            font-family: 'JetBrains Mono', monospace;
            color: #ffffff;
        }}
    </style>
</head>
<body>
    <div class="scanline-overlay"></div>
    <div class="hud-grid-bg"></div>

    <div class="header">
        <div class="header-title-box">
            <h1>IEEE-CIS Tabular Dataset Profile</h1>
            <p>Pre-training diagnostic dashboard summary mapping feature structures & quality indices</p>
        </div>
        <div class="hud-status">
            <span class="pulse-dot"></span>
            <span>SYSTEM READY // MONOCHROME HUD v1.0</span>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Dimensions & Distribution</h2>
            <div class="card-stat">{inventory["train_rows"]:,}</div>
            <div class="card-label">Training Records</div>
            <div class="card-stat"
                 style="font-size: 1.5rem; margin-top: 1rem;">
                 {inventory["test_rows"]:,}
            </div>
            <div class="card-label">Testing Records</div>
        </div>

        <div class="card">
            <h2>Memory Footprints</h2>
            <div class="card-stat">
                 {mem_summary["total_mem_train_bytes"] / (1024 * 1024):.2f} MB
            </div>
            <div class="card-label">Total Train RAM</div>
            <div class="card-stat"
                 style="font-size: 1.5rem; margin-top: 1rem;">
                 {mem_summary["avg_mem_per_row_train_bytes"]:.2f} B
            </div>
            <div class="card-label">Average Footprint per row</div>
        </div>

        <div class="card">
            <h2>Quality & Density</h2>
            <div class="card-stat">{comp_summary["overall_completeness_pct"]:.2f}%</div>
            <div class="card-label">Non-Null Matrix Coverage</div>
            <div class="card-stat"
                 style="font-size: 1.5rem; margin-top: 1rem; color: var(--text-main);">
                 {inventory["fraud_pct"]:.3f}%
            </div>
            <div class="card-label">Fraud target class ratio</div>
        </div>
    </div>

    <div class="grid" style="grid-template-columns: 2fr 1fr;">
        <div class="card">
            <h2>Automatic Preprocessing Recommendations</h2>
            <div class="recs-container">"""

    for r in recs:
        raw_html += f"""
                <div class="rec-item">
                    <div class="rec-title">{r["category"]}</div>
                    <div class="rec-targets">Target features: {r["target"]}</div>
                    <div class="rec-body">{r["recommendation"]}</div>
                </div>"""

    raw_html += """
            </div>
        </div>

        <div class="card">
            <h2>Family Completeness</h2>
            <ul class="flex-list">"""

    for fam, pct in comp_summary["family_completeness_pct"].items():
        raw_html += f"""
                <li class="flex-item">
                    <span>{fam}</span>
                    <strong style="color: var(--text-main);">{pct:.2f}%</strong>
                </li>"""

    raw_html += """
            </ul>
        </div>
    </div>

    <div class="card" style="margin-bottom: 2.5rem;">
        <h2>Top 15 Columns Topology List</h2>
        <table class="profile-table">
            <thead>
                <tr>
                    <th>Column Name</th>
                    <th>Family</th>
                    <th>Unique Values</th>
                    <th>Classification</th>
                    <th>Completeness %</th>
                    <th>Recommended Processing</th>
                </tr>
            </thead>
            <tbody>"""

    # Populate top 15 rows in HTML
    for _, row in feat_dict.head(15).iterrows():
        raw_html += f"""
                <tr>
                    <td><strong>{row["column"]}</strong></td>
                    <td>{row["family"]}</td>
                    <td>{row["unique_values"]:,}</td>
                    <td>{row["classification"]}</td>
                    <td>{row["completeness_pct"]:.2f}%</td>
                    <td>{row["recommended_processing"]}</td>
                </tr>"""

    raw_html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    with output_path.open("w", encoding="utf-8") as f:
        f.write(raw_html)
    logger.info("Saved premium html dashboard visualization to %s", output_path)



def write_json_report(
    inventory: dict[str, Any],
    mem_summary: dict[str, Any],
    comp_summary: dict[str, Any],
    recs: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Saves structured report details to a JSON file."""
    profile_json = {
        "dataset_inventory": inventory,
        "memory_profile": mem_summary,
        "data_completeness": comp_summary,
        "recommendations": recs,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(profile_json, f, indent=4)
    logger.info("Saved structured JSON report profile to %s", output_path)
