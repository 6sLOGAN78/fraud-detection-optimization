"""Data quality assessment module for evaluating IEEE-CIS dataset integrity."""

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

from src.eda.profiling import classify_column_sdd

logger = logging.getLogger(__name__)


class DataQualityAssessor:
    """Evaluates dataset integrity, anomaly metrics, and consistency."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_col: str = "isFraud",
    ) -> None:
        """Initializes DataQualityAssessor with train and test datasets."""
        self.df_train = df_train
        # Align test columns (e.g. id-01 -> id_01) to match training schema
        self.df_test = df_test.rename(columns=lambda x: x.replace("-", "_"))
        self.target_col = target_col
        self.common_cols = [c for c in df_train.columns if c != target_col]

    def audit_missingness(
        self, report_dir: Path
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Audits null values and missing rates of training and testing features."""
        report_dir.mkdir(parents=True, exist_ok=True)
        records = []
        n_train = len(self.df_train)
        n_test = len(self.df_test)

        for col in self.df_train.columns:
            family = classify_column_sdd(col) if col != self.target_col else "Target"
            missing_tr = int(self.df_train[col].isna().sum())
            pct_tr = float((missing_tr / n_train) * 100) if n_train > 0 else 0.0

            if col in self.df_test.columns:
                missing_te = int(self.df_test[col].isna().sum())
                pct_te = float((missing_te / n_test) * 100) if n_test > 0 else 0.0
            else:
                missing_te = 0
                pct_te = 100.0  # Entirely missing in test

            records.append({
                "column": col,
                "family": family,
                "missing_count_train": missing_tr,
                "missing_pct_train": pct_tr,
                "available_pct_train": 100.0 - pct_tr,
                "missing_count_test": missing_te,
                "missing_pct_test": pct_te,
                "available_pct_test": 100.0 - pct_te,
            })

        df_missing = pd.DataFrame(records)
        df_missing.to_csv(report_dir / "missing_summary.csv", index=False)

        # Summarize by family
        fam_grouped = df_missing.groupby("family").agg({
            "missing_pct_train": "mean",
            "missing_pct_test": "mean"
        }).to_dict(orient="index")

        overall_null_cells_tr = int(self.df_train.isna().sum().sum())
        total_cells_tr = n_train * len(self.df_train.columns)
        overall_pct_tr = (
            float((overall_null_cells_tr / total_cells_tr) * 100)
            if total_cells_tr > 0
            else 0.0
        )

        overall_null_cells_te = int(self.df_test.isna().sum().sum())
        total_cells_te = n_test * len(self.df_test.columns)
        overall_pct_te = (
            float((overall_null_cells_te / total_cells_te) * 100)
            if total_cells_te > 0
            else 0.0
        )

        summary = {
            "train_overall_missing_pct": overall_pct_tr,
            "test_overall_missing_pct": overall_pct_te,
            "family_missing_pct": {
                fam: {
                    "train": float(vals["missing_pct_train"]),
                    "test": float(vals["missing_pct_test"]),
                }
                for fam, vals in fam_grouped.items()
            }
        }

        with (report_dir / "missing_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        # Plot 1: Top 20 columns by missing %
        top_missing_tr = df_missing.sort_values(
            by="missing_pct_train", ascending=False
        ).head(20)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="missing_pct_train",
            y="column",
            data=top_missing_tr,
            hue="column",
            palette="flare",
            legend=False,
        )
        plt.title("Top 20 Features with Highest Missing Percentage (Train)")
        plt.xlabel("Missing (%)")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(report_dir / "missing_bar_chart.png", dpi=100)
        plt.close()

        # Plot 2: Missing percentage histogram
        plt.figure(figsize=(8, 5))
        sns.histplot(
            df_missing["missing_pct_train"],
            bins=20,
            kde=True,
            color="#2a9d8f",
        )
        plt.title("Distribution of Missing Percentages across Features")
        plt.xlabel("Missing (%)")
        plt.ylabel("Number of Features")
        plt.tight_layout()
        plt.savefig(report_dir / "missing_percentage_histogram.png", dpi=100)
        plt.close()

        # Plot 3: Feature family missing chart
        family_df = pd.DataFrame([
            {"family": fam, "Dataset": "Train", "Missing %": vals["train"]}
            for fam, vals in summary["family_missing_pct"].items()
        ] + [
            {"family": fam, "Dataset": "Test", "Missing %": vals["test"]}
            for fam, vals in summary["family_missing_pct"].items()
        ])
        plt.figure(figsize=(10, 5))
        sns.barplot(
            x="family",
            y="Missing %",
            hue="Dataset",
            data=family_df,
            palette="mako",
        )
        plt.title("Average Missing Percentage by Feature Family")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Feature Family")
        plt.ylabel("Missing (%)")
        plt.tight_layout()
        plt.savefig(report_dir / "family_missing_chart.png", dpi=100)
        plt.close()

        return df_missing, summary

    def audit_duplicates(self, report_dir: Path) -> pd.DataFrame:
        """Evaluates redundant/duplicate instances, IDs, and column alignments."""
        report_dir.mkdir(parents=True, exist_ok=True)
        n_train = len(self.df_train)
        n_test = len(self.df_test)

        dup_rows_tr = int(self.df_train.duplicated().sum())
        dup_rows_te = int(self.df_test.duplicated().sum())

        dup_ids_tr = 0
        if "TransactionID" in self.df_train.columns:
            dup_ids_tr = int(self.df_train["TransactionID"].duplicated().sum())

        dup_ids_te = 0
        if "TransactionID" in self.df_test.columns:
            dup_ids_te = int(self.df_test["TransactionID"].duplicated().sum())

        # Duplicate transaction profiles (excluding unique identifier
        # and target/time keys)
        excl_keys = ["TransactionID", "TransactionDT", "isFraud", "has_identity"]
        sub_cols = [c for c in self.common_cols if c not in excl_keys]

        dup_trans_tr = 0
        if sub_cols:
            dup_trans_tr = int(self.df_train.duplicated(subset=sub_cols).sum())

        dup_trans_te = 0
        if sub_cols and set(sub_cols).issubset(self.df_test.columns):
            dup_trans_te = int(self.df_test.duplicated(subset=sub_cols).sum())

        # Duplicate Columns Check (columns having identical contents)
        identical_cols = []
        groups: dict[tuple[int, str], list[str]] = {}
        for col in self.common_cols:
            if col in self.df_train.columns:
                null_cnt = int(self.df_train[col].isna().sum())
                dtype_str = str(self.df_train[col].dtype)
                key = (null_cnt, dtype_str)
                groups.setdefault(key, []).append(col)

        for col_list in groups.values():
            if len(col_list) < 2:
                continue
            for i in range(len(col_list)):
                c1 = col_list[i]
                for j in range(i + 1, len(col_list)):
                    c2 = col_list[j]
                    if self.df_train[c1].equals(self.df_train[c2]):
                        identical_cols.append((c1, c2))

        records = [
            {
                "Category": "Duplicate Rows",
                "Metric": "Exact Duplicate Rows (Train)",
                "Value": dup_rows_tr,
                "Percentage": (
                    float(dup_rows_tr / n_train * 100)
                    if n_train > 0
                    else 0.0
                ),
            },
            {
                "Category": "Duplicate Rows",
                "Metric": "Exact Duplicate Rows (Test)",
                "Value": dup_rows_te,
                "Percentage": float(dup_rows_te / n_test * 100) if n_test > 0 else 0.0,
            },
            {
                "Category": "Duplicate Identifiers",
                "Metric": "Duplicate TransactionID (Train)",
                "Value": dup_ids_tr,
                "Percentage": float(dup_ids_tr / n_train * 100) if n_train > 0 else 0.0,
            },
            {
                "Category": "Duplicate Identifiers",
                "Metric": "Duplicate TransactionID (Test)",
                "Value": dup_ids_te,
                "Percentage": float(dup_ids_te / n_test * 100) if n_test > 0 else 0.0,
            },
            {
                "Category": "Duplicate Transactions",
                "Metric": "Duplicate Transaction Records (Train)",
                "Value": dup_trans_tr,
                "Percentage": (
                    float(dup_trans_tr / n_train * 100)
                    if n_train > 0
                    else 0.0
                ),
            },
            {
                "Category": "Duplicate Transactions",
                "Metric": "Duplicate Transaction Records (Test)",
                "Value": dup_trans_te,
                "Percentage": float(dup_trans_te / n_test * 100) if n_test > 0 else 0.0,
            },
            {
                "Category": "Duplicate Columns",
                "Metric": "Identical Feature Pairs",
                "Value": len(identical_cols),
                "Percentage": 0.0,
            },
        ]

        df_dup = pd.DataFrame(records)
        df_dup.to_csv(report_dir / "duplicate_report.csv", index=False)

        # Plot duplicates metrics
        plt.figure(figsize=(9, 5))
        sns.barplot(
            x=[r["Value"] for r in records if "Duplicate" in r["Category"]],
            y=[r["Metric"] for r in records if "Duplicate" in r["Category"]],
            hue=[r["Metric"] for r in records if "Duplicate" in r["Category"]],
            palette="crest",
            legend=False,
        )
        plt.title("Duplicate Profile Summary Counts")
        plt.xlabel("Occurrences")
        plt.ylabel("Category")
        plt.tight_layout()
        plt.savefig(report_dir / "duplicate_statistics.png", dpi=100)
        plt.close()

        # Add explicit file output for duplicated columns details
        dup_cols_path = report_dir / "duplicate_columns.json"
        with dup_cols_path.open("w", encoding="utf-8") as f:
            identical_list = [
                {"col1": c1, "col2": c2} for c1, c2 in identical_cols
            ]
            json.dump(identical_list, f, indent=4)

        return df_dup

    def detect_constant_features(self, report_dir: Path) -> pd.DataFrame:
        """Finds fields offering zero information capacity (1 unique value)."""
        report_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for col in self.common_cols:
            if col in self.df_train.columns:
                n_uniq = int(self.df_train[col].nunique(dropna=True))
                if n_uniq == 1:
                    val = self.df_train[col].dropna().iloc[0]
                    records.append({
                        "column": col,
                        "value": str(val),
                        "family": classify_column_sdd(col),
                    })

        df_const = pd.DataFrame(records)
        if df_const.empty:
            df_const = pd.DataFrame(columns=["column", "value", "family"])

        df_const.to_csv(report_dir / "constant_features.csv", index=False)

        # Visual chart
        plt.figure(figsize=(6, 4))
        sns.countplot(
            y="family",
            data=df_const,
            hue="family",
            palette="rocket",
            stat="count",
            legend=False,
        )
        plt.title(f"Constant Features Count by Family (Total: {len(df_const)})")
        plt.xlabel("Number of Constant Features")
        plt.ylabel("Feature Family")
        plt.tight_layout()
        plt.savefig(report_dir / "constant_feature_chart.png", dpi=100)
        plt.close()

        return df_const

    def detect_near_constant_features(
        self, report_dir: Path, threshold: float = 0.99
    ) -> pd.DataFrame:
        """Finds low-information columns dominated by a single value (>= 99%)."""
        report_dir.mkdir(parents=True, exist_ok=True)
        records = []
        n_train = len(self.df_train)

        # Get list of constant features to exclude
        const_features = set()
        const_csv = report_dir / "constant_features.csv"
        if const_csv.exists():
            const_features = set(pd.read_csv(const_csv)["column"].tolist())

        for col in self.common_cols:
            if col in self.df_train.columns and col not in const_features:
                series = self.df_train[col].dropna()
                if series.empty:
                    continue
                v_counts = series.value_counts(dropna=True)
                if not v_counts.empty:
                    dom_count = int(v_counts.iloc[0])
                    dom_pct = float(dom_count / n_train)
                    if dom_pct >= threshold:
                        records.append({
                            "column": col,
                            "dominant_value": str(v_counts.index[0]),
                            "frequency": dom_count,
                            "percentage": dom_pct * 100,
                            "family": classify_column_sdd(col),
                        })

        df_near = pd.DataFrame(records)
        if df_near.empty:
            df_near = pd.DataFrame(
                columns=[
                    "column",
                    "dominant_value",
                    "frequency",
                    "percentage",
                    "family",
                ]
            )

        df_near.to_csv(report_dir / "near_constant_features.csv", index=False)

        # Visual chart
        plt.figure(figsize=(6, 4))
        sns.countplot(
            y="family",
            data=df_near,
            hue="family",
            palette="viridis",
            stat="count",
            legend=False,
        )
        plt.title(f"Near-Constant Features by Family (Total: {len(df_near)})")
        plt.xlabel("Number of Near-Constant Features")
        plt.ylabel("Feature Family")
        plt.tight_layout()
        plt.savefig(report_dir / "near_constant_feature_chart.png", dpi=100)
        plt.close()

        return df_near

    def detect_invalid_values(self, report_dir: Path) -> pd.DataFrame:
        """Assess semantic errors, corrupted string representations or empty states."""
        report_dir.mkdir(parents=True, exist_ok=True)
        records = []
        n_train = len(self.df_train)

        # 1. Check Negative TransactionAmt
        if "TransactionAmt" in self.df_train.columns:
            neg_amt = int((self.df_train["TransactionAmt"] < 0).sum())
            if neg_amt > 0:
                records.append({
                    "column": "TransactionAmt",
                    "invalid_value_type": "Negative Numeric Amount",
                    "affected_rows": neg_amt,
                    "percentage": float(neg_amt / n_train * 100),
                    "description": (
                        "Financial Transaction amounts must be non-negative."
                    ),
                })

        # 2. Check unexpected string placeholders mimicking nulls
        null_strs = {"nan", "null", "none", "undefined", ""}
        for col in self.df_train.columns:
            if self.df_train[col].dtype == object or isinstance(
                self.df_train[col].dtype, pd.CategoricalDtype
            ):
                s = self.df_train[col].astype(str).str.strip().str.lower()
                matched = s.isin(null_strs) & self.df_train[col].notna()
                bad_cnt = int(matched.sum())
                if bad_cnt > 0:
                    records.append({
                        "column": col,
                        "invalid_value_type": (
                            "Corrupted Null String representing missingness"
                        ),
                        "affected_rows": bad_cnt,
                        "percentage": float(bad_cnt / n_train * 100),
                        "description": "String values encoding fake nulls.",
                    })

        # 3. Check invalid negative IDs or empty IDs
        if "TransactionID" in self.df_train.columns:
            neg_ids = int((self.df_train["TransactionID"] < 0).sum())
            if neg_ids > 0:
                records.append({
                    "column": "TransactionID",
                    "invalid_value_type": "Negative Identifier Index",
                    "affected_rows": neg_ids,
                    "percentage": float(neg_ids / n_train * 100),
                    "description": (
                        "Primary identifiers should not have negative inputs."
                    ),
                })

        df_invalid = pd.DataFrame(records)
        if df_invalid.empty:
            df_invalid = pd.DataFrame(
                columns=[
                    "column",
                    "invalid_value_type",
                    "affected_rows",
                    "percentage",
                    "description",
                ]
            )

        df_invalid.to_csv(report_dir / "invalid_values_report.csv", index=False)
        return df_invalid

    def detect_infinite_values(self, report_dir: Path) -> pd.DataFrame:
        """Finds infinite floats or values resulting in mathematical divergence."""
        report_dir.mkdir(parents=True, exist_ok=True)
        records = []
        n_train = len(self.df_train)

        for col in self.df_train.columns:
            if pd.api.types.is_numeric_dtype(self.df_train[col]):
                s = self.df_train[col].dropna()
                inf_count = int(np.isinf(s).sum())
                if inf_count > 0:
                    records.append({
                        "column": col,
                        "infinite_count": inf_count,
                        "total_count": n_train,
                        "percentage": float(inf_count / n_train * 100),
                    })

        df_inf = pd.DataFrame(records)
        if df_inf.empty:
            df_inf = pd.DataFrame(
                columns=[
                    "column",
                    "infinite_count",
                    "total_count",
                    "percentage",
                ]
            )

        df_inf.to_csv(report_dir / "infinite_values_report.csv", index=False)
        return df_inf

    def assess_outliers(self, report_dir: Path) -> pd.DataFrame:
        """Highlights highly skewed numerical features with outlier indicators."""
        report_dir.mkdir(parents=True, exist_ok=True)
        records = []

        for col in self.common_cols:
            if col in self.df_train.columns and pd.api.types.is_numeric_dtype(
                self.df_train[col]
            ):
                series = self.df_train[col].dropna()
                n_uniq = series.nunique()
                # Skip target variables, constant values and indicators
                if n_uniq <= 10 or col in ["TransactionDT", "TransactionID"]:
                    continue

                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                iqr_outliers = int(
                    (
                        (series < (q1 - 1.5 * iqr))
                        | (series > (q3 + 1.5 * iqr))
                    ).sum()
                )

                mean_val = float(series.mean())
                std_val = float(series.std())
                z_outliers = 0
                if std_val > 0:
                    z_outliers = int((np.abs((series - mean_val) / std_val) > 3).sum())

                records.append({
                    "column": col,
                    "family": classify_column_sdd(col),
                    "outliers_iqr_count": iqr_outliers,
                    "outliers_iqr_pct": (
                        float(iqr_outliers / len(series) * 100)
                        if len(series) > 0
                        else 0.0
                    ),
                    "outliers_zscore_count": z_outliers,
                    "outliers_zscore_pct": (
                        float(z_outliers / len(series) * 100)
                        if len(series) > 0
                        else 0.0
                    ),
                })

        df_out = pd.DataFrame(records)
        if df_out.empty:
            df_out = pd.DataFrame(
                columns=[
                    "column",
                    "family",
                    "outliers_iqr_count",
                    "outliers_iqr_pct",
                    "outliers_zscore_count",
                    "outliers_zscore_pct"
                ]
            )

        df_out.to_csv(report_dir / "outlier_summary.csv", index=False)

        # Plot Outlier Overview
        if not df_out.empty:
            top_outliers = df_out.sort_values(
                by="outliers_iqr_pct", ascending=False
            ).head(5)
            # Create subplots for boxplots
            _, axes = plt.subplots(
                1, min(5, len(top_outliers)), figsize=(12, 4), squeeze=False
            )
            for idx, (_, row) in enumerate(top_outliers.iterrows()):
                col_name = str(row["column"])
                sns.boxplot(
                    y=self.df_train[col_name].dropna(),
                    ax=axes[0, idx],
                    color="#e76f51",
                )
                axes[0, idx].set_title(col_name)
                axes[0, idx].set_ylabel("")
            plt.suptitle("Boxplots for Top Outlier-heavy Numeric Features")
            plt.tight_layout()
            plt.savefig(report_dir / "outlier_overview.png", dpi=100)
            plt.close()

            # Plot Outlier Frequency
            plt.figure(figsize=(9, 5))
            sns.barplot(
                x="outliers_iqr_pct",
                y="column",
                data=df_out.sort_values(
                    by="outliers_iqr_pct", ascending=False
                ).head(15),
                hue="column",
                palette="flare",
                legend=False,
            )
            plt.title("Features with Highest Outlier Percentages (IQR Method)")
            plt.xlabel("Outliers (% of records)")
            plt.ylabel("Numeric Features")
            plt.tight_layout()
            plt.savefig(report_dir / "outlier_frequency_chart.png", dpi=100)
            plt.close()

        return df_out

    def validate_consistency(self, report_dir: Path) -> dict[str, Any]:
        """Validates alignment of structures, features, groups, and schemas."""
        report_dir.mkdir(parents=True, exist_ok=True)
        reports = {}

        # 1. Target column presence
        train_target = self.target_col in self.df_train.columns
        test_target = self.target_col in self.df_test.columns
        reports["target_checks"] = {
            "is_target_in_train": train_target,
            "is_target_in_test": test_target,
        }

        # 2. Schema check (missing/extra fields)
        train_only_cols = list(
            set(self.df_train.columns)
            - set(self.df_test.columns)
            - {self.target_col}
        )
        test_only_cols = list(set(self.df_test.columns) - set(self.df_train.columns))
        reports["schema_alignment"] = {
            "mismatch_count": len(train_only_cols) + len(test_only_cols),
            "train_only_features": train_only_cols,
            "test_only_features": test_only_cols,
        }

        # 3. Type mismatches
        type_mismatches = []
        for col in self.common_cols:
            if col in self.df_train.columns and col in self.df_test.columns:
                t1 = str(self.df_train[col].dtype)
                t2 = str(self.df_test[col].dtype)
                if t1 != t2:
                    type_mismatches.append(
                        {"column": col, "train_type": t1, "test_type": t2}
                    )

        reports["type_checks"] = {
            "mismatch_count": len(type_mismatches),
            "mismatches": type_mismatches,
        }

        # 4. Unique TransactionID check
        train_tid_unique = True
        if "TransactionID" in self.df_train.columns:
            train_tid_unique = bool(self.df_train["TransactionID"].is_unique)

        test_tid_unique = True
        if "TransactionID" in self.df_test.columns:
            test_tid_unique = bool(self.df_test["TransactionID"].is_unique)

        reports["identifier_uniqueness"] = {
            "train_transaction_id_unique": train_tid_unique,
            "test_transaction_id_unique": test_tid_unique,
        }

        # 5. Duplicated column names
        dup_names_tr = [
            col
            for col in self.df_train.columns
            if list(self.df_train.columns).count(col) > 1
        ]
        reports["duplicate_names"] = {
            "has_duplicated_column_names": len(dup_names_tr) > 0,
            "duplicated_names": list(set(dup_names_tr)),
        }

        with (report_dir / "consistency_report.json").open("w", encoding="utf-8") as f:
            json.dump(reports, f, indent=4)

        return reports

    def compute_quality_score(
        self, report_dir: Path, metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculates normalized score (0-100) assessing metrics weights."""
        report_dir.mkdir(parents=True, exist_ok=True)

        # 1. Missing Values Score (25% weight)
        missing_pct_tr = metrics.get("missing_pct_train", 0.0)
        missing_score = max(0.0, 100.0 - (missing_pct_tr * 2.0))

        # 2. Duplicate Records Score (15% weight)
        dup_percent = metrics.get("duplicate_trans_pct_train", 0.0)
        dup_score = max(0.0, 100.0 - (dup_percent * 5.0))

        # 3. Invalid Values (20% weight)
        invalid_cnt = metrics.get("invalid_count", 0)
        invalid_score = max(0.0, 100.0 - (invalid_cnt * 10.0))

        # 4. Constant Features (10% weight)
        const_pct = metrics.get("constant_pct", 0.0)
        const_score = max(0.0, 100.0 - (const_pct * 5.0))

        # 5. Near-Constant Features (10% weight)
        near_const_pct = metrics.get("near_constant_pct", 0.0)
        near_const_score = max(0.0, 100.0 - (near_const_pct * 3.0))

        # 6. Outliers (10% weight)
        outlier_pct = metrics.get("outlier_pct", 0.0)
        outlier_score = max(0.0, 100.0 - outlier_pct)

        # 7. Schema Consistency (10% weight)
        mismatch_cnt = metrics.get("schema_mismatches", 0)
        type_mismatch_cnt = metrics.get("type_mismatches", 0)
        consistency_score = max(
            0.0,
            (
                100.0
                - (mismatch_cnt * 10.0 + type_mismatch_cnt * 10.0)
            ),
        )

        # Weighted calculation
        total_score = (
            (missing_score * 0.25)
            + (dup_score * 0.15)
            + (invalid_score * 0.20)
            + (const_score * 0.10)
            + (near_const_score * 0.10)
            + (outlier_score * 0.10)
            + (consistency_score * 0.10)
        )

        rating = "Excellent"
        if total_score < 60.0:
            rating = "Critical"
        elif total_score < 70.0:
            rating = "Poor"
        elif total_score < 80.0:
            rating = "Acceptable"
        elif total_score < 90.0:
            rating = "Good"

        summary = {
            "overall_data_quality_score": round(total_score, 2),
            "rating": rating,
            "breakdown": {
                "missing_values_score": round(missing_score, 2),
                "duplicate_records_score": round(dup_score, 2),
                "invalid_values_score": round(invalid_score, 2),
                "constant_features_score": round(const_score, 2),
                "near_constant_features_score": round(near_const_score, 2),
                "outlier_score": round(outlier_score, 2),
                "schema_consistency_score": round(consistency_score, 2),
            }
        }

        summary_path = report_dir / "data_quality_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        return summary

    def compile_html_report(
        self,
        report_dir: Path,
        summary: dict[str, Any],
        recs: list[dict[str, str]],
        missing_df: pd.DataFrame,
        dup_df: pd.DataFrame,
        const_df: pd.DataFrame,
        near_const_df: pd.DataFrame,
        invalid_df: pd.DataFrame,
        outlier_df: pd.DataFrame,
    ) -> None:
        """HTML compiler generating glassmorphic visual report layouts."""
        score_val = summary["overall_data_quality_score"]
        rating_val = summary["rating"]

        rating_colors = {
            "Excellent": "#13b981",  # Low-sat emerald
            "Good": "#8e97a4",      # Muted platinum
            "Acceptable": "#8e97a4",# Muted platinum
            "Poor": "#d63031",      # Low-sat crimson
            "Critical": "#d63031",  # Low-sat crimson
        }
        accent_color = rating_colors.get(rating_val, "#8e97a4")

        recs_html = ""
        for r in recs:
            recs_html += f"""
            <div class="rec-card">
                <span class="rec-badge">{r['category']}</span>
                <h4>Targets: <code>{r['target']}</code></h4>
                <p>{r['recommendation']}</p>
            </div>
            """

        # Generate HTML tables
        def to_html_table(df: pd.DataFrame, max_rows: int = 15) -> str:
            if df.empty:
                return "<p class='no-data'>No violations flagged.</p>"
            return df.head(max_rows).to_html(
                classes="styled-table", index=False, border=0, justify="left"
            )

        # Sort tables and pre-render them to avoid long lines in HTML template
        missing_sorted = missing_df.sort_values(
            by="missing_pct_train", ascending=False
        )
        missing_table_html = to_html_table(missing_sorted)

        outlier_sorted = outlier_df.sort_values(
            by="outliers_iqr_pct", ascending=False
        )
        outlier_table_html = to_html_table(outlier_sorted)

        const_count = len(const_df)
        const_table_html = to_html_table(const_df)

        near_const_count = len(near_const_df)
        near_const_table_html = to_html_table(near_const_df)

        invalid_table_html = to_html_table(invalid_df)
        dup_table_html = to_html_table(dup_df)

        bd = summary["breakdown"]
        missing_score = bd["missing_values_score"]
        dup_score = bd["duplicate_records_score"]
        invalid_score = bd["invalid_values_score"]
        const_score = bd["constant_features_score"]
        near_const_score = bd["near_constant_features_score"]
        outlier_score = bd["outlier_score"]
        consistency_score = bd["schema_consistency_score"]

        font_link = (
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
    <title>IEEE-CIS Fraud Detection - Data Quality Audit</title>
    <link href="{font_link}" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #06070b;
            --card-bg: rgba(18, 22, 32, 0.45);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-color: #ffffff;
            --text-muted: #8e97a4;
            --accent: {accent_color};
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
            background: var(--bg-color);
            color: var(--text-color);
            font-family: 'Inter', sans-serif;
            padding: 40px;
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
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }}
        header {{
            margin-bottom: 40px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        .header-title-box {{
            display: flex;
            flex-direction: column;
        }}
        h1 {{
            margin: 0;
            font-family: 'Orbitron', sans-serif;
            font-size: 2.2rem;
            font-weight: 900;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #ffffff;
            text-shadow: 0 0 15px rgba(255, 255, 255, 0.15);
        }}
        .subtitle {{
            margin-top: 10px;
            color: var(--text-muted);
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .hud-status {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.03);
            padding: 6px 12px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }}
        .pulse-dot {{
            width: 8px;
            height: 8px;
            background-color: var(--accent);
            border-radius: 50%;
            animation: pulse-grey 2s infinite;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }}
        .glass-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 30px;
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
        }}
        .glass-card h3, .glass-card h4 {{
            font-family: 'Orbitron', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #ffffff;
            margin-bottom: 12px;
        }}
        .score-panel {{
            display: flex;
            align-items: center;
            justify-content: space-around;
        }}
        .score-circle {{
            width: 150px;
            height: 150px;
            border-radius: 50%;
            border: 4px solid var(--accent);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            background: rgba(255, 255, 255, 0.02);
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.03);
        }}
        .score-num {{
            font-size: 3rem;
            font-weight: 900;
            color: #ffffff;
            font-family: 'JetBrains Mono', monospace;
        }}
        .score-lbl {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .score-rating-box {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .score-rating-lbl {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-muted);
        }}
        .score-rating {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.8rem;
            font-weight: 900;
            color: var(--accent);
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}
        .breakdown-list {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .breakdown-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px dashed rgba(255, 255, 255, 0.05);
        }}
        .breakdown-row:last-child {{
            border-bottom: none;
        }}
        .breakdown-row span:first-child {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .breakdown-row span:last-child {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            font-weight: 700;
            color: #ffffff;
        }}

        .rec-container {{
            margin-top: 20px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .rec-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            padding: 20px;
            border-radius: 0;
            border-left: 2px solid #ffffff;
        }}
        .rec-badge {{
            display: inline-block;
            padding: 4px 8px;
            font-size: 0.7rem;
            font-weight: 700;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border-color);
            color: #ffffff;
            border-radius: 0;
            text-transform: uppercase;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.05em;
        }}
        .rec-card h4 {{
            margin: 10px 0 5px 0;
            color: #ffffff;
            font-size: 0.85rem;
            font-family: 'JetBrains Mono', monospace;
        }}
        .rec-card p {{
            margin: 0;
            color: var(--text-muted);
            font-size: 0.85rem;
            line-height: 1.4;
        }}

        .styled-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.85rem;
        }}
        .styled-table th {{
            background: rgba(255, 255, 255, 0.02);
            color: #ffffff;
            padding: 10px 12px;
            text-align: left;
            border-bottom: 2px solid var(--border-color);
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}
        .styled-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-muted);
        }}
        .styled-table td strong {{
            color: #ffffff;
            font-family: 'JetBrains Mono', monospace;
        }}
        .styled-table tr:hover {{
            background: rgba(255, 255, 255, 0.03);
        }}

        .metrics-section {{
            margin-top: 40px;
        }}
        .sect-title {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
            margin-bottom: 20px;
            color: #ffffff;
            font-family: 'Orbitron', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 1.1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .sect-title::after {{
            content: "[SECTION.DATA]";
            font-size: 0.70rem;
            color: var(--text-muted);
        }}
        .chart-box {{
            text-align: center;
            margin-top: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}
        .chart-box img {{
            max-width: 100%;
            border-radius: 0;
            border: 1px solid var(--border-color);
            filter: grayscale(100%) contrast(1.1) brightness(0.9);
        }}
        .no-data {{
            color: var(--text-muted);
            font-style: italic;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="scanline-overlay"></div>
    <div class="hud-grid-bg"></div>
    <div class="container">
        <header>
            <div class="header-title-box">
                <h1>Data Quality Assessment Profile</h1>
                <div class="subtitle">
                    IEEE-CIS Fraud Detection Pipeline diagnostic assessment
                </div>
            </div>
            <div class="hud-status">
                <span class="pulse-dot"></span>
                <span>QUALITY MONITOR READY</span>
            </div>
        </header>

        <div class="dashboard-grid">
            <div class="glass-card score-panel">
                <div class="score-circle">
                    <span class="score-num">{score_val}</span>
                    <span class="score-lbl">Score / 100</span>
                </div>
                <div class="score-rating-box">
                    <span class="score-rating-lbl">Status Level</span>
                    <span class="score-rating">{rating_val}</span>
                </div>
            </div>
            <div class="glass-card">
                <h3>Scores Breakdown</h3>
                <ul class="breakdown-list">
                    <li class="breakdown-row">
                        <span>Missing Values Audit</span>
                        <span>{missing_score}/100</span>
                    </li>
                    <li class="breakdown-row">
                        <span>Duplicate Analytics</span>
                        <span>{dup_score}/100</span>
                    </li>
                    <li class="breakdown-row">
                        <span>Invalid Range Checks</span>
                        <span>{invalid_score}/100</span>
                    </li>
                    <li class="breakdown-row">
                        <span>Zero-Variance Constants</span>
                        <span>{const_score}/100</span>
                    </li>
                    <li class="breakdown-row">
                        <span>Low-Information Near-Constants</span>
                        <span>{near_const_score}/100</span>
                    </li>
                    <li class="breakdown-row">
                        <span>Extremes & Outliers</span>
                        <span>{outlier_score}/100</span>
                    </li>
                    <li class="breakdown-row">
                        <span>Schema Mismatches</span>
                        <span>{consistency_score}/100</span>
                    </li>
                </ul>
            </div>
        </div>

        <div class="glass-card">
            <h3>Actionable Quality Directives</h3>
            <div class="rec-container">
                {recs_html}
            </div>
        </div>

        <div class="hud-grid-bg" style="position: relative; height: 10px;"></div>

        <div class="metrics-section">
            <h3 class="sect-title">Missing Values Audit</h3>
            <div class="dashboard-grid">
                <div class="glass-card">
                    <h4>Top Null Value Features (Train)</h4>
                    {missing_table_html}
                </div>
                <div class="glass-card chart-box">
                    <h4>Top Missing Features Visual</h4>
                    <img src="missing_bar_chart.png" alt="Missing Value Map">
                </div>
            </div>
        </div>

        <div class="metrics-section">
            <h3 class="sect-title">Duplicate Analytics Summary</h3>
            <div class="glass-card">
                {dup_table_html}
            </div>
        </div>

        <div class="metrics-section">
            <h3 class="sect-title">Variance & Information Density</h3>
            <div class="dashboard-grid">
                <div class="glass-card">
                    <h4>Constant Features (Zero-Variance) (Total: {const_count})</h4>
                    {const_table_html}
                </div>
                <div class="glass-card">
                    <h4>
                        Near-Constant Features (Dominant >= 99%)
                        (Total: {near_const_count})
                    </h4>
                    {near_const_table_html}
                </div>
            </div>
        </div>

        <div class="metrics-section">
            <h3 class="sect-title">Extremes and Invalid Distributions</h3>
            <div class="dashboard-grid">
                <div class="glass-card">
                    <h4>Invalid Values Detected</h4>
                    {invalid_table_html}
                </div>
                <div class="glass-card">
                    <h4>Top Outlier Features (IQR)</h4>
                    {outlier_table_html}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        with (report_dir / "data_quality_report.html").open("w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(
            "Saved Data Quality dashboard to %s",
            report_dir / "data_quality_report.html",
        )

