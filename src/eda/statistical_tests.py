"""Part 3.14: Statistical Tests Core Engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, ks_2samp, mannwhitneyu
from sklearn.feature_selection import mutual_info_classif

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalTestsAnalyzer:
    """Performs statistical hypothesis testing (KS, Chi2, Mann-Whitney U, ANOVA), multiple testing corrections, mutual information, effect sizing, and unified significance rank."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_col: str = "isFraud",
        threshold: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.target_col = target_col
        self.threshold = threshold
        self.random_state = random_state

        # Pre-process binary variables
        for c in self.df_train.columns:
            if c != self.target_col:
                col_data = self.df_train[c]
                if pd.api.types.is_object_dtype(col_data) or isinstance(col_data.dtype, pd.CategoricalDtype):
                    if col_data.nunique(dropna=True) == 2:
                        col_str = col_data.astype(str)
                        cats = col_str.dropna().unique()
                        cats = [cat for cat in cats if cat != "nan"]
                        if len(cats) >= 2:
                            mapping = {cats[0]: 0.0, cats[1]: 1.0}
                            self.df_train[c] = col_str.map(mapping).astype(float)
                            if c in self.df_test.columns:
                                self.df_test[c] = self.df_test[c].astype(str).map(mapping).astype(float)

        self._select_candidates()

        rng = np.random.default_rng(self.random_state)
        sample_size_large = min(len(self.df_train), 15000)
        sample_size_small = min(len(self.df_train), 2500)

        indices_large = rng.choice(self.df_train.index, sample_size_large, replace=False)
        self.df_sample_large = self.df_train.loc[indices_large]

        indices_small = rng.choice(self.df_train.index, sample_size_small, replace=False)
        self.df_sample_small = self.df_train.loc[indices_small]

        logger.info(
            "StatisticalTestsAnalyzer initialized. Large sample: %d rows. Small sample: %d rows.",
            len(self.df_sample_large),
            len(self.df_sample_small),
        )

    def _select_candidates(self) -> None:
        """Selects a subset of features to keep runtimes extremely fast."""
        ignore = {"TransactionID", "TransactionDT", "isFraud"}
        candidates = [c for c in self.df_train.columns if c not in ignore]

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

    def analyze_statistical_inventory(self, report_dir: Path) -> pd.DataFrame:
        """Categorizes variables, checks completeness, and establishes eligible statistical tests."""
        logger.info("Generating statistical feature inventory...")
        records = []
        
        # Process numerical candidate features
        for f in self.num_features:
            col_data = self.df_train[f]
            records.append({
                "feature_name": f,
                "data_type": "Numerical",
                "missing_percentage": round(float(col_data.isna().mean() * 100), 4),
                "sample_size": int(col_data.notna().sum()),
                "unique_values": int(col_data.nunique(dropna=True)),
                "eligible_tests": "KS, Mann-Whitney U, ANOVA, Mutual Info",
            })
            
        # Process categorical candidate features
        for f in self.cat_features:
            col_data = self.df_train[f]
            records.append({
                "feature_name": f,
                "data_type": "Categorical",
                "missing_percentage": round(float(col_data.isna().mean() * 100), 4),
                "sample_size": int(col_data.notna().sum()),
                "unique_values": int(col_data.nunique(dropna=True)),
                "eligible_tests": "Chi-Square, ANOVA (group), Mutual Info",
            })

        df = pd.DataFrame(records)
        df.to_csv(report_dir / "statistical_feature_inventory.csv", index=False)

        metadata = {
            "total_analyzed_features": len(self.num_features) + len(self.cat_features),
            "numerical_features_count": len(self.num_features),
            "categorical_features_count": len(self.cat_features),
            "average_missing_percentage": round(float(df["missing_percentage"].mean()), 4) if not df.empty else 0.0,
        }
        with open(report_dir / "statistical_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        return df

    def analyze_ks_test(self, report_dir: Path) -> pd.DataFrame:
        """Performs two-sample Kolmogorov-Smirnov test between fraud vs legit distributions."""
        logger.info("Computing Kolmogorov-Smirnov statistics...")
        records = []
        
        y = self.df_sample_large[self.target_col]
        mask_fraud = (y == 1)
        mask_legit = (y == 0)
        n_fraud = mask_fraud.sum()
        n_legit = mask_legit.sum()

        # Critical value calculation for KS test alpha=0.05
        # D_crit = 1.36 * sqrt((n1 + n2)/(n1 * n2))
        if n_fraud > 0 and n_legit > 0:
            crit_val = 1.36 * np.sqrt((n_fraud + n_legit) / (n_fraud * n_legit))
        else:
            crit_val = 0.0

        for col in self.num_features:
            x_col = self.df_sample_large[col]
            x_fraud = x_col[mask_fraud].dropna()
            x_legit = x_col[mask_legit].dropna()

            if len(x_fraud) > 5 and len(x_legit) > 5:
                try:
                    res = ks_2samp(x_fraud.values, x_legit.values)
                    ks_stat = float(res.statistic)
                    p_val = float(res.pvalue)
                except Exception as e:
                    logger.warning("KS test failed on %s: %s", col, e)
                    ks_stat, p_val = 0.0, 1.0
            else:
                ks_stat, p_val = 0.0, 1.0

            records.append({
                "feature_name": col,
                "ks_statistic": round(ks_stat, 6),
                "p_value": round(p_val, 6),
                "critical_value": round(crit_val, 6),
                "significant": "Yes" if p_val < self.threshold and ks_stat > crit_val else "No",
                "distribution_difference": round(ks_stat, 6),
            })
            
        df = pd.DataFrame(records).sort_values(by="ks_statistic", ascending=False)
        df.to_csv(report_dir / "ks_results.csv", index=False)

        # Generate plot showing distributions of top shifted numeric feature
        if not df.empty and n_fraud > 0 and n_legit > 0:
            top_col = df.iloc[0]["feature_name"]
            x_col = self.df_sample_large[top_col]
            
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#06070b")
            ax.set_facecolor("#06070b")
            
            vals_legit = x_col[mask_legit].dropna()
            vals_fraud = x_col[mask_fraud].dropna()

            # Clean extreme outliers for plotting
            if not vals_legit.empty and not vals_fraud.empty:
                q99 = x_col.quantile(0.99)
                q01 = x_col.quantile(0.01)
                vals_l_clean = vals_legit[(vals_legit >= q01) & (vals_legit <= q99)]
                vals_f_clean = vals_fraud[(vals_fraud >= q01) & (vals_fraud <= q99)]
                
                ax.hist(vals_l_clean.values, bins=25, density=True, color="#8e97a4", alpha=0.3, label="Legit")
                ax.hist(vals_f_clean.values, bins=25, density=True, color="#ffffff", alpha=0.8, label="Fraud", edgecolor="red", linewidth=0.5)

            ax.set_title(f"[KS SHIFT] Distribution diff: {top_col}", color="#fff", fontname="Orbitron", fontsize=9)
            ax.set_xlabel(top_col, color="#8e97a4", fontsize=8)
            ax.tick_params(colors="#8e97a4", labelsize=7)
            ax.spines[:].set_color((1.0, 1.0, 1.0, 0.08))
            ax.legend(facecolor="#06070b", edgecolor=(1.0, 1.0, 1.0, 0.08), fontsize=8)

            (report_dir / "plots").mkdir(parents=True, exist_ok=True)
            fig.savefig(report_dir / "plots" / "ks_distribution_shifts.png", dpi=110, facecolor="#06070b")
            plt.close(fig)

        return df

    def analyze_chi_square_test(self, report_dir: Path) -> pd.DataFrame:
        """Determines relationship of categorical features to target and computes Cramér's V."""
        logger.info("Computing Chi-Square association statistics...")
        records = []
        
        y = self.df_sample_large[self.target_col]

        for col in self.cat_features:
            x_col = self.df_sample_large[col].astype(str).fillna("Missing")
            
            # Contingency table
            c_matrix = pd.crosstab(x_col, y)
            
            if c_matrix.shape[0] > 1 and c_matrix.shape[1] > 1:
                try:
                    res = chi2_contingency(c_matrix)
                    chi2_stat = float(res[0])
                    p_val = float(res[1])
                    dof = int(res[2])
                    expected = res[3]
                    
                    # Cramér's V effect size
                    n = int(c_matrix.values.sum())
                    r, c_dim = c_matrix.shape
                    cramer_v = float(np.sqrt(chi2_stat / (n * min(r - 1, c_dim - 1))))
                except Exception as e:
                    logger.warning("Chi-square contingency failing on %s: %s", col, e)
                    chi2_stat, p_val, dof, cramer_v = 0.0, 1.0, 1, 0.0
                    expected = np.array([])
            else:
                chi2_stat, p_val, dof, cramer_v = 0.0, 1.0, 1, 0.0
                expected = np.array([])

            records.append({
                "feature_name": col,
                "chi_square_statistic": round(chi2_stat, 6),
                "p_value": round(p_val, 6),
                "degrees_of_freedom": dof,
                "cramers_v_effect_size": round(cramer_v, 6),
                "significant": "Yes" if p_val < self.threshold else "No",
            })
            
        df = pd.DataFrame(records).sort_values(by="chi_square_statistic", ascending=False)
        df.to_csv(report_dir / "chi_square_results.csv", index=False)
        return df

    def analyze_mann_whitney_test(self, report_dir: Path) -> pd.DataFrame:
        """Runs non-parametric class comparisons and calculates rank biserial values."""
        logger.info("Computing Mann-Whitney U rank-sum statistics...")
        records = []
        
        y = self.df_sample_small[self.target_col]
        mask_fraud = (y == 1)
        mask_legit = (y == 0)
        n_fraud = int(mask_fraud.sum())
        n_legit = int(mask_legit.sum())

        for col in self.num_features:
            x_col = self.df_sample_small[col]
            x_fraud = x_col[mask_fraud].dropna()
            x_legit = x_col[mask_legit].dropna()
            
            n_f = len(x_fraud)
            n_l = len(x_legit)

            if n_f > 5 and n_l > 5:
                try:
                    res = mannwhitneyu(x_fraud.values, x_legit.values, alternative="two-sided")
                    u_stat = float(res.statistic)
                    p_val = float(res.pvalue)
                    
                    # Estimate rank difference / rank biserial correlation
                    # r = 1 - (2 * U / (n1 * n2))
                    r_biserial = float(1.0 - (2.0 * u_stat) / (n_f * n_l))
                except Exception as e:
                    logger.warning("Mann-Whitney U failed on %s: %s", col, e)
                    u_stat, p_val, r_biserial = 0.0, 1.0, 0.0
            else:
                u_stat, p_val, r_biserial = 0.0, 1.0, 0.0

            # Rank difference representation
            rank_diff = abs(r_biserial)

            records.append({
                "feature_name": col,
                "u_statistic": round(u_stat, 6),
                "p_value": round(p_val, 6),
                "rank_difference": round(rank_diff, 6),
                "effect_size": round(r_biserial, 6),
                "significant": "Yes" if p_val < self.threshold else "No",
            })
            
        df = pd.DataFrame(records).sort_values(by="rank_difference", ascending=False)
        df.to_csv(report_dir / "mann_whitney_results.csv", index=False)
        return df

    def analyze_anova(self, report_dir: Path) -> pd.DataFrame:
        """Determines variance shifts of numerical attributes across major category partitions."""
        logger.info("Computing Analysis of Variance (ANOVA)...")
        records = []
        
        # Test TransactionAmt continuous variation against the categorical columns of interest
        tr_amt = self.df_sample_large["TransactionAmt"].fillna(self.df_sample_large["TransactionAmt"].median())
        
        for cat_col in self.cat_features:
            cat_series = self.df_sample_large[cat_col].astype(str).fillna("Missing")
            groups = cat_series.unique()
            
            group_lists = []
            for g in groups:
                group_lists.append(tr_amt[cat_series == g].values)
                
            if len(group_lists) > 1 and all(len(grp) > 5 for grp in group_lists):
                try:
                    f_res = f_oneway(*group_lists)
                    f_stat = float(f_res.statistic)
                    p_val = float(f_res.pvalue)

                    # Compute Eta Squared: SSB / (SSB + SSW)
                    total_mean = tr_amt.mean()
                    ssb = sum(len(grp) * (grp.mean() - total_mean)**2 for grp in group_lists)
                    ssw = sum(((grp - grp.mean())**2).sum() for grp in group_lists)
                    eta_sq = float(ssb / (ssb + ssw)) if (ssb + ssw) > 0 else 0.0
                except Exception as e:
                    logger.warning("ANOVA failing on %s: %s", cat_col, e)
                    f_stat, p_val, eta_sq = 0.0, 1.0, 0.0
            else:
                f_stat, p_val, eta_sq = 0.0, 1.0, 0.0
                
            records.append({
                "group_categorical": cat_col,
                "tested_numerical": "TransactionAmt",
                "f_statistic": round(f_stat, 6),
                "p_value": round(p_val, 6),
                "eta_squared_effect_size": round(eta_sq, 6),
                "significant": "Yes" if p_val < self.threshold else "No",
            })
            
        df = pd.DataFrame(records).sort_values(by="f_statistic", ascending=False)
        df.to_csv(report_dir / "anova_results.csv", index=False)
        return df

    def analyze_mutual_information(self, report_dir: Path) -> pd.DataFrame:
        """Finds nonlinear dependencies using sklearn mutual information calculations."""
        logger.info("Computing Mutual Information rankings...")
        
        # Combine numerical and categorical features
        all_cols = self.num_features + self.cat_features
        if not all_cols:
            return pd.DataFrame()
            
        df_clean = self.df_sample_small[all_cols].copy()
        
        # Clean columns to ensure it doesn't fail
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                mean_val = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(mean_val if pd.notna(mean_val) else 0.0)
            else:
                df_clean[col] = df_clean[col].astype(str).fillna("Missing")
                # Object attributes require numerical encoding for MI
                df_clean[col] = pd.factorize(df_clean[col])[0]
                
        y = self.df_sample_small[self.target_col].fillna(0).values
        
        try:
            mi_scores = mutual_info_classif(df_clean.values, y, random_state=self.random_state)
            mi_map = dict(zip(all_cols, mi_scores))
        except Exception as e:
            logger.warning("Mutual Info calculations failed: %s", e)
            mi_map = {c: 0.0 for c in all_cols}

        records = []
        max_mi = max(mi_map.values()) if mi_map.values() and max(mi_map.values()) > 0 else 1.0

        for col, score in mi_map.items():
            records.append({
                "feature_name": col,
                "mutual_information_score": round(float(score), 6),
                "normalized_mutual_information": round(float(score / max_mi), 6),
                "relative_importance": round(float(score / max_mi), 6),
            })
            
        df = pd.DataFrame(records).sort_values(by="mutual_information_score", ascending=False)
        df.to_csv(report_dir / "mutual_information_results.csv", index=False)

        # Plot output
        if not df.empty:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(6, 5), facecolor="#06070b")
            ax.set_facecolor("#06070b")
            
            top_ten = df.head(10)
            y_pos = np.arange(len(top_ten))
            ax.barh(y_pos, top_ten["mutual_information_score"].values[::-1], color="#ffffff", height=0.5, edgecolor="red", linewidth=0.5)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_ten["feature_name"].values[::-1], fontsize=7, color="#8e97a4")
            ax.set_title("[RELEVANCE] Non-linear Mutual Information Scores", color="#ffffff", fontname="Orbitron", fontsize=9)
            ax.set_xlabel("MI Score", color="#8e97a4", fontsize=8)
            ax.tick_params(colors="#8e97a4", labelsize=7)
            ax.spines[:].set_color((1.0, 1.0, 1.0, 0.08))

            (report_dir / "plots").mkdir(parents=True, exist_ok=True)
            fig.savefig(report_dir / "plots" / "mi_relevance_plot.png", dpi=110, facecolor="#06070b")
            plt.close(fig)

        return df

    def analyze_multiple_testing_correction(
        self,
        report_dir: Path,
        p_values_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Performs Bonferroni, Holm, & Benjamini-Hochberg FDR adjustments over combined tests p-values."""
        logger.info("Computing multiple hypothesis testing corrections...")
        
        if p_values_df.empty:
            return pd.DataFrame()
            
        raw_p = p_values_df["raw_p_value"].values
        
        def _manual_multipletests(pvals: np.ndarray, alpha: float, method: str) -> tuple[np.ndarray, np.ndarray]:
            n = len(pvals)
            if n == 0:
                return np.array([]), np.array([])
            pvals = np.asarray(pvals)
            if method == "bonferroni":
                adjusted = np.clip(pvals * n, 0.0, 1.0)
                significant = (adjusted < alpha)
                return significant, adjusted
            elif method == "holm":
                sort_idx = np.argsort(pvals)
                ranked_p = pvals[sort_idx]
                adjusted = np.zeros(n)
                for i in range(n):
                    adjusted[i] = ranked_p[i] * (n - i)
                for i in range(1, n):
                    adjusted[i] = max(adjusted[i], adjusted[i - 1])
                adjusted = np.clip(adjusted, 0.0, 1.0)
                unsort_idx = np.argsort(sort_idx)
                adj_unsorted = adjusted[unsort_idx]
                significant = (adj_unsorted < alpha)
                return significant, adj_unsorted
            elif method == "fdr_bh":
                sort_idx = np.argsort(pvals)
                ranked_p = pvals[sort_idx]
                adjusted = np.zeros(n)
                for i in range(n):
                    adjusted[i] = ranked_p[i] * n / (i + 1)
                for i in range(n - 2, -1, -1):
                    adjusted[i] = min(adjusted[i], adjusted[i + 1])
                adjusted = np.clip(adjusted, 0.0, 1.0)
                unsort_idx = np.argsort(sort_idx)
                adj_unsorted = adjusted[unsort_idx]
                significant = (adj_unsorted < alpha)
                return significant, adj_unsorted
            else:
                raise ValueError(f"Unknown correction method {method}")
        
        # Apply corrections
        # Bonferroni
        r_bonf = _manual_multipletests(raw_p, alpha=self.threshold, method="bonferroni")
        # Holm-Bonferroni
        r_holm = _manual_multipletests(raw_p, alpha=self.threshold, method="holm")
        # Benjamini-Hochberg FDR
        r_fdr = _manual_multipletests(raw_p, alpha=self.threshold, method="fdr_bh")

        corrected_df = p_values_df.copy()
        corrected_df["bonferroni_p_value"] = np.round(r_bonf[1], 6)
        corrected_df["holm_p_value"] = np.round(r_holm[1], 6)
        corrected_df["fdr_p_value"] = np.round(r_fdr[1], 6)
        
        corrected_df["bonferroni_significant"] = np.where(r_bonf[0], "Yes", "No")
        corrected_df["holm_significant"] = np.where(r_holm[0], "Yes", "No")
        corrected_df["fdr_significant"] = np.where(r_fdr[0], "Yes", "No")

        corrected_df.to_csv(report_dir / "multiple_testing_correction.csv", index=False)
        return corrected_df

    def analyze_effect_size(self, report_dir: Path) -> pd.DataFrame:
        """Determines standardized Cohen d, Cliff delta shifts over continuous attributes."""
        logger.info("Computing Cohen's d and Cliff's Delta effect sizes...")
        records = []
        
        y = self.df_sample_large[self.target_col]
        mask_fraud = (y == 1)
        mask_legit = (y == 0)

        for col in self.num_features:
            x_col = self.df_sample_large[col]
            x_fr = x_col[mask_fraud].dropna().values
            x_le = x_col[mask_legit].dropna().values
            
            n_f, n_l = len(x_fr), len(x_le)
            if n_f > 5 and n_l > 5:
                # 1. Cohen's d
                mean_f, mean_l = x_fr.mean(), x_le.mean()
                var_f, var_l = x_fr.var(ddof=1), x_le.var(ddof=1)
                pooled_sd = np.sqrt(((n_f - 1) * var_f + (n_l - 1) * var_l) / (n_f + n_l - 2))
                cohen_d = float((mean_f - mean_l) / pooled_sd) if pooled_sd > 0 else 0.0

                # 2. Cliff's Delta (non-parametric comparison)
                # Compute using matrix operations to remain vectorized
                # delta = (sum(sign(x_i - y_j))) / (n1 * n2)
                try:
                    diff_matrix = np.sign(x_fr[:, None] - x_le)
                    cliffs_delta = float(diff_matrix.mean())
                except Exception:
                    cliffs_delta = 0.0
            else:
                cohen_d, cliffs_delta = 0.0, 0.0

            records.append({
                "feature_name": col,
                "cohens_d": round(cohen_d, 6),
                "cliffs_delta": round(cliffs_delta, 6),
                "practical_magnitude": "Negligible" if abs(cliffs_delta) < 0.147 else "Small" if abs(cliffs_delta) < 0.33 else "Medium" if abs(cliffs_delta) < 0.474 else "Large",
            })
            
        df = pd.DataFrame(records).sort_values(by="cliffs_delta", key=abs, ascending=False)
        df.to_csv(report_dir / "effect_size_analysis.csv", index=False)
        return df

    def rank_statistical_significance(
        self,
        report_dir: Path,
        corrected_df: pd.DataFrame,
        mi_df: pd.DataFrame,
        ks_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Determines class separation value rank based on p-value indices, KS shift scale, and MI."""
        logger.info("Computing Statistical Significance rankings...")
        
        # Merge all metrics
        # corrected_df keys feature_name, raw_p_value, fdr_p_value, test_type
        # mi_df keys feature_name, mutual_information_score
        # ks_df keys feature_name, ks_statistic
        
        mi_map = {}
        if not mi_df.empty:
            mi_map = dict(zip(mi_df["feature_name"], mi_df["mutual_information_score"]))
            
        ks_map = {}
        if not ks_df.empty:
            ks_map = dict(zip(ks_df["feature_name"], ks_df["ks_statistic"]))

        records = []
        for _, row in corrected_df.iterrows():
            f = row["feature_name"]
            p_val = float(row["raw_p_value"])
            fdr_p = float(row["fdr_p_value"])
            t_type = row["test_type"]
            
            mi = float(mi_map.get(f, 0.0))
            ks = float(ks_map.get(f, 0.0))
            
            # Rank score = -log10(p_val + 1e-15) + (50 * MI) + (10 * KS)
            score = -np.log10(p_val + 1e-15) + (100.0 * mi) + (10.0 * ks)
            
            if fdr_p < 0.01:
                cat = "HIGHLY SIGNIFICANT"
            elif fdr_p < 0.05:
                cat = "MODERATELY SIGNIFICANT"
            else:
                cat = "NOT SIGNIFICANT"
                
            records.append({
                "feature_name": f,
                "test_type": t_type,
                "raw_p_value": p_val,
                "fdr_adjusted_p_value": fdr_p,
                "mutual_information": mi,
                "ks_statistic_if_num": ks if t_type == "KS (Numerical)" else np.nan,
                "significance_rank_score": round(score, 4),
                "significance_category": cat,
            })
            
        df = pd.DataFrame(records).sort_values(by="significance_rank_score", ascending=False)
        df.to_csv(report_dir / "statistical_significance_ranking.csv", index=False)

        # Generate automated feature selector recommendation recipe
        rec_records = []
        for _, row in df.iterrows():
            f = row["feature_name"]
            cat = row["significance_category"]
            mi = row["mutual_information"]
            
            if cat == "HIGHLY SIGNIFICANT" and mi > 0.005:
                rec_act = "RETAIN"
                rec_reason = "Highly statistically significant representation & positive information gain"
            elif cat == "NOT SIGNIFICANT":
                rec_act = "PRUNE"
                rec_reason = "Not statistically significant under FDR multiple testing corrections"
            else:
                rec_act = "MONITOR"
                rec_reason = "Marginally significant; monitor distribution drift in downstream validation splits"

            rec_records.append({
                "feature_name": f,
                "statistical_significance": cat,
                "recommended_action": rec_act,
                "rational": rec_reason,
            })
            
        df_rec = pd.DataFrame(rec_records)
        df_rec.to_csv(report_dir / "statistical_feature_recommendations.csv", index=False)

        return df

    def compile_html_dashboard(
        self,
        report_dir: Path,
        df_inventory: pd.DataFrame,
        df_ks: pd.DataFrame,
        df_chi: pd.DataFrame,
        df_anova: pd.DataFrame,
        df_mi: pd.DataFrame,
        df_corrected: pd.DataFrame,
        df_effect: pd.DataFrame,
        df_rank: pd.DataFrame,
    ) -> None:
        """HTML compiler generating glassmorphic HUD summary dashboard."""
        logger.info("Compiling Statistical Tests HTML report...")

        def _to_html(df: pd.DataFrame) -> str:
            if df.empty:
                return "<div class='no-data'>NO STATISTICAL TEST RESULTS REPORTED</div>"
            return df.to_html(
                classes="hud-table",
                index=False,
                border=0,
                justify="left",
            )
            
        summary = {
            "total_tested": len(df_rank),
            "highly_significant": int((df_rank["significance_category"] == "HIGHLY SIGNIFICANT").sum()),
            "max_mi": float(df_mi["mutual_information_score"].max()) if not df_mi.empty else 0.0,
            "max_ks": float(df_ks["ks_statistic"].max()) if not df_ks.empty else 0.0,
        }

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>IEEE-CIS STATISTICAL ANALYSIS REPORT</title>
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
            font-size: 0.7rem;
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
                'plots/ks_distribution_shifts.png',
                'plots/mi_relevance_plot.png'
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
            <h1>IEEE-CIS STATISTICAL ANALYSIS REPORT</h1>
            <p style="font-size: 0.65rem; color: var(--text-color); margin-top: 0.25rem; letter-spacing: 1px;">STAGE 3.14: SIGNIFICANCE TESTING PROPERTY MAP</p>
        </div>
        <div class="status-pill">STATISTICAL ENGINE OK</div>
    </header>

    <div class="hud-grid">
        <div class="hud-panel">
            <p class="metric-label">Analyzed Features</p>
            <p class="metric-value">{summary['total_tested']}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Highly Significant (FDR)</p>
            <p class="metric-value">{summary['highly_significant']}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Max Mutual Information</p>
            <p class="metric-value">{summary['max_mi']:.4f}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Max Kolmogorov-Smirnov</p>
            <p class="metric-value">{summary['max_ks']:.4f}</p>
        </div>
    </div>

    <div class="dashboard-body">
        <div class="hud-panel visualizer-card">
            <h2>DIAGNOSTIC VISUALIZATIONS</h2>
            <div class="carousel-tabs">
                <button class="carousel-tab active" onclick="switchTab(0)">KS Distribution Shift</button>
                <button class="carousel-tab" onclick="switchTab(1)">MI Relevance Rank</button>
            </div>
            <div class="carousel-content">
                <img id="carousel-img" class="carousel-img" src="plots/ks_distribution_shifts.png" alt="KS Shift">
            </div>
        </div>

        <div class="hud-panel">
            <h2>SIGNIFICANCE RANKING (CORRECTED FDR)</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_rank.head(30))}
            </div>
        </div>
    </div>

    <div class="secondary-panel-grid">
        <div class="hud-panel">
            <h2>MULTIPLE TESTING CORRECTIONS</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_corrected.head(30))}
            </div>
        </div>
        <div class="hud-panel">
            <h2>EFFECT SIZE ANALYSIS</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_effect.head(30))}
            </div>
        </div>
    </div>

</body>
</html>
"""
        with open(report_dir / "statistical_tests_report.html", "w") as f:
            f.write(html_template)
        logger.info("Compiled Statistical HTML report saved.")

    def analyze_all(self, report_dir: Path) -> None:
        """Executes full Statistical testing pipeline."""
        report_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = report_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("--- Stage 3.14: Statistical Analysis (Starting) ---")

        df_inv = self.analyze_statistical_inventory(report_dir)
        df_ks = self.analyze_ks_test(report_dir)
        df_chi = self.analyze_chi_square_test(report_dir)
        df_anova = self.analyze_anova(report_dir)
        df_mw = self.analyze_mann_whitney_test(report_dir)
        df_mi = self.analyze_mutual_information(report_dir)
        df_effect = self.analyze_effect_size(report_dir)

        # Combine all computed p-values for FDR adjustment
        p_val_records = []
        
        # KS Numerical
        for _, row in df_ks.iterrows():
            p_val_records.append({
                "feature_name": row["feature_name"],
                "raw_p_value": row["p_value"],
                "test_type": "KS (Numerical)",
            })
            
        # Chi Square Categorical
        for _, row in df_chi.iterrows():
            p_val_records.append({
                "feature_name": row["feature_name"],
                "raw_p_value": row["p_value"],
                "test_type": "Chi-Square (Categorical)",
            })

        df_pvals = pd.DataFrame(p_val_records)
        df_corrected = self.analyze_multiple_testing_correction(report_dir, df_pvals)

        df_rank = self.rank_statistical_significance(
            report_dir,
            df_corrected,
            df_mi,
            df_ks,
        )

        self.compile_html_dashboard(
            report_dir,
            df_inv,
            df_ks,
            df_chi,
            df_anova,
            df_mi,
            df_corrected,
            df_effect,
            df_rank,
        )

        logger.info("--- Stage 3.14: Statistical Analysis Complete ---")
