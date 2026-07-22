"""Part 3.13: Feature Interaction Analysis Engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.feature_selection import mutual_info_classif

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureInteractionAnalyzer:
    """Computes pairwise and multi-variable interactions, strength metrics, stability drift, and recommendations."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_col: str = "isFraud",
        random_state: int = 42,
    ) -> None:
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.target_col = target_col
        self.random_state = random_state

        # Pre-process columns: encode binary variables
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

        self.features = self._select_candidates()

        rng = np.random.default_rng(self.random_state)
        sample_size_large = min(len(self.df_train), 15000)
        sample_size_small = min(len(self.df_train), 2500)
        
        indices_large = rng.choice(self.df_train.index, sample_size_large, replace=False)
        self.df_sample_large = self.df_train.loc[indices_large]

        indices_small = rng.choice(self.df_train.index, sample_size_small, replace=False)
        self.df_sample_small = self.df_train.loc[indices_small]

        logger.info(
            "FeatureInteractionAnalyzer initialized. Candidates selected: %d. Small Sample: %d rows.",
            len(self.features),
            len(self.df_sample_small),
        )

    def _select_candidates(self) -> list[str]:
        """Prioritizes features of interest across groups based on variance and data types."""
        df = self.df_train
        ignore = {"TransactionID", "TransactionDT", "isFraud", "DeviceInfo", "DeviceType"}
        candidates = [c for c in df.columns if c not in ignore]
        
        # Filter for numeric or binary encoded features only
        valid_cols = []
        for c in candidates:
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1:
                valid_cols.append(c)

        # To keep combinatorial pairs manageable, we select the top 20 candidate features by variance and mutual information subset
        if len(valid_cols) <= 20:
            return valid_cols
            
        variances = df[valid_cols].var().fillna(0)
        top_variance_cols = list(variances.sort_values(ascending=False).head(30).index)
        
        # Downselect using mutual information on small sample
        df_sample = df.sample(min(len(df), 2000), random_state=self.random_state)
        X = df_sample[top_variance_cols].fillna(0)
        y = df_sample[self.target_col].fillna(0)
        
        try:
            mi_scores = mutual_info_classif(X.values, y.values, random_state=self.random_state)
            mi_series = pd.Series(mi_scores, index=top_variance_cols)
            return list(mi_series.sort_values(ascending=False).head(15).index)
        except Exception:
            return top_variance_cols[:15]

    def analyze_interaction_inventory(self, report_dir: Path) -> pd.DataFrame:
        """Saves interaction candidate feature inventory details."""
        logger.info("Generating interaction feature inventory...")
        records = []
        for col in self.features:
            col_data = self.df_train[col]
            missing_pct = float(col_data.isna().mean() * 100)
            
            # Map clean types
            if pd.api.types.is_float_dtype(col_data.dtype):
                t = "Float"
            elif pd.api.types.is_integer_dtype(col_data.dtype):
                t = "Integer"
            elif col_data.nunique() <= 2:
                t = "Binary"
            else:
                t = "Numerical"

            train_avail = len(col_data.dropna()) > 0
            test_avail = col in self.df_test.columns and len(self.df_test[col].dropna()) > 0

            records.append({
                "feature_name": col,
                "feature_type": t,
                "missing_percentage": round(missing_pct, 4),
                "data_availability": "Both" if (train_avail and test_avail) else "Train Only" if train_avail else "None",
            })
            
        df = pd.DataFrame(records)
        df.to_csv(report_dir / "interaction_feature_inventory.csv", index=False)

        metadata = {
            "total_candidate_interaction_features": len(self.features),
            "feature_types": df["feature_type"].value_counts().to_dict(),
            "average_missing_percentage": round(float(df["missing_percentage"].mean()), 4),
        }
        with open(report_dir / "interaction_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        return df

    def analyze_pairplots(self, report_dir: Path) -> pd.DataFrame:
        """Analyzes class separation statistics of linear/non-linear pairings."""
        logger.info("Computing pair plot separation statistics...")
        records = []
        cols = self.features
        
        # Take pairs of top candidates
        pairs_to_eval = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs_to_eval.append((cols[i], cols[j]))

        # Limit to top 30 pairs for detailed assessment
        pairs_to_eval = pairs_to_eval[:30]
        y = self.df_sample_large[self.target_col]

        for c1, c2 in pairs_to_eval:
            x1 = self.df_sample_large[c1].fillna(0)
            x2 = self.df_sample_large[c2].fillna(0)
            
            # Estimate separation metric: Wasserstein/Distance of centroid between class 0 and 1
            mask_fraud = (y == 1)
            mask_legit = (y == 0)
            
            if mask_fraud.sum() > 5 and mask_legit.sum() > 5:
                ctr_fraud = [x1[mask_fraud].mean(), x2[mask_fraud].mean()]
                ctr_legit = [x1[mask_legit].mean(), x2[mask_legit].mean()]
                dist = float(np.sqrt((ctr_fraud[0] - ctr_legit[0])**2 + (ctr_fraud[1] - ctr_legit[1])**2))
            else:
                dist = 0.0

            # Correlation coefficient
            corr = float(np.corrcoef(x1, x2)[0, 1]) if x1.var() > 0 and x2.var() > 0 else 0.0
            
            records.append({
                "feature_1": c1,
                "feature_2": c2,
                "class_separation_distance": round(dist, 4),
                "pearson_correlation": round(corr, 4),
                "density_overlap_index": round(1.0 / (1.0 + dist), 4),  # higher separation = lower overlap
            })
            
        df = pd.DataFrame(records).sort_values(by="class_separation_distance", ascending=False)
        df.to_csv(report_dir / "pairplot_analysis.csv", index=False)

        # Plot the highest separation pair
        if not df.empty:
            top_pair = df.iloc[0]
            f1, f2 = top_pair["feature_1"], top_pair["feature_2"]
            
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(6, 5), facecolor="#06070b")
            ax.set_facecolor("#06070b")
            
            y_large = self.df_sample_large[self.target_col]
            x1_vals = self.df_sample_large[f1].fillna(self.df_sample_large[f1].median())
            x2_vals = self.df_sample_large[f2].fillna(self.df_sample_large[f2].median())
            
            # Plot legit (grey) and fraud (white)
            ax.scatter(x1_vals[y_large == 0], x2_vals[y_large == 0], color="#8e97a4", alpha=0.3, s=12, label="Legit")
            ax.scatter(x1_vals[y_large == 1], x2_vals[y_large == 1], color="#ffffff", alpha=0.9, s=20, label="Fraud", edgecolors="red", linewidths=0.5)
            
            ax.set_xlabel(f1, color="#8e97a4", fontsize=8)
            ax.set_ylabel(f2, color="#8e97a4", fontsize=8)
            ax.set_title(f"[PAIRWISE] Class Separation: {f1} x {f2}", color="#fff", fontname="Orbitron", fontsize=9)
            ax.tick_params(colors="#8e97a4", labelsize=7)
            ax.spines[:].set_color((1.0, 1.0, 1.0, 0.08))
            ax.legend(facecolor="#06070b", edgecolor=(1.0, 1.0, 1.0, 0.08), fontsize=8)
            
            plt.tight_layout()
            fig.savefig(report_dir / "plots" / "pairplot_scatter.png", dpi=110, facecolor="#06070b")
            plt.close(fig)

        return df

    def _bin_column(self, col_data: pd.Series) -> pd.Series:
        """Utility bins a numerical column to 4 unique values safely."""
        if col_data.nunique(dropna=True) <= 4:
            return col_data.fillna("Missing").astype(str)
        try:
            return pd.qcut(col_data, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop").astype(str).fillna("Missing")
        except Exception:
            return pd.cut(col_data, bins=4, labels=["B1", "B2", "B3", "B4"]).astype(str).fillna("Missing")

    def analyze_cross_features(self, report_dir: Path) -> pd.DataFrame:
        """Analyzes joint frequencies, fraud rates, and lift scores."""
        logger.info("Computing cross feature analysis...")
        records = []
        cols = self.features
        
        # Take pairs of top candidates
        pairs_to_eval = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs_to_eval.append((cols[i], cols[j]))

        pairs_to_eval = pairs_to_eval[:20]
        y = self.df_sample_large[self.target_col]
        global_fraud_rate = float(y.mean())
        if global_fraud_rate == 0:
            global_fraud_rate = 0.035

        for c1, c2 in pairs_to_eval:
            # Bin both attributes to cross them
            bin_1 = self._bin_column(self.df_sample_large[c1])
            bin_2 = self._bin_column(self.df_sample_large[c2])
            
            joint = bin_1 + "_" + bin_2
            df_joint = pd.DataFrame({"joint": joint, "isFraud": y})
            
            # Group by joint category
            counts = df_joint.groupby("joint")["isFraud"].agg(["count", "mean"])
            
            # Select the sub-combination with the highest fraud rate (support >= 15)
            filtered = counts[counts["count"] >= 15]
            if not filtered.empty:
                hp = filtered.sort_values(by="mean", ascending=False).iloc[0]
                fraud_rate = float(hp["mean"])
                cnt = int(hp["count"])
                lift = fraud_rate / global_fraud_rate
            else:
                fraud_rate = 0.0
                cnt = 0
                lift = 0.0

            # Calculate relative risk
            # RR = Fraud rate in hotspot / Fraud rate outside hotspot
            outside = counts.drop(index=hp.name) if (cnt > 0 and len(counts) > 1) else counts
            outside_rate = float(outside["mean"].mean()) if not outside.empty else global_fraud_rate
            rr = fraud_rate / outside_rate if outside_rate > 0 else 1.0

            records.append({
                "feature_1": c1,
                "feature_2": c2,
                "hotspot_combination": hp.name if cnt > 0 else "N/A",
                "combination_count": cnt,
                "hotspot_fraud_rate": round(fraud_rate, 4),
                "lift_score": round(lift, 4),
                "relative_risk": round(rr, 4),
            })
            
        df = pd.DataFrame(records).sort_values(by="lift_score", ascending=False)
        df.to_csv(report_dir / "cross_feature_analysis.csv", index=False)
        return df

    def analyze_fraud_interactions(self, report_dir: Path) -> pd.DataFrame:
        """Focuses on high-risk feature combinations and maps hotspots."""
        logger.info("Analyzing fraud interactions...")
        records = []
        cols = self.features
        
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j]))
        
        pairs = pairs[:20]
        y = self.df_sample_large[self.target_col]
        total_fraud = int(y.sum())
        total_legit = int(len(y) - total_fraud)
        
        for c1, c2 in pairs:
            # Cross binning
            b1 = self._bin_column(self.df_sample_large[c1])
            b2 = self._bin_column(self.df_sample_large[c2])
            joint = b1 + "_" + b2
            
            df_joint = pd.DataFrame({"joint": joint, "isFraud": y})
            grouped = df_joint.groupby("joint")["isFraud"].agg(["count", "sum"])
            
            for j_cat, row in grouped.iterrows():
                cnt = int(row["count"])
                fr_cnt = int(row["sum"])
                leg_cnt = cnt - fr_cnt
                
                # Check support
                if cnt >= 10:
                    fr_rate = fr_cnt / cnt
                    
                    # Odds Ratio: (A/B) / (C/D)
                    # A = fraud in combo, B = legit in combo
                    # C = fraud outside combo, D = legit outside combo
                    a = fr_cnt + 0.5
                    b = leg_cnt + 0.5
                    c = (total_fraud - fr_cnt) + 0.5
                    d = (total_legit - leg_cnt) + 0.5
                    odds_ratio = (a / b) / (c / d)
                    
                    lift = fr_rate / (total_fraud / len(y)) if total_fraud > 0 else 1.0
                    
                    records.append({
                        "feature_1": c1,
                        "feature_2": c2,
                        "category_combination": str(j_cat),
                        "fraud_count": fr_cnt,
                        "legitimate_count": leg_cnt,
                        "fraud_rate": round(fr_rate, 4),
                        "lift": round(lift, 4),
                        "odds_ratio": round(odds_ratio, 4),
                    })
                    
        df = pd.DataFrame(records).sort_values(by="lift", ascending=False).head(40)
        if df.empty:
            df = pd.DataFrame(columns=["feature_1", "feature_2", "category_combination", "fraud_count", "legitimate_count", "fraud_rate", "lift", "odds_ratio"])
        df.to_csv(report_dir / "fraud_interactions.csv", index=False)
        return df

    def analyze_interaction_strength(self, report_dir: Path) -> pd.DataFrame:
        """Computes joint mutual information with the target class."""
        logger.info("Computing interaction strength...")
        records = []
        cols = self.features
        
        # Pre-fill missing values for MI computation on small sample
        df_clean = self.df_sample_small[cols].copy()
        for c in df_clean.columns:
            mean_val = df_clean[c].mean()
            df_clean[c] = df_clean[c].fillna(mean_val if pd.notna(mean_val) else 0.0)
            
        y = self.df_sample_small[self.target_col].fillna(0).values
        
        # Calculate raw MI scores
        mi_scores = mutual_info_classif(df_clean.values, y, random_state=self.random_state)
        mi_map = dict(zip(cols, mi_scores))
        
        # Compute joint MI for each pair
        # We can construct joint variables: Cartesian product or simple sum
        # To represent nonlinear combinations, we can concatenate normalized variables
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1 = cols[i]
                c2 = cols[j]
                
                # Make simple joint vector: x1 + x2 (sum) and x1 * x2 (product) concatenated
                x1 = df_clean[c1].values
                x2 = df_clean[c2].values
                joint_vec = np.column_stack([x1, x2, x1 * x2, x1 + x2])
                
                try:
                    mi_joint = float(mutual_info_classif(joint_vec, y, random_state=self.random_state).sum())
                except Exception:
                    mi_joint = 0.0
                    
                # Interaction Gain: I(X_1, X_2; Y) - I(X_1; Y) - I(X_2; Y)
                gain = mi_joint - mi_map[c1] - mi_map[c2]
                
                records.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "mutual_information_1": round(float(mi_map[c1]), 6),
                    "mutual_information_2": round(float(mi_map[c2]), 6),
                    "joint_mutual_information": round(mi_joint, 6),
                    "interaction_gain": round(gain, 6),
                    "interaction_strength": round(max(0.0, gain), 6),
                })
                
        df = pd.DataFrame(records).sort_values(by="interaction_strength", ascending=False)
        df.to_csv(report_dir / "interaction_strength.csv", index=False)

        # Ensure plots directory exists
        (report_dir / "plots").mkdir(parents=True, exist_ok=True)

        # Plot interaction strength heatmap
        # Pivot to square matrix
        unique_nodes = list(cols)
        strength_mat = pd.DataFrame(0.0, index=unique_nodes, columns=unique_nodes)
        
        for _, row in df.iterrows():
            f1, f2, val = row["feature_1"], row["feature_2"], row["interaction_strength"]
            strength_mat.loc[f1, f2] = val
            strength_mat.loc[f2, f1] = val
            
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#06070b")
        ax.set_facecolor("#06070b")
        im = ax.imshow(strength_mat.values, cmap="coolwarm", vmin=0)
        ax.set_title("[STRENGTH] Interaction Information Gain Heatmap", color="#fff", fontname="Orbitron", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.tick_params(colors="#8e97a4", labelsize=6)
        ax.set_xticks(range(len(unique_nodes)))
        ax.set_xticklabels(unique_nodes, rotation=90, fontsize=6)
        ax.set_yticks(range(len(unique_nodes)))
        ax.set_yticklabels(unique_nodes, fontsize=6)
        ax.spines[:].set_color((1.0, 1.0, 1.0, 0.08))
        
        plt.tight_layout()
        fig.savefig(report_dir / "plots" / "interaction_heatmap.png", dpi=110, facecolor="#06070b")
        plt.close(fig)

        return df

    def analyze_higher_order_interactions(self, report_dir: Path) -> pd.DataFrame:
        """Evaluates interactions involving three or more variables."""
        logger.info("Computing higher order interactions...")
        records = []
        cols = self.features
        
        # Take triplets from top variables
        triplets = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                for k in range(j + 1, len(cols)):
                    triplets.append((cols[i], cols[j], cols[k]))
                    
        # Limit to top 15 triplets
        triplets = triplets[:15]
        y = self.df_sample_large[self.target_col]
        global_fr = float(y.mean())
        if global_fr == 0:
            global_fr = 0.035

        for c1, c2, c3 in triplets:
            b1 = self._bin_column(self.df_sample_large[c1])
            b2 = self._bin_column(self.df_sample_large[c2])
            b3 = self._bin_column(self.df_sample_large[c3])
            
            joint = b1 + "_" + b2 + "_" + b3
            df_joint = pd.DataFrame({"joint": joint, "isFraud": y})
            grouped = df_joint.groupby("joint")["isFraud"].agg(["count", "mean"])
            
            # Select hotspot with support >= 10
            filtered = grouped[grouped["count"] >= 10]
            if not filtered.empty:
                hp = filtered.sort_values(by="mean", ascending=False).iloc[0]
                fraud_rate = float(hp["mean"])
                cnt = int(hp["count"])
                lift = fraud_rate / global_fr
            else:
                fraud_rate = 0.0
                cnt = 0
                lift = 0.0
                
            records.append({
                "feature_1": c1,
                "feature_2": c2,
                "feature_3": c3,
                "triplet_hotspot": hp.name if cnt > 0 else "N/A",
                "combination_count": cnt,
                "hotspot_fraud_rate": round(fraud_rate, 4),
                "lift_score": round(lift, 4),
            })
            
        df = pd.DataFrame(records).sort_values(by="lift_score", ascending=False)
        df.to_csv(report_dir / "higher_order_interactions.csv", index=False)
        return df

    def analyze_interaction_clustering(self, report_dir: Path, strength_df: pd.DataFrame) -> pd.DataFrame:
        """Clusters feature interactions according to similarity."""
        logger.info("Computing interaction clusters linkage...")
        cols = self.features
        
        # Build symmetric linkage matrix from interaction strength
        unique_nodes = list(cols)
        dist_mat = pd.DataFrame(1.0, index=unique_nodes, columns=unique_nodes)
        
        for _, row in strength_df.iterrows():
            f1, f2, val = row["feature_1"], row["feature_2"], row["interaction_strength"]
            # Convert strength to distance: higher strength = lower distance
            d = 1.0 / (1.0 + float(val))
            dist_mat.loc[f1, f2] = d
            dist_mat.loc[f2, f1] = d
            
        np.fill_diagonal(dist_mat.values, 0)
        
        from scipy.spatial.distance import squareform
        condensed_dist = squareform(dist_mat.values, checks=False)
        Z = linkage(condensed_dist, method="average")
        
        from scipy.cluster.hierarchy import fcluster
        labels = fcluster(Z, 0.40, criterion="distance")
        
        records = []
        for i, col in enumerate(unique_nodes):
            records.append({
                "feature_name": col,
                "interaction_cluster_id": int(labels[i]),
            })
            
        df_clusters = pd.DataFrame(records)
        df_clusters.to_csv(report_dir / "interaction_clusters.csv", index=False)

        # Ensure plots directory exists
        (report_dir / "plots").mkdir(parents=True, exist_ok=True)

        # Plot dendrogram
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#06070b")
        ax.set_facecolor("#06070b")
        dendrogram(
            Z,
            labels=unique_nodes,
            ax=ax,
            orientation="top",
            leaf_rotation=90,
            leaf_font_size=7,
            color_threshold=0.40,
            above_threshold_color="#8e97a4",
        )
        ax.set_title("[LINKAGE] Feature Interaction Linkage Dendrogram", color="#ffffff", fontname="Orbitron", fontsize=9)
        ax.tick_params(colors="#8e97a4")
        ax.spines[:].set_color((1.0, 1.0, 1.0, 0.08))
        
        plt.tight_layout()
        fig.savefig(report_dir / "plots" / "interaction_dendrogram.png", dpi=110, facecolor="#06070b")
        plt.close(fig)

        return df_clusters

    def recommend_interactions(self, report_dir: Path, strength_df: pd.DataFrame) -> pd.DataFrame:
        """Generates interaction feature recommendations for downstream pipelines."""
        logger.info("Generating interaction engineering recommendations...")
        records = []
        
        # Recommend top 10 strong interactions
        top_interactions = strength_df.head(10)
        for _, row in top_interactions.iterrows():
            f1 = row["feature_1"]
            f2 = row["feature_2"]
            st = float(row["interaction_strength"])
            
            # Rule engine based recommendation type
            # If both are numeric, suggest Product & Ratio
            # If one is numeric and another is binary/low card, suggest Group Mean
            # For simplicity, output multiple recipes
            records.append({
                "feature_1": f1,
                "feature_2": f2,
                "interaction_strength": st,
                "recommended_operation": f"Product: {f1} * {f2}",
                "rational": f"Joint mutual info exceeds individual sum with gain {st:.4f}",
            })
            records.append({
                "feature_1": f1,
                "feature_2": f2,
                "interaction_strength": st,
                "recommended_operation": f"Ratio: {f1} / ({f2} + 1e-5)",
                "rational": f"Evaluate relative ratios of high variance attributes",
            })
            
        df = pd.DataFrame(records)
        df.to_csv(report_dir / "interaction_feature_recommendations.csv", index=False)
        return df

    def analyze_interaction_stability(self, report_dir: Path) -> tuple[pd.DataFrame, dict]:
        """Compares joint distributions in train vs test to calculate joint drift index."""
        logger.info("Computing interaction persistence in train vs test...")
        records = []
        cols = self.features
        
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j]))
        
        pairs = pairs[:15]
        drift_report = {}

        for c1, c2 in pairs:
            # Skip if either is not present in test
            if c1 not in self.df_test.columns or c2 not in self.df_test.columns:
                continue
                
            # Train bins
            tr_b1 = self._bin_column(self.df_train[c1])
            tr_b2 = self._bin_column(self.df_train[c2])
            tr_joint = (tr_b1 + "_" + tr_b2).value_counts(normalize=True).to_dict()
            
            # Test bins
            te_b1 = self._bin_column(self.df_test[c1])
            te_b2 = self._bin_column(self.df_test[c2])
            te_joint = (te_b1 + "_" + te_b2).value_counts(normalize=True).to_dict()
            
            # Calculate PSI of joint category
            all_keys = set(tr_joint.keys()) | set(te_joint.keys())
            psi_score = 0.0
            for k in all_keys:
                act = tr_joint.get(k, 0.0)
                exp = te_joint.get(k, 0.0)
                # handle zero allocations
                act_safe = max(act, 0.0001)
                exp_safe = max(exp, 0.0001)
                psi_score += (act_safe - exp_safe) * np.log(act_safe / exp_safe)
                
            records.append({
                "feature_1": c1,
                "feature_2": c2,
                "joint_psi": round(float(psi_score), 6),
                "stability_class": "STABLE" if psi_score < 0.10 else "MODERATE DRIFT" if psi_score < 0.25 else "UNSTABLE",
            })
            
        df = pd.DataFrame(records).sort_values(by="joint_psi")
        if df.empty:
            df = pd.DataFrame(columns=["feature_1", "feature_2", "joint_psi", "stability_class"])
        df.to_csv(report_dir / "interaction_stability.csv", index=False)

        drift_report = {
            "average_joint_psi": float(df["joint_psi"].mean()) if not df.empty else 0.0,
            "stable_interactions_pct": float((df["stability_class"] == "STABLE").mean() * 100) if not df.empty else 100.0,
        }
        with open(report_dir / "interaction_drift_report.json", "w") as f:
            json.dump(drift_report, f, indent=4)

        return df, drift_report

    def screen_production_interactions(self, report_dir: Path, stability_df: pd.DataFrame) -> pd.DataFrame:
        """Screen engineered interactions checking overfitting, sparsity or cost indicators."""
        logger.info("Computing interaction screening metrics...")
        records = []
        
        # Cross refer with stability
        for _, row in stability_df.iterrows():
            f1 = row["feature_1"]
            f2 = row["feature_2"]
            psi = float(row["joint_psi"])
            
            # Compute sparsity in train
            b1 = self._bin_column(self.df_train[f1])
            b2 = self._bin_column(self.df_train[f2])
            joint_counts = (b1 + "_" + b2).value_counts()
            
            # Sparsity = percentage of potential crossings containing less than 1% of rows
            sparse_pct = float((joint_counts < (0.01 * len(self.df_train))).mean() * 100)
            
            # Recommendation
            if psi >= 0.25:
                rec = "REMOVE"
                reason = f"High temporal drift (PSI={psi:.4f})"
            elif sparse_pct >= 60.0:
                rec = "MERGE"
                reason = f"High category sparsity ({sparse_pct:.1f}% crossings have <1% support)"
            else:
                rec = "RETAIN"
                reason = "Stable joint representation and representative categories"
                
            records.append({
                "feature_1": f1,
                "feature_2": f2,
                "sparsity_percentage": round(sparse_pct, 4),
                "joint_psi_score": round(psi, 4),
                "production_action": rec,
                "rational": reason,
            })
            
        df = pd.DataFrame(records).sort_values(by="sparsity_percentage")
        df.to_csv(report_dir / "production_interaction_screening.csv", index=False)
        return df

    def compile_html_dashboard(
        self,
        report_dir: Path,
        df_inv: pd.DataFrame,
        df_sep: pd.DataFrame,
        df_cross: pd.DataFrame,
        df_strength: pd.DataFrame,
        df_higher: pd.DataFrame,
        df_stability: pd.DataFrame,
        df_screen: pd.DataFrame,
    ) -> None:
        """Compiles HTML summary dashboard showing interaction details."""
        logger.info("Compiling Interaction HTML report...")

        def _to_html(df: pd.DataFrame) -> str:
            if df.empty:
                return "<div class='no-data'>NO INTERACTION DATA COMPUTED</div>"
            return df.to_html(
                classes="hud-table",
                index=False,
                border=0,
                justify="left",
            )
            
        summary = {
            "total_candidates": len(self.features),
            "max_interaction_lift": float(df_cross["lift_score"].max()) if not df_cross.empty else 1.0,
            "top_pair_separation": float(df_sep["class_separation_distance"].max()) if not df_sep.empty else 0.0,
            "retained_for_prod": int((df_screen["production_action"] == "RETAIN").sum()) if not df_screen.empty else 0,
        }

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>IEEE-CIS FEATURE INTERACTION DIAGNOSTICS</title>
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
                'plots/interaction_heatmap.png',
                'plots/pairplot_scatter.png',
                'plots/interaction_dendrogram.png'
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
            <h1>IEEE-CIS FEATURE INTERACTION DIAGNOSTICS</h1>
            <p style="font-size: 0.65rem; color: var(--text-color); margin-top: 0.25rem; letter-spacing: 1px;">STAGE 3.13: SYSTEM INTERACTION PROPERTIES</p>
        </div>
        <div class="status-pill">INTERACTION ANALYZER OK</div>
    </header>

    <div class="hud-grid">
        <div class="hud-panel">
            <p class="metric-label">Candidate Features</p>
            <p class="metric-value">{summary['total_candidates']}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Max Interaction Lift</p>
            <p class="metric-value">{summary['max_interaction_lift']:.2f}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Top Pair Class Separation</p>
            <p class="metric-value">{summary['top_pair_separation']:.2f}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Retained for Production</p>
            <p class="metric-value">{summary['retained_for_prod']}</p>
        </div>
    </div>

    <div class="dashboard-body">
        <div class="hud-panel visualizer-card">
            <h2>DIAGNOSTIC VISUALIZATIONS</h2>
            <div class="carousel-tabs">
                <button class="carousel-tab active" onclick="switchTab(0)">Gain Heatmap</button>
                <button class="carousel-tab" onclick="switchTab(1)">Class Separation</button>
                <button class="carousel-tab" onclick="switchTab(2)">Interaction Linkage</button>
            </div>
            <div class="carousel-content">
                <img id="carousel-img" class="carousel-img" src="plots/interaction_heatmap.png" alt="Interaction Heatmap">
            </div>
        </div>

        <div class="hud-panel">
            <h2>3-WAY HIGHER ORDER INTERACTIONS</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_higher.head(30))}
            </div>
        </div>
    </div>

    <div class="secondary-panel-grid">
        <div class="hud-panel">
            <h2>PRODUCTION SCREENING METRICS</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_screen.head(30))}
            </div>
        </div>
        <div class="hud-panel">
            <h2>CROSS-FEATURE JOINT FRAUD RATES</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_cross.head(30))}
            </div>
        </div>
    </div>

</body>
</html>
"""
        with open(report_dir / "interaction_analysis_report.html", "w") as f:
            f.write(html_template)
        logger.info("Compiled Interaction HTML report saved.")

    def analyze_all(self, report_dir: Path) -> None:
        """Runs full feature interaction pipeline."""
        report_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = report_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("--- Stage 3.13: Feature Interaction Analysis (Starting) ---")
        
        df_inv = self.analyze_interaction_inventory(report_dir)
        df_sep = self.analyze_pairplots(report_dir)
        df_cross = self.analyze_cross_features(report_dir)
        self.analyze_fraud_interactions(report_dir)
        df_strength = self.analyze_interaction_strength(report_dir)
        df_higher = self.analyze_higher_order_interactions(report_dir)
        self.analyze_interaction_clustering(report_dir, df_strength)
        self.recommend_interactions(report_dir, df_strength)
        df_stability, _ = self.analyze_interaction_stability(report_dir)
        df_screen = self.screen_production_interactions(report_dir, df_stability)
        
        self.compile_html_dashboard(
            report_dir,
            df_inv,
            df_sep,
            df_cross,
            df_strength,
            df_higher,
            df_stability,
            df_screen,
        )
        logger.info("Feature Interaction Analysis complete.")
