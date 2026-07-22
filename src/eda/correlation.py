"""Part 3.12: Correlation Analysis Engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import kendalltau, pearsonr, pointbiserialr, spearmanr
from sklearn.feature_selection import mutual_info_classif

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Computes linear/rank correlations, hierarchy clusters, networks, and pruning parameters."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_col: str = "isFraud",
        threshold: float = 0.90,
        random_state: int = 42,
    ) -> None:
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.target_col = target_col
        self.threshold = threshold
        self.random_state = random_state

        # Encode binary object/category columns to 0.0/1.0
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

        # Pre-process columns selection
        self.features = self._select_features()
        
        # Safe samples for long calculation runs
        # Use 10000 sample for calculations, 2000 for mutual information and Kendall's tau due to CPU latency
        rng = np.random.default_rng(self.random_state)
        sample_size_large = min(len(self.df_train), 15000)
        sample_size_small = min(len(self.df_train), 2500)
        
        indices_large = rng.choice(self.df_train.index, sample_size_large, replace=False)
        self.df_sample_large = self.df_train.loc[indices_large]

        indices_small = rng.choice(self.df_train.index, sample_size_small, replace=False)
        self.df_sample_small = self.df_train.loc[indices_small]

        logger.info(
            "CorrelationAnalyzer initialized. Features identified: %d. Data Sample: %d rows.",
            len(self.features),
            sample_size_large,
        )

    def _select_features(self) -> list[str]:
        """Automatically filters out text, target, identification or high-cardinality values."""
        df = self.df_train
        
        # Standard columns to ignore
        ignore = {"TransactionID", "TransactionDT", "isFraud", "DeviceInfo", "DeviceType"}
        candidate_cols = [c for c in df.columns if c not in ignore]
        
        selected = []
        for c in candidate_cols:
            col_data = df[c]
            # Must be numerical or lowcardinality encoded
            if pd.api.types.is_numeric_dtype(col_data):
                # Avoid constant/empty columns
                if col_data.nunique(dropna=True) > 1:
                    selected.append(c)
            elif pd.api.types.is_object_dtype(col_data) or isinstance(col_data.dtype, pd.CategoricalDtype):
                # If binary
                if col_data.nunique(dropna=True) == 2:
                    selected.append(c)
        
        return selected

    def analyze_correlation_inventory(self, report_dir: Path) -> pd.DataFrame:
        """Create feature records with missingness and types."""
        logger.info("Generating correlation feature inventory...")
        records = []
        for col in self.features:
            col_data = self.df_train[col]
            missing_pct = float(col_data.isna().mean() * 100)
            
            # Categorize type
            if pd.api.types.is_integer_dtype(col_data.dtype):
                t = "Integer"
            elif pd.api.types.is_float_dtype(col_data.dtype):
                t = "Float"
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
        df.to_csv(report_dir / "correlation_feature_inventory.csv", index=False)

        metadata = {
            "total_correlation_features": len(self.features),
            "feature_type_counts": df["feature_type"].value_counts().to_dict(),
            "avg_missing_percentage": round(float(df["missing_percentage"].mean()), 4),
        }
        with open(report_dir / "correlation_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        return df

    def compute_correlations(self, report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute Pearson, Spearman and Kendall correlations on features subset."""
        logger.info("Computing Pearson & Spearman correlations...")
        
        # To avoid massive dimensions, we limit maximum columns used for pairwise correlation mapping to top 40 sorted by variance
        # This keeps computations very fast while representing the highly-moving variables.
        variances = self.df_sample_large[self.features].var().fillna(0)
        top_cols = list(variances.sort_values(ascending=False).head(40).index)
        
        # 1. Pearson
        pearson_mat = self.df_sample_large[top_cols].corr(method="pearson").fillna(0)
        # 2. Spearman
        spearman_mat = self.df_sample_large[top_cols].corr(method="spearman").fillna(0)
        
        logger.info("Computing Kendall tau correlation on top 10 features...")
        # Kendall is extremely heavy, limit to top 10 high-variance features & smaller sample
        top_10_cols = top_cols[:10]
        kendall_mat = self.df_sample_small[top_10_cols].corr(method="kendall").fillna(0)

        # Convert to flat records for target list reporting
        def _to_flat(m: pd.DataFrame) -> pd.DataFrame:
            cols = m.columns
            rows = []
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    val = m.loc[cols[i], cols[j]]
                    rows.append({
                        "feature_1": cols[i],
                        "feature_2": cols[j],
                        "correlation": round(float(val), 6),
                        "abs_correlation": round(abs(float(val)), 6),
                    })
            return pd.DataFrame(rows).sort_values(by="abs_correlation", ascending=False)

        pearson_flat = _to_flat(pearson_mat)
        spearman_flat = _to_flat(spearman_mat)
        kendall_flat = _to_flat(kendall_mat)

        pearson_flat.to_csv(report_dir / "pearson_correlation.csv", index=False)
        spearman_flat.to_csv(report_dir / "spearman_correlation.csv", index=False)
        kendall_flat.to_csv(report_dir / "kendall_correlation.csv", index=False)

        # Save Pearson matrix as the common matrix
        pearson_mat.to_csv(report_dir / "correlation_matrix.csv")

        return pearson_mat, spearman_mat, kendall_mat

    def analyze_cluster_map(self, report_dir: Path, pearson_mat: pd.DataFrame) -> pd.DataFrame:
        """Dendrogram and hierarchical clustering analysis."""
        logger.info("Executing Hierarchical Clustering...")
        
        # Linkage from correlation matrix
        corr = pearson_mat.values
        # Handle NaN values safely
        corr = np.nan_to_num(corr)
        # Convert to distance matrix
        dist = 1 - np.abs(corr)
        # Symmetrize
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        
        # Scipy Linkage
        from scipy.spatial.distance import squareform
        # link expects squareform or condensed distance vector
        condensed_dist = squareform(dist, checks=False)
        Z = linkage(condensed_dist, method="complete")
        
        # Find feature clusters from cutting threshold
        # Default cut distance is 0.15 (correlation > 0.85)
        from scipy.cluster.hierarchy import fcluster
        c_labels = fcluster(Z, 0.15, criterion="distance")
        
        records = []
        for i, col in enumerate(pearson_mat.columns):
            records.append({
                "feature_name": col,
                "cluster_id": int(c_labels[i]),
            })
            
        df_clusters = pd.DataFrame(records)
        df_clusters.to_csv(report_dir / "feature_clusters.csv", index=False)

        # Save Dendrogram Plot
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#06070b")
        ax.set_facecolor("#06070b")
        dendrogram(
            Z,
            labels=list(pearson_mat.columns),
            ax=ax,
            orientation="top",
            leaf_rotation=90,
            leaf_font_size=7,
            color_threshold=0.15,
            above_threshold_color="#8e97a4",
        )
        ax.set_title("[HIERARCHICAL] Feature Dendrogram", color="#ffffff", fontname="Orbitron", fontsize=10)
        ax.tick_params(colors="#8e97a4")
        ax.spines[:].set_color((1.0, 1.0, 1.0, 0.08))
        plt.tight_layout()
        fig.savefig(report_dir / "plots" / "dendrogram.png", dpi=110, facecolor="#06070b")
        plt.close(fig)

        return df_clusters

    def analyze_network(self, report_dir: Path, pearson_mat: pd.DataFrame) -> pd.DataFrame:
        """Graph representation of highly-correlated variables."""
        logger.info("Computing network graph hubs...")
        cols = pearson_mat.columns
        corr_val = pearson_mat.values

        edges = []
        node_degrees: dict[str, int] = {c: 0 for c in cols}
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_val[i, j]
                if abs(val) >= 0.85:
                    edges.append({
                        "source": cols[i],
                        "target": cols[j],
                        "weight": round(float(val), 4),
                    })
                    node_degrees[cols[i]] += 1
                    node_degrees[cols[j]] += 1

        df_net = pd.DataFrame(edges)
        if df_net.empty:
            df_net = pd.DataFrame(columns=["source", "target", "weight"])
        df_net.to_csv(report_dir / "correlation_network.csv", index=False)

        # Assign node properties
        node_records = []
        for col in cols:
            node_records.append({
                "feature_name": col,
                "degree": node_degrees[col],
                "is_hub": "YES" if node_degrees[col] >= 3 else "NO",
            })
        df_nodes = pd.DataFrame(node_records).sort_values(by="degree", ascending=False)
        df_nodes.to_csv(report_dir / "network_nodes.csv", index=False)

        return df_nodes

    def analyze_target_correlation(self, report_dir: Path) -> pd.DataFrame:
        """Relation to target isFraud."""
        logger.info("Analyzing Target correlations & mutual information...")
        if self.target_col not in self.df_train.columns:
            return pd.DataFrame()
            
        y = self.df_sample_large[self.target_col]
        records = []
        
        # We also compute Mutual Information using the small sample for efficiency
        X_mi = self.df_sample_small[self.features].copy()
        for col in X_mi.columns:
            mean = X_mi[col].mean()
            X_mi[col] = X_mi[col].fillna(mean if pd.notna(mean) else 0)
            
        mi_scores = mutual_info_classif(
            X_mi.values,
            self.df_sample_small[self.target_col].values,
            random_state=self.random_state,
        )
        mi_map = dict(zip(self.features, mi_scores))
        
        for col in self.features:
            col_data = self.df_sample_large[col]
            # Clean missing values for correlation
            valid_mask = col_data.notna() & y.notna()
            clean_col = col_data[valid_mask]
            clean_y = y[valid_mask]
            
            p_val, pb_val, s_val = 0.0, 0.0, 0.0
            
            if len(clean_col) > 10 and clean_col.nunique() > 1:
                try:
                    p_val = pearsonr(clean_col, clean_y)[0]
                except Exception:
                    pass
                try:
                    s_val = spearmanr(clean_col, clean_y)[0]
                except Exception:
                    pass
                try:
                    pb_val = pointbiserialr(clean_col, clean_y)[0]
                except Exception:
                    pass

            records.append({
                "feature_name": col,
                "pearson_correlation": round(float(p_val) if np.isfinite(p_val) else 0.0, 6),
                "spearman_correlation": round(float(s_val) if np.isfinite(s_val) else 0.0, 6),
                "point_biserial_correlation": round(float(pb_val) if np.isfinite(pb_val) else 0.0, 6),
                "mutual_information": round(float(mi_map.get(col, 0.0)), 6),
            })
            
        df = pd.DataFrame(records).sort_values(by="mutual_information", ascending=False)
        df.to_csv(report_dir / "target_correlation.csv", index=False)
        return df

    def correlation_pruning(
        self,
        report_dir: Path,
        pearson_mat: pd.DataFrame,
        df_target_corr: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Automatically identify redundant variables for deletion."""
        logger.info("Running correlation pruning...")
        
        # Build lookup tools
        mi_map = df_target_corr.set_index("feature_name")["mutual_information"].to_dict()
        
        missing_map = {}
        var_map = {}
        for c in pearson_mat.columns:
            missing_map[c] = float(self.df_train[c].isna().mean())
            var_map[c] = float(self.df_train[c].var())

        cols = pearson_mat.columns
        corr_val = pearson_mat.values

        pruning_records = []
        removed: set[str] = set()
        retained: set[str] = set(cols)

        # Loop to identify redundant pairs
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1 = cols[i]
                c2 = cols[j]
                val = corr_val[i, j]
                
                if abs(val) >= self.threshold:
                    # Decide which feature to retain
                    # 1. Prefer higher Mutual Information
                    mi1 = mi_map.get(c1, 0.0)
                    mi2 = mi_map.get(c2, 0.0)
                    
                    if abs(mi1 - mi2) > 1e-4:
                        keep, drop = (c1, c2) if mi1 > mi2 else (c2, c1)
                        reason = f"MI: {keep}({mi1:.4f}) > {drop}({mi2:.4f})"
                    else:
                        # 2. Prefer lower missingness
                        miss1 = missing_map.get(c1, 1.0)
                        miss2 = missing_map.get(c2, 1.0)
                        if abs(miss1 - miss2) > 1e-4:
                            keep, drop = (c1, c2) if miss1 < miss2 else (c2, c1)
                            reason = f"Missing: {keep}({miss1:.2%}) < {drop}({miss2:.2%})"
                        else:
                            # 3. Prefer higher variance
                            v1 = var_map.get(c1, 0.0)
                            v2 = var_map.get(c2, 0.0)
                            keep, drop = (c1, c2) if v1 >= v2 else (c2, c1)
                            reason = f"Variance: {keep}({v1:.2e}) >= {drop}({v2:.2e})"
                    
                    removed.add(drop)
                    if drop in retained:
                        retained.remove(drop)

                    pruning_records.append({
                        "feature_1": c1,
                        "feature_2": c2,
                        "correlation": round(float(val), 4),
                        "decision": f"Retain {keep}, Remove {drop}",
                        "reason": reason,
                    })

        df_pruni = pd.DataFrame(pruning_records)
        df_pruni.to_csv(report_dir / "correlation_pruning.csv", index=False)

        df_keep = pd.DataFrame({"feature_name": list(retained)})
        df_keep.to_csv(report_dir / "retained_features.csv", index=False)

        df_drop = pd.DataFrame({"feature_name": list(removed)})
        df_drop.to_csv(report_dir / "removed_features.csv", index=False)

        return df_pruni, df_keep, df_drop

    def generate_plots(
        self,
        plots_dir: Path,
        pearson_mat: pd.DataFrame,
        spearman_mat: pd.DataFrame,
        df_target: pd.DataFrame,
    ) -> None:
        """Produce heatmap visual plots."""
        logger.info("Generating correlation plots...")
        plt.style.use("dark_background")

        # 1. Pearson Correlation Heatmap
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#06070b")
        ax.set_facecolor("#06070b")
        im = ax.imshow(pearson_mat.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("[PEARSON] Pairwise Linear Heatmap", color="#fff", fontname="Orbitron", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.tick_params(colors="#8e97a4", labelsize=6)
        ax.spines[:].set_color((1.0, 1.0, 1.0, 0.08))
        plt.tight_layout()
        fig.savefig(plots_dir / "pearson_heatmap.png", dpi=110, facecolor="#06070b")
        plt.close(fig)

        # 2. Spearman Rank Heatmap
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#06070b")
        ax.set_facecolor("#06070b")
        im = ax.imshow(spearman_mat.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("[SPEARMAN] Pairwise Rank Heatmap", color="#fff", fontname="Orbitron", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.tick_params(colors="#8e97a4", labelsize=6)
        ax.spines[:].set_color((1.0, 1.0, 1.0, 0.08))
        plt.tight_layout()
        fig.savefig(plots_dir / "spearman_heatmap.png", dpi=110, facecolor="#06070b")
        plt.close(fig)

        # 3. Target Mutual Information rankings (Top 15)
        if not df_target.empty:
            top_15 = df_target.head(15)
            fig, ax = plt.subplots(figsize=(8, 4), facecolor="#06070b")
            ax.set_facecolor("#06070b")
            ax.barh(
                top_15["feature_name"][::-1],
                top_15["mutual_information"][::-1],
                color="#8e97a4",
                edgecolor=(1.0, 1.0, 1.0, 0.08),
            )
            ax.set_title("[TARGET] Mutual Information (Top 15)", color="#fff", fontname="Orbitron", fontsize=10)
            ax.tick_params(colors="#8e97a4")
            ax.spines[:].set_color((1.0, 1.0, 1.0, 0.08))
            plt.tight_layout()
            fig.savefig(plots_dir / "target_mutual_info.png", dpi=110, facecolor="#06070b")
            plt.close(fig)

    def compile_html_dashboard(
        self,
        report_dir: Path,
        df_inv: pd.DataFrame,
        df_target: pd.DataFrame,
        df_pruni: pd.DataFrame,
        df_keep: pd.DataFrame,
        df_drop: pd.DataFrame,
        df_nodes: pd.DataFrame,
    ) -> None:
        """Glassmorphic dark minimal sci-fi correlation summary report."""
        logger.info("Compiling Correlation HTML Dashboard...")

        def _to_html(df: pd.DataFrame) -> str:
            if df.empty:
                return "<div class='no-data'>NO COMPATIBLE DATA FOUND</div>"
            return df.to_html(
                classes="hud-table",
                index=False,
                border=0,
                justify="left",
            )
            
        summary = {
            "total_features": len(self.features),
            "target_mutual_info_top": str(df_target.iloc[0]["feature_name"]) if not df_target.empty else "N/A",
            "redundant_features": len(df_drop),
            "network_density_hubs": int((df_nodes["is_hub"] == "YES").sum()),
        }

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>IEEE-CIS CORRELATION DIAGNOSTICS</title>
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

        .metric-val-red {{
            color: var(--alert-red);
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
                'plots/pearson_heatmap.png',
                'plots/spearman_heatmap.png',
                'plots/dendrogram.png',
                'plots/target_mutual_info.png'
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
            <h1>IEEE-CIS CORRELATION DIAGNOSTICS</h1>
            <p style="font-size: 0.65rem; color: var(--text-color); margin-top: 0.25rem; letter-spacing: 1px;">STAGE 3.12: SYSTEM CORRELATION PROPERTIES</p>
        </div>
        <div class="status-pill">CORRELATION ANALYZER OK</div>
    </header>

    <div class="hud-grid">
        <div class="hud-panel">
            <p class="metric-label">Analyzed Features</p>
            <p class="metric-value">{summary['total_features']}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Top Target Predictor (MI)</p>
            <p class="metric-value">{summary['target_mutual_info_top']}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Redundant Features Removed</p>
            <p class="metric-value metric-val-red">{summary['redundant_features']}</p>
        </div>
        <div class="hud-panel">
            <p class="metric-label">Correlation network Hubs</p>
            <p class="metric-value">{summary['network_density_hubs']}</p>
        </div>
    </div>

    <div class="dashboard-body">
        <div class="hud-panel visualizer-card">
            <h2>DIAGNOSTIC VISUALIZATIONS</h2>
            <div class="carousel-tabs">
                <button class="carousel-tab active" onclick="switchTab(0)">Pearson Linear</button>
                <button class="carousel-tab" onclick="switchTab(1)">Spearman Rank</button>
                <button class="carousel-tab" onclick="switchTab(2)">Dendrogram Map</button>
                <button class="carousel-tab" onclick="switchTab(3)">Target Mutual Info</button>
            </div>
            <div class="carousel-content">
                <img id="carousel-img" class="carousel-img" src="plots/pearson_heatmap.png" alt="Correlation Map">
            </div>
        </div>

        <div class="hud-panel">
            <h2>MUTUAL INFORMATION & TARGET CORRELATION</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_target.head(30))}
            </div>
        </div>
    </div>

    <div class="secondary-panel-grid">
        <div class="hud-panel">
            <h2>REDUNDANCY PRUNING DECISIONS (THRESHOLD: {self.threshold:.2f})</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_pruni.head(30))}
            </div>
        </div>
        <div class="hud-panel">
            <h2>NETWORK NODES DEGREES</h2>
            <div class="hud-table-wrapper">
                {_to_html(df_nodes.head(30))}
            </div>
        </div>
    </div>

</body>
</html>
"""
        with open(report_dir / "correlation_analysis_report.html", "w") as f:
            f.write(html_template)
        logger.info("Compiled Correlation HTML report saved.")

    def analyze_all(self, report_dir: Path) -> None:
        """Executes full diagnostic pipeline."""
        report_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = report_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("--- Stage 3.12: Correlation Analysis (Starting Process) ---")
        
        df_inv = self.analyze_correlation_inventory(report_dir)
        pearson_mat, spearman_mat, kendall_mat = self.compute_correlations(report_dir)
        df_clusters = self.analyze_cluster_map(report_dir, pearson_mat)
        df_nodes = self.analyze_network(report_dir, pearson_mat)
        df_target = self.analyze_target_correlation(report_dir)
        
        df_pruni, df_keep, df_drop = self.correlation_pruning(
            report_dir,
            pearson_mat,
            df_target,
        )
        
        self.generate_plots(
            plots_dir,
            pearson_mat,
            spearman_mat,
            df_target,
        )
        
        self.compile_html_dashboard(
            report_dir,
            df_inv,
            df_target,
            df_pruni,
            df_keep,
            df_drop,
            df_nodes,
        )
        
        logger.info("Correlation Analysis completed.")
