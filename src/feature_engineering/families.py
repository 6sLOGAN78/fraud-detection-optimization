"""Feature Families extraction, tracking, and integration engine."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionFamilyBuilder:
    """Extracts features relating directly to transaction details."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing Transaction Feature Family...")
        out = pd.DataFrame(index=df.index)
        out["TransactionID"] = df["TransactionID"]
        
        # Transaction Frequency
        if "ProductCD" in df.columns:
            # We convert ProductCD to str to avoid Categorical type errors during maps
            counts = df["ProductCD"].astype(str).value_counts()
            out["ProductCD_count"] = df["ProductCD"].astype(str).map(counts).fillna(0.0)

        # Rolling statistics (velocity placeholder)
        if "TransactionDT" in df.columns:
            out["transaction_density_index"] = df["TransactionDT"].diff().fillna(60.0)

        return out


class IdentityFamilyBuilder:
    """Extracts features relating to customer and device identity profile."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing Identity Feature Family...")
        out = pd.DataFrame(index=df.index)
        out["TransactionID"] = df["TransactionID"]
        
        # Score completeness of identity columns
        id_cols = [col for col in df.columns if col.startswith("id_") or col.startswith("id-")]
        if id_cols:
            out["identity_completeness_score"] = df[id_cols].notnull().mean(axis=1)
        else:
            out["identity_completeness_score"] = 0.0

        if "DeviceType" in df.columns:
            # Check for mobile vs desktop
            out["is_mobile_device"] = (df["DeviceType"].astype(str).str.lower() == "mobile").astype(int)
        else:
            out["is_mobile_device"] = 0

        return out


class TimeFamilyBuilder:
    """Extracts temporal/chronological characteristics from TransactionDT."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing Time Feature Family...")
        out = pd.DataFrame(index=df.index)
        out["TransactionID"] = df["TransactionID"]

        # TransactionDT is in seconds
        if "TransactionDT" in df.columns:
            dt = df["TransactionDT"]
            out["transaction_hour"] = (dt % 86400) / 3600.0
            out["transaction_day"] = (dt / 86400).astype(int) % 7
            out["is_business_hour"] = ((out["transaction_hour"] >= 9.0) & (out["transaction_hour"] <= 17.0)).astype(int)
        else:
            out["transaction_hour"] = 0.0
            out["transaction_day"] = 0
            out["is_business_hour"] = 0

        return out


class AmountFamilyBuilder:
    """Extracts monetary characteristics from TransactionAmt."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing Amount Feature Family...")
        out = pd.DataFrame(index=df.index)
        out["TransactionID"] = df["TransactionID"]

        if "TransactionAmt" in df.columns:
            amt = df["TransactionAmt"].fillna(0.0)
            out["log_TransactionAmt"] = np.log1p(amt)
            out["is_high_value_amt"] = (amt > 150.0).astype(int)
            out["fractional_amt"] = amt % 1.0
        else:
            out["log_TransactionAmt"] = 0.0
            out["is_high_value_amt"] = 0
            out["fractional_amt"] = 0.0

        return out


class EmailFamilyBuilder:
    """Extracts domain properties from P_emaildomain and R_emaildomain."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing Email Feature Family...")
        out = pd.DataFrame(index=df.index)
        out["TransactionID"] = df["TransactionID"]

        p_email = df["P_emaildomain"].astype(str) if "P_emaildomain" in df.columns else pd.Series("nan", index=df.index)
        r_email = df["R_emaildomain"].astype(str) if "R_emaildomain" in df.columns else pd.Series("nan", index=df.index)

        # Domain Match indicator
        out["is_domain_match"] = (p_email == r_email).astype(int)

        # Corporate / Free domains classifications
        free_domains = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "icloud.com"}
        out["p_is_free_domain"] = p_email.isin(free_domains).astype(int)
        out["r_is_free_domain"] = r_email.isin(free_domains).astype(int)

        return out


class DeviceFamilyBuilder:
    """Parses hardware manufacturers and popularity from DeviceInfo."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing Device Feature Family...")
        out = pd.DataFrame(index=df.index)
        out["TransactionID"] = df["TransactionID"]

        if "DeviceInfo" in df.columns:
            dev = df["DeviceInfo"].astype(str).str.lower()
            out["device_is_windows"] = dev.str.contains("windows").astype(int)
            out["device_is_ios_apple"] = (dev.str.contains("ios") | dev.str.contains("mac") | dev.str.contains("iphone")).astype(int)
            out["device_is_android"] = (dev.str.contains("android") | dev.str.contains("samsung") | dev.str.contains("huawei")).astype(int)
        else:
            out["device_is_windows"] = 0
            out["device_is_ios_apple"] = 0
            out["device_is_android"] = 0

        return out


class AddressFamilyBuilder:
    """Extracts geographical characteristics from addr1 and addr2."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing Address Feature Family...")
        out = pd.DataFrame(index=df.index)
        out["TransactionID"] = df["TransactionID"]

        addr1 = df["addr1"].astype(str) if "addr1" in df.columns else pd.Series("nan", index=df.index)
        addr2 = df["addr2"].astype(str) if "addr2" in df.columns else pd.Series("nan", index=df.index)

        # Missing indicator checks
        out["is_addr1_missing"] = (addr1 == "nan").astype(int)
        out["is_addr2_missing"] = (addr2 == "nan").astype(int)

        # Frequency mapping mapping
        if "addr1" in df.columns:
            addr1_counts = df["addr1"].value_counts().to_dict()
            out["addr1_frequency"] = df["addr1"].map(addr1_counts).fillna(0.0)
        else:
            out["addr1_frequency"] = 0.0

        return out


class CardFamilyBuilder:
    """Extracts issuer characteristics relative to card variables."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing Card Feature Family...")
        out = pd.DataFrame(index=df.index)
        out["TransactionID"] = df["TransactionID"]

        if "card1" in df.columns:
            card1_counts = df["card1"].value_counts().to_dict()
            out["card1_frequency"] = df["card1"].map(card1_counts).fillna(0.0)
        else:
            out["card1_frequency"] = 0.0

        if "card4" in df.columns:
            # Card type popular mapping (e.g. visa, mastercard)
            c4 = df["card4"].astype(str).str.lower()
            out["card_is_visa"] = c4.str.contains("visa").astype(int)
            out["card_is_mastercard"] = c4.str.contains("mastercard").astype(int)
        else:
            out["card_is_visa"] = 0
            out["card_is_mastercard"] = 0

        return out


class FeatureFamilyTracker:
    """Tracks metadata and builds feature lists catalogs."""
    def __init__(self) -> None:
        self.catalog: list[dict[str, Any]] = []

    def record_family(
        self,
        family_name: str,
        features_added: list[str],
        source_dependencies: list[str],
        version: str = "v1.0",
    ) -> None:
        self.catalog.append({
            "family_name": family_name,
            "feature_count": len(features_added),
            "features": features_added,
            "dependencies": source_dependencies,
            "version": version,
            "last_updated": pd.Timestamp.now().isoformat(),
            "owner": "ML-Engineering-Team",
        })

    def save_metadata(self, dest_dir: Path) -> tuple[Path, Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        meta_json = dest_dir / "feature_family_metadata.json"
        with open(meta_json, "w") as f:
            json.dump(self.catalog, f, indent=4)

        # Convert to CSV catalog format
        flat_list = []
        for family in self.catalog:
            for feat in family["features"]:
                flat_list.append({
                    "family_name": family["family_name"],
                    "feature_name": feat,
                    "version": family["version"],
                    "dependencies": ",".join(family["dependencies"]),
                })
        
        meta_csv = dest_dir / "feature_family_catalog.csv"
        df_cat = pd.DataFrame(flat_list)
        df_cat.to_csv(meta_csv, index=False)

        logger.info("Saved family metadata to %s and %s", meta_json, meta_csv)
        return meta_json, meta_csv


class FeatureFamilyIntegrator:
    """Validates, merges, and integrates all feature families into a single feature store."""
    def __init__(self, store_dir: Path) -> None:
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def integrate(self, families: dict[str, pd.DataFrame], partition: str, version: str) -> Path:
        logger.info("Merging feature families...")
        
        # Verify all directories exist
        dest_dir = self.store_dir / version
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Save individual families to independent parquet files
        for name, df_family in families.items():
            path_fam = dest_dir / f"{partition}_{name}_feature_family.parquet"
            df_family.to_parquet(path_fam, index=False)
            logger.info("Saved family %s path: %s", name, path_fam)

        # Merge them sequentially based on TransactionID
        merged_df = None
        for name, df_family in families.items():
            if merged_df is None:
                merged_df = df_family.copy()
            else:
                # Merge on TransactionID. Avoid duplicating columns
                cols_to_use = [col for col in df_family.columns if col not in merged_df.columns or col == "TransactionID"]
                merged_df = pd.merge(merged_df, df_family[cols_to_use], on="TransactionID", how="left")

        # Save complete feature matrix
        out_path = dest_dir / f"{partition}_complete_feature_matrix.parquet"
        merged_df.to_parquet(out_path, index=False)
        logger.info("Unified feature store matrix written: %s", out_path)
        
        return out_path
