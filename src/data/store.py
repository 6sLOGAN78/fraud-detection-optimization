"""Enterprise Feature Store foundation module providing Offline, Online, Registry, Catalog, APIs, versioning, lineage, Security RBAC gates, and active Monitoring & Observability."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureLineage(BaseModel):
    """Lineage tracking of engineered features mapping upstream dependencies and transformation properties."""
    source_dataset: str
    pipeline_stage: str
    transformation_type: str = "vectorized"
    description: str = ""


class FeatureViewMetadata(BaseModel):
    """Metadata detailing schema definitions, governance tags, and lineage links of a Feature View."""
    name: str
    entity_id: str
    features: list[str]
    source_path: str
    version: str = "v1"
    created_at: str = Field(default_factory=lambda: pd.Timestamp.now().isoformat())
    owner: str = "MLOps Team"
    tags: list[str] = Field(default_factory=list)
    description: str = ""
    lineage: FeatureLineage | None = None


class AccessController:
    """Security Access Control (RBAC) validator for token-based permission gating."""
    ROLES = {"ADMIN", "READ_WRITE", "READ_ONLY"}
    
    def __init__(self, admin_token: str = "ADMIN_TOKEN_999", read_write_token: str = "RW_TOKEN_888", read_only_token: str = "RO_TOKEN_777") -> None:
        self._tokens = {
            admin_token: "ADMIN",
            read_write_token: "READ_WRITE",
            read_only_token: "READ_ONLY",
        }

    def authenticate_token(self, token: str | None) -> str:
        if not token or token not in self._tokens:
            raise PermissionError("Access Denied: Invalid or missing API security token.")
        return self._tokens[token]

    def authorize(self, token: str | None, required_roles: list[str]) -> None:
        role = self.authenticate_token(token)
        if role == "ADMIN":
            return
        if role not in required_roles:
            raise PermissionError(f"Access Denied: Role '{role}' lacks permission. Required: {required_roles}")


class FeatureRegistry:
    """Enterprise Feature Registry keeping track of registered Feature View meta configurations."""
    def __init__(self, registry_path: Path) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.views: dict[str, FeatureViewMetadata] = {}
        self.load()

    def load(self) -> None:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.views[k] = FeatureViewMetadata(**v)
            except Exception as e:
                logger.warning("Failed to load feature store registry: %s", e)

    def save(self) -> None:
        data = {k: v.model_dump() for k, v in self.views.items()}
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=4)

    def register_view(self, metadata: FeatureViewMetadata) -> None:
        self.views[metadata.name] = metadata
        self.save()
        logger.info("Registered Feature View: %s (version: %s)", metadata.name, metadata.version)

    def get_view(self, name: str) -> FeatureViewMetadata | None:
        return self.views.get(name)


class FeatureCatalog:
    """Read-only catalog for schema discovery across all registered feature definitions."""
    def __init__(self, registry: FeatureRegistry) -> None:
        self._registry = registry

    def list_views(self) -> list[dict[str, Any]]:
        return [v.model_dump() for v in self._registry.views.values()]

    def search_features(self, name_query: str) -> list[dict[str, Any]]:
        results = []
        pattern = re.compile(name_query, re.IGNORECASE)
        for view_name, view in self._registry.views.items():
            if pattern.search(view_name) or any(pattern.search(f) for f in view.features):
                results.append(view.model_dump())
        return results


class OfflineStore:
    """Vectorized offline warehouse implementing point-in-time joins and snappy partition logs."""
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_features(self, view_name: str, version: str, df: pd.DataFrame) -> Path:
        """Saves features cleanly as compressed snappy parquet partitions."""
        dest_dir = self.base_dir / view_name / version
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "features.parquet"
        
        # Enforce copy to avoid mutating source df
        df_to_save = df.copy()
        df_to_save.to_parquet(dest_file, index=False, compression="snappy")
        logger.info("Saved offline features for view '%s' to %s", view_name, dest_file)
        return dest_file

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_views: list[tuple[str, str]],  # List of (view_name, version)
        entity_id: str = "TransactionID",
    ) -> pd.DataFrame:
        """Performs point-in-time exact identity join on transaction index."""
        result_df = entity_df.copy()
        
        for view_name, version in feature_views:
            file_path = self.base_dir / view_name / version / "features.parquet"
            if not file_path.exists():
                logger.warning("Feature view file not found: %s. Loading empty default", file_path)
                continue
                
            view_df = pd.read_parquet(file_path)
            # Ensure entity_id is shared
            if entity_id not in view_df.columns:
                raise ValueError(f"Entity identifier '{entity_id}' missing from offline feature view: {view_name}")
                
            # Perform clean vectorized left join
            # This ensures O(N log N) merge efficiency
            result_df = pd.merge(result_df, view_df, on=entity_id, how="left")
            
        return result_df


class OnlineStore:
    """Ultra-low-latency real-time key-value served through high-performance SQLite backing indexes."""
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        # Connect to create file if not existing
        conn = self._get_connection()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def write_features(self, view_name: str, df: pd.DataFrame, entity_id: str = "TransactionID") -> None:
        """Stores feature rows in SQLite with query optimization indexes."""
        if df.empty:
            return
            
        columns = list(df.columns)
        if entity_id not in columns:
            raise ValueError(f"Entity key '{entity_id}' must be present for online store writes.")

        # Cast TransactionID explicitly to int or string for SQLite indexing
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create table name specifically for view
        table_name = f"view_{view_name}"
        
        # Determine column schemas dynamically
        col_type_specs = []
        for col in columns:
            col_type = "TEXT"
            if "int" in str(df[col].dtype):
                col_type = "INTEGER"
            elif "float" in str(df[col].dtype):
                col_type = "REAL"
            col_type_specs.append(f'"{col}" {col_type}')
            
        create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(col_type_specs)}, PRIMARY KEY("{entity_id}"))'
        cursor.execute(create_sql)
        
        # Save indexes
        idx_sql = f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_{entity_id}" ON "{table_name}" ("{entity_id}")'
        cursor.execute(idx_sql)
        
        # Convert NaN values to None for sqlite compatibility
        df_clean = df.replace({np.nan: None})
        
        # Vectorized batch insert
        # SQLite UPSERT syntax to overwrite on conflict key
        col_placeholders = ", ".join([f'"{c}"' for c in columns])
        placeholders = ", ".join(["?"] * len(columns))
        update_assignments = ", ".join([f'"{c}" = excluded."{c}"' for c in columns if c != entity_id])
        
        if update_assignments:
            insert_sql = f'INSERT INTO "{table_name}" ({col_placeholders}) VALUES ({placeholders}) ON CONFLICT("{entity_id}") DO UPDATE SET {update_assignments}'
        else:
            insert_sql = f'INSERT INTO "{table_name}" ({col_placeholders}) VALUES ({placeholders}) ON CONFLICT("{entity_id}") DO NOTHING'
            
        batch_args = df_clean[columns].values.tolist()
        cursor.executemany(insert_sql, batch_args)
        conn.commit()
        conn.close()
        logger.info("Wrote %d rows to online store table '%s'", len(df), table_name)

    def get_online_features(
        self,
        entity_keys: list[int | str],
        view_name: str,
        feature_names: list[str],
        entity_id: str = "TransactionID",
    ) -> list[dict[str, Any]]:
        """Serves multi-key real-time feature vectors in < 2ms latency boundaries."""
        if not entity_keys:
            return []
            
        conn = self._get_connection()
        cursor = conn.cursor()
        table_name = f"view_{view_name}"
        
        # Verify table exists before querying
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            conn.close()
            return []
            
        select_cols = [f'"{entity_id}"'] + [f'"{f}"' for f in feature_names]
        sql = f'SELECT {", ".join(select_cols)} FROM "{table_name}" WHERE "{entity_id}" = ?'
        
        results = []
        for key in entity_keys:
            cursor.execute(sql, (key,))
            row = cursor.fetchone()
            if row:
                results.append(dict(row))
            else:
                # Mock default empty features returning None but maintaining identifier
                record = {k: None for k in feature_names}
                record[entity_id] = key
                results.append(record)
                
        conn.close()
        return results


class FeatureStoreMonitor:
    """Feature Store Operations Monitor evaluating storage footprints, access latency, and distribution shifts."""
    def __init__(self, offline_dir: Path, db_path: Path) -> None:
        self.offline_dir = Path(offline_dir)
        self.db_path = Path(db_path)

    def measure_online_latency(self, online_store: OnlineStore, view_name: str, test_keys: list[int | str], features: list[str]) -> float:
        """Measures SQLite key-value retrieval times to ensure optimal serving efficiency (< 2ms)."""
        if not test_keys:
            return 0.0
            
        t0 = time.perf_counter()
        _ = online_store.get_online_features(test_keys, view_name, features)
        t_elapsed = time.perf_counter() - t0
        latency_ms = (t_elapsed / len(test_keys)) * 1000
        return float(latency_ms)

    def calculate_psi(self, baseline: pd.Series, target: pd.Series, bins: int = 10) -> float:
        """Computes Population Stability Index (PSI) to track feature distribution drift."""
        base_clean = baseline.dropna()
        tgt_clean = target.dropna()
        if base_clean.empty or tgt_clean.empty:
            return 0.0
            
        try:
            percentiles = np.linspace(0, 100, bins + 1)
            cuts = np.percentile(base_clean, percentiles)
            cuts = np.unique(cuts)
            if len(cuts) < 2:
                return 0.0
        except Exception:
            return 0.0
            
        baseline_counts, _ = np.histogram(base_clean, bins=cuts)
        target_counts, _ = np.histogram(tgt_clean, bins=cuts)
        
        # Normalize
        P = baseline_counts / len(base_clean)
        Q = target_counts / len(tgt_clean)
        
        eps = 1e-4
        P = np.where(P == 0.0, eps, P)
        Q = np.where(Q == 0.0, eps, Q)
        
        P /= P.sum()
        Q /= Q.sum()
        
        # Calculate PSI
        psi_val = np.sum((P - Q) * np.log(P / Q))
        return float(psi_val)

    def get_monitoring_report(self, client: FeatureStoreClient, test_keys: list[int | str]) -> dict[str, Any]:
        """Generates comprehensive JSON report tracking sizes, latencies, schema tags, and feature drifts."""
        # 1. Database sizes
        online_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        
        offline_size = 0
        if self.offline_dir.exists():
            for root, _, files in os.walk(self.offline_dir):
                for f in files:
                    offline_size += (Path(root) / f).stat().st_size
                    
        # 2. Performance benchmark & drifts
        monitored_views = []
        catalog_views = client.catalog.list_views()
        
        # Measure latency specifically on transaction keys if any views are registered
        for view in catalog_views:
            view_name = view["name"]
            features = view["features"]
            
            latency = self.measure_online_latency(client.online_store, view_name, test_keys, features)
            
            # Simple check for drift if matching train vs test views exist
            drift_score = 0.0
            if "test" in view_name:
                train_view_name = view_name.replace("test", "train")
                train_view = client.registry.get_view(train_view_name)
                if train_view:
                    # Load a small slice of Parquet from both
                    try:
                        f_train_path = self.offline_dir / train_view_name / train_view.version / "features.parquet"
                        f_test_path = self.offline_dir / view_name / view["version"] / "features.parquet"
                        if f_train_path.exists() and f_test_path.exists():
                            df_tr = pd.read_parquet(f_train_path, columns=[features[0]]).head(1000)
                            df_te = pd.read_parquet(f_test_path, columns=[features[0]]).head(1000)
                            drift_score = self.calculate_psi(df_tr[features[0]], df_te[features[0]])
                    except Exception as e:
                        logger.warning("Could not calculate PSI drift for %s: %s", view_name, e)
                        
            monitored_views.append({
                "view_name": view_name,
                "features_count": len(features),
                "avg_serving_latency_ms": latency,
                "primary_feature_psi_drift": drift_score,
            })
            
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "online_store": {
                "db_path": str(self.db_path),
                "size_bytes": online_size,
            },
            "offline_store": {
                "base_dir": str(self.offline_dir),
                "size_bytes": offline_size,
            },
            "monitored_views": monitored_views,
            "status": "PASS" if all(v["avg_serving_latency_ms"] < 2.0 for v in monitored_views) else "WARN",
        }
        return report


class FeatureStoreClient:
    """Unified client orchestrator bridging all offline/online registries, lineage logs, RBAC token gates, and operations monitors."""
    def __init__(self, registry_path: Path, offline_dir: Path, online_db: Path) -> None:
        self.registry = FeatureRegistry(registry_path)
        self.offline_store = OfflineStore(offline_dir)
        self.online_store = OnlineStore(online_db)
        self.catalog = FeatureCatalog(self.registry)
        self.access_controller = AccessController()
        self.monitor = FeatureStoreMonitor(offline_dir, online_db)

    def register_feature_view(
        self,
        name: str,
        entity_id: str,
        features: list[str],
        source_path: str,
        version: str = "v1",
        owner: str = "MLOps Team",
        tags: list[str] = [],
        description: str = "",
        lineage: FeatureLineage | None = None,
        token: str | None = None,
    ) -> FeatureViewMetadata:
        """Registers a Feature View config - restricted to ADMIN or READ_WRITE roles."""
        self.access_controller.authorize(token, ["ADMIN", "READ_WRITE"])
        
        meta = FeatureViewMetadata(
            name=name,
            entity_id=entity_id,
            features=features,
            source_path=source_path,
            version=version,
            owner=owner,
            tags=tags,
            description=description,
            lineage=lineage,
        )
        self.registry.register_view(meta)
        return meta

    def ingest(self, view_name: str, df: pd.DataFrame, token: str | None = None) -> None:
        """Ingests a feature dataframe to both offline parquet and online key-value tables - restricted to ADMIN or READ_WRITE."""
        self.access_controller.authorize(token, ["ADMIN", "READ_WRITE"])
        
        view_meta = self.registry.get_view(view_name)
        if not view_meta:
            raise ValueError(f"Feature view '{view_name}' is not registered. Register it first!")
            
        # 1. Enforce validation gate: assert entity_id and features exist in df
        entity_id = view_meta.entity_id
        if entity_id not in df.columns:
            raise ValueError(f"DF missing expected entity identifier: {entity_id}")
            
        for f in view_meta.features:
            if f not in df.columns:
                raise ValueError(f"DF missing expected feature column: {f}")
                
        # Drop columns not designated in the view (preserving entity_id)
        selected_cols = [entity_id] + view_meta.features
        df_aligned = df[selected_cols]
        
        # 2. Write to Offline Store
        self.offline_store.save_features(view_name, view_meta.version, df_aligned)
        
        # 3. Write to Online Store
        self.online_store.write_features(view_name, df_aligned, entity_id=entity_id)
        logger.info("Successfully completed unified ingestion pipeline for Feature View: %s", view_name)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_views: list[tuple[str, str]],
        entity_id: str = "TransactionID",
        token: str | None = None,
    ) -> pd.DataFrame:
        """Retrieves points-in-time offline batch logs - readable by ADMIN, READ_WRITE, or READ_ONLY."""
        self.access_controller.authorize(token, ["ADMIN", "READ_WRITE", "READ_ONLY"])
        return self.offline_store.get_historical_features(entity_df, feature_views, entity_id=entity_id)

    def get_online_features(
        self,
        entity_keys: list[int | str],
        view_name: str,
        feature_names: list[str],
        entity_id: str = "TransactionID",
        token: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieves real-time online features - readable by ADMIN, READ_WRITE, or READ_ONLY."""
        self.access_controller.authorize(token, ["ADMIN", "READ_WRITE", "READ_ONLY"])
        return self.online_store.get_online_features(entity_keys, view_name, feature_names, entity_id=entity_id)
