"""Feature Documentation engine generating structured Markdown documentation and JSON schemas from registry catalog metadata configurations."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from src.data.store import FeatureRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureDocumentationGenerator:
    """Consolidates Feature view registry metadata into clean Markdown books and JSON schema objects."""
    def __init__(self, registry_path: Path, workspace_dir: Path) -> None:
        self.registry = FeatureRegistry(registry_path)
        self.workspace_dir = Path(workspace_dir)

    def get_git_revision(self) -> str:
        """Retrieves active Git commit signature tracking documentation updates."""
        try:
            val = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(self.workspace_dir)).decode("utf-8").strip()
            return val
        except Exception as e:
            logger.warning("Could not read git revision hash: %s", e)
            return "unknown-ref"

    def get_role_description(self, role: str) -> str:
        """Provides access control guides detailing role-based security permissions."""
        roles = {
            "ADMIN": "Full catalog ownership. Read, Write, Delete, and Metadata override capacity.",
            "READ_WRITE": "Feature registration and parquet partition ingestion capabilities. Restricted registry settings.",
            "READ_ONLY": "Consumptive fetch-only query execution rights. Denied feature insertions.",
        }
        return roles.get(role, "No defined security clearances.")

    def compile_markdown_documentation(self, output_file: Path) -> str:
        """Generates comprehensive Markdown document summarizing registered Feature definitions, ownership, tags, and lifecycle states."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        git_rev = self.get_git_revision()
        
        md = []
        md.append("# Feature Store Registry Catalog Documentation")
        md.append("Automated documentation generator logs capturing schema definitions and governance guidelines.")
        md.append("")
        md.append("## System Environment & Lineage")
        md.append(f"- **Active Git Commit**: `{git_rev}`")
        md.append(f"- **Metadata Format**: Standard JSON Registry")
        md.append(f"- **Total Registered Views**: {len(self.registry.views)}")
        md.append("")
        
        md.append("## Access Control & Usage Guidelines")
        md.append("Strict token authorization guards serving permissions based on API credential roles:")
        md.append("")
        md.append("| Access Role | Description | Allowed Operations |")
        md.append("| :--- | :--- | :--- |")
        md.append(f"| `ADMIN` | {self.get_role_description('ADMIN')} | `register_feature_view`, `ingest`, queries |")
        md.append(f"| `READ_WRITE` | {self.get_role_description('READ_WRITE')} | `register_feature_view`, `ingest`, queries |")
        md.append(f"| `READ_ONLY` | {self.get_role_description('READ_ONLY')} | Historical retrieve, real-time key queries |")
        md.append("")

        md.append("## Feature Views Catalog")
        
        if not self.registry.views:
            md.append("No active feature view configurations detected.")
        else:
            for view_name, view in self.registry.views.items():
                md.append(f"### Feature View: `{view_name}`")
                md.append(f"- **Entity Index Identifier**: `{view.entity_id}`")
                md.append(f"- **Version Tag**: `{view.version}`")
                md.append(f"- **Owner**: `{view.owner}`")
                md.append(f"- **Lifecycle Status**: Active")
                md.append(f"- **Created At**: `{view.created_at}`")
                md.append(f"- **Feature List**: {', '.join([f'`{f}`' for f in view.features])}")
                
                if view.description:
                    md.append(f"- **Scope Target Description**: {view.description}")
                    
                if view.tags:
                    md.append(f"- **Governance Tags**: {', '.join([f'`{t}`' for t in view.tags])}")
                    
                lineage = view.lineage
                if lineage:
                    md.append("- **Lineage Tracking**:")
                    md.append(f"  - *Upstream Source Dataset*: `{lineage.source_dataset}`")
                    md.append(f"  - *Pipeline Stage Ingestion*: `{lineage.pipeline_stage}`")
                    md.append(f"  - *Vectorized Transformation Model*: `{lineage.transformation_type}`")
                md.append("")

        content = "\n".join(md)
        with open(output_file, "w") as f:
            f.write(content)
            
        logger.info("Compiled Feature Store Markdown documentation to: %s", output_file)
        return content

    def compile_json_manifest(self, output_file: Path) -> dict[str, Any]:
        """Exports unified JSON analytical structure details mapping schemas and registries for REST API endpoints."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        git_rev = self.get_git_revision()
        
        manifest = {
            "git_revision": git_rev,
            "version_control_integrated": True,
            "access_control_policies": {
                role: self.get_role_description(role) for role in ["ADMIN", "READ_WRITE", "READ_ONLY"]
            },
            "registered_views_count": len(self.registry.views),
            "feature_views": {}
        }
        
        for view_name, view in self.registry.views.items():
            manifest["feature_views"][view_name] = {
                "entity_id": view.entity_id,
                "version": view.version,
                "owner": view.owner,
                "lifecycle_state": "ACTIVE",
                "registered_features": view.features,
                "tags": view.tags,
                "description": view.description,
                "lineage": view.lineage.model_dump() if view.lineage else None
            }
            
        with open(output_file, "w") as f:
            json.dump(manifest, f, indent=4)
            
        logger.info("Wrote Feature Store JSON manifest documentation to: %s", output_file)
        return manifest
