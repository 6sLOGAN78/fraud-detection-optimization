"""14.7 - 14.10 Build Pipeline, CD Pipeline, Infrastructure as Code, and Release Management Module.

Provides Docker build specs, Staging/Production CD release gates, IaC validation, and semantic versioning:
- 14.7 Build Pipeline & Packaging Engine
- 14.8 Continuous Deployment Pipeline
- 14.9 Infrastructure as Code (IaC) Validator
- 14.10 Release Manager (Semantic versioning & changelog generation)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuildPackagingEngine:
    """14.7 Generates Docker build specification and container tag manifests."""

    def generate_build_manifest(self, version: str = "v1.0.0", image_name: str = "fraud-detection-api") -> Dict[str, Any]:
        """Generates container image build manifest."""
        manifest = {
            "image_name": image_name,
            "version_tag": version,
            "full_image_uri": f"{image_name}:{version}",
            "dockerfile": "Dockerfile",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        logger.info(f"Build Manifest generated for image: {manifest['full_image_uri']}")
        return manifest


class ContinuousDeploymentPipeline:
    """14.8 CD release gate verifying environment readiness for Staging and Production deployment."""

    def evaluate_cd_release_gate(self, environment: str, ci_tests_passed: bool, security_passed: bool) -> Dict[str, Any]:
        """Verifies if code meets CD promotion requirements for target environment."""
        approved = ci_tests_passed and security_passed

        status = "DEPLOY_APPROVED" if approved else "DEPLOY_REJECTED"
        logger.info(f"CD Deployment Gate for '{environment}': {status}")

        return {
            "environment": environment,
            "ci_tests_passed": ci_tests_passed,
            "security_passed": security_passed,
            "deployment_approved": approved,
            "status": status,
        }


class InfrastructureAsCodeValidator:
    """14.9 Validates Docker Compose and deployment infrastructure specifications."""

    def validate_docker_compose(self, compose_file: str = "docker-compose.yml") -> Dict[str, Any]:
        """Parses and validates syntax of docker-compose.yml file."""
        path = Path(compose_file)
        if not path.exists():
            return {"file_found": False, "valid_yaml": False}

        try:
            with open(path, "r") as f:
                content = yaml.safe_load(f)

            has_services = isinstance(content, dict) and "services" in content
            return {
                "file_found": True,
                "valid_yaml": True,
                "has_services": has_services,
                "services": list(content.get("services", {}).keys()) if has_services else [],
            }
        except Exception as e:
            return {"file_found": True, "valid_yaml": False, "error": str(e)}


class ReleaseManager:
    """14.10 Semantic versioning manager creating release tags, changelogs, and release manifests."""

    def create_release(
        self,
        major: int = 1,
        minor: int = 0,
        patch: int = 0,
        changes: Optional[List[str]] = None,
        release_dir: str = "reports/releases",
    ) -> Dict[str, Any]:
        """Creates semantic version release manifest and saves changelog."""
        version_str = f"v{major}.{minor}.{patch}"
        release_notes = changes or [
            "Part 14 CI/CD, Quality, Security, and Release Management implementation",
            "Added GitHub Actions workflow specification",
            "Added automated security scanning and semantic versioning",
        ]

        manifest = {
            "release_version": version_str,
            "release_date": time.strftime("%Y-%m-%d"),
            "changelog": release_notes,
            "author": "Antigravity MLOps",
        }

        out_path = Path(release_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        release_file = out_path / f"release_{version_str}.json"

        with open(release_file, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Created Release Manifest '{version_str}' at {release_file}")
        return manifest
