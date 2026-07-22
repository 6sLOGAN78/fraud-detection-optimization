"""Wrapper pipeline script to execute Feature Store Documentation compiler, logging outputs to MLflow as artifacts."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow

from src.data.documentation import FeatureDocumentationGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    logger.info("Executing Pre-Execution Pipeline Verification Gate...")

    # Define paths
    registry_path = Path("data/feature_store_foundation/registry.json")
    
    if not registry_path.exists():
        msg = f"Dependency verification failed! Missing prior feature store registry configuration: {registry_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Dependency verification passed. Initializing Feature Documentation Generator...")

    workspace_dir = Path(os.getcwd())
    output_dir = Path("data/feature_store_foundation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    md_output = output_dir / "feature_documentation.md"
    json_output = output_dir / "feature_documentation_manifest.json"

    # Generate documents
    generator = FeatureDocumentationGenerator(registry_path, workspace_dir)
    
    logger.info("Compiling feature store documentation files...")
    generator.compile_markdown_documentation(md_output)
    generator.compile_json_manifest(json_output)

    # MLflow tracking
    logger.info("Logging documentation metadata and files to MLflow...")
    active = mlflow.active_run()
    started = False
    if active is None:
        mlflow.start_run(run_name="feature_store_documentation")
        started = True

    try:
        # Log metadata parameters
        mlflow.log_params({
            "pipeline_stage": "feature_documentation",
            "git_commit": generator.get_git_revision()[:8],
            "feature_documentation_md_path": str(md_output),
            "feature_documentation_json_path": str(json_output),
        })
        
        # Log artifacts
        mlflow.log_artifact(str(md_output), artifact_path="feature_store")
        mlflow.log_artifact(str(json_output), artifact_path="feature_store")
    except Exception as e:
        logger.warning("MLflow tracking logging encountered warning: %s", e)
    finally:
        if started:
            mlflow.end_run()

    logger.info("Feature Store Documentation pipeline completed successfully.")


if __name__ == "__main__":
    main()
