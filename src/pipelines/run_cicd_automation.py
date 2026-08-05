"""Pipeline script to execute Part 14 — MLOps CI/CD, Quality, Security & Release Automation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.utils import (
    AutomatedTestingPipelineRunner,
    BuildPackagingEngine,
    CICDPreExecutionGate,
    CodeQualityChecker,
    ContinuousDeploymentPipeline,
    GitHubActionsWorkflowBuilder,
    InfrastructureAsCodeValidator,
    LintingFormattingEngine,
    ReleaseManager,
    SecurityScanner,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 14 MLOps CI/CD Automation Pipeline")
    args = parser.parse_args()

    gate = CICDPreExecutionGate()
    gate.verify()

    logger.info("Generating GitHub Actions CI/CD workflow...")
    workflow_builder = GitHubActionsWorkflowBuilder()
    workflow_path = workflow_builder.generate_ci_cd_workflow()

    logger.info("Executing Code Quality checks...")
    quality_checker = CodeQualityChecker()
    quality_res = quality_checker.check_file_quality("src/utils/quality_security.py")

    logger.info("Executing Linting & Formatting checks...")
    linter = LintingFormattingEngine()
    lint_res = linter.check_formatting("src")

    logger.info("Executing Security Vulnerability & Secret Scan...")
    sec_scanner = SecurityScanner()
    sec_res = sec_scanner.scan_for_secrets_and_vulnerabilities("src")

    logger.info("Executing CI Automated Test stage runner...")
    ci_runner = AutomatedTestingPipelineRunner()
    ci_res = ci_runner.run_ci_pipeline()

    logger.info("Executing Build Packaging Engine...")
    build_engine = BuildPackagingEngine()
    build_manifest = build_engine.generate_build_manifest(version="v1.0.0")

    logger.info("Executing Infrastructure as Code (IaC) Validation...")
    iac_validator = InfrastructureAsCodeValidator()
    iac_res = iac_validator.validate_docker_compose("docker-compose.yml")

    logger.info("Executing Continuous Deployment (CD) Gate...")
    cd_pipeline = ContinuousDeploymentPipeline()
    cd_res = cd_pipeline.evaluate_cd_release_gate(
        environment="Production",
        ci_tests_passed=ci_res["tests_stage_passed"],
        security_passed=sec_res["security_passed"],
    )

    logger.info("Creating Release Manifest & Changelog...")
    release_mgr = ReleaseManager()
    release_res = release_mgr.create_release(major=1, minor=0, patch=0)

    summary_report = {
        "workflow_file": str(workflow_path),
        "code_quality": quality_res,
        "linting": lint_res,
        "security_scan": sec_res,
        "ci_pipeline": ci_res,
        "build_manifest": build_manifest,
        "iac_validation": iac_res,
        "cd_release_gate": cd_res,
        "release_manifest": release_res,
    }

    out_file = Path("reports/cicd/cicd_automation_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary_report, f, indent=2)

    logger.info(f"Part 14 CI/CD Automation Pipeline completed successfully. Report saved to {out_file}")


if __name__ == "__main__":
    main()
