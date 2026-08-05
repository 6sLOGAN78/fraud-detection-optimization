"""Unit tests for Part 14 — MLOps CI/CD Automation, Quality, Security & Release Management."""

import tempfile
from pathlib import Path

import pytest

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


def test_14_1_to_14_3_cicd_architecture():
    gate = CICDPreExecutionGate()
    assert gate.verify() is True

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "ci_cd.yml"
        builder = GitHubActionsWorkflowBuilder()
        generated = builder.generate_ci_cd_workflow(str(out_path))
        assert generated.exists()

    ci_runner = AutomatedTestingPipelineRunner()
    res = ci_runner.run_ci_pipeline()
    assert "pipeline_status" in res


def test_14_4_to_14_6_quality_and_security():
    quality = CodeQualityChecker()
    res = quality.check_file_quality("src/utils/quality_security.py")
    assert "docstring_coverage_ratio" in res

    linter = LintingFormattingEngine()
    lint_res = linter.check_formatting("src/utils")
    assert "formatting_passed" in lint_res

    scanner = SecurityScanner()
    sec_res = scanner.scan_for_secrets_and_vulnerabilities("src/utils")
    assert sec_res["security_passed"] is True


def test_14_7_to_14_10_release_and_iac():
    build_engine = BuildPackagingEngine()
    manifest = build_engine.generate_build_manifest(version="v1.0.0")
    assert manifest["version_tag"] == "v1.0.0"

    iac = InfrastructureAsCodeValidator()
    iac_res = iac.validate_docker_compose("docker-compose.yml")
    assert "valid_yaml" in iac_res

    cd = ContinuousDeploymentPipeline()
    cd_res = cd.evaluate_cd_release_gate("Staging", ci_tests_passed=True, security_passed=True)
    assert cd_res["deployment_approved"] is True

    with tempfile.TemporaryDirectory() as tmpdir:
        release_mgr = ReleaseManager()
        rel_res = release_mgr.create_release(major=1, minor=0, patch=0, release_dir=tmpdir)
        assert rel_res["release_version"] == "v1.0.0"
