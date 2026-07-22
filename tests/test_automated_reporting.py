"""Unit tests for AutomatedReporter — Part 3.17."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.reporting import AutomatedReporter


@pytest.fixture()
def mock_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates mock train/test datasets."""
    rng = np.random.default_rng(42)
    n = 100

    df_train = pd.DataFrame({
        "TransactionID": range(n),
        "feat_x": rng.normal(50, 10, n),
        "cat_feat": rng.choice(["A", "B"], n),
        "isFraud": rng.choice([0, 1], n, p=[0.9, 0.1]),
    })
    df_test = pd.DataFrame({
        "TransactionID": range(n, n + n),
        "feat_x": rng.normal(51, 10, n),
        "cat_feat": rng.choice(["A", "B"], n),
    })

    return df_train, df_test


@pytest.fixture()
def reporter(mock_datasets: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path) -> AutomatedReporter:
    # Set workspace root temporarily to tmp_path or let it write reports into custom root dir
    rep = AutomatedReporter(
        df_train=mock_datasets[0],
        df_test=mock_datasets[1],
        config_path="tests/config.yaml",
    )
    # Override report root and rebuild directory structural dict to use tmp_path
    rep.reports_root = tmp_path / "reports"
    rep.dirs = {k: tmp_path / "reports" / v.name for k, v in rep.dirs.items()}
    for d in rep.dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return rep


def test_collect_metadata(reporter: AutomatedReporter) -> None:
    meta = reporter.collect_metadata()
    assert meta["project_name"] == "IEEE-CIS-Fraud-Detection-Optimization"
    assert meta["train_dataset_shape"] == [100, 4]
    assert meta["test_dataset_shape"] == [100, 3]
    
    metadata_json = reporter.dirs["metadata"] / "report_metadata.json"
    assert metadata_json.exists()
    
    with open(metadata_json) as f:
        stored = json.load(f)
    assert stored["project_name"] == "IEEE-CIS-Fraud-Detection-Optimization"


def test_generate_json_report(reporter: AutomatedReporter) -> None:
    meta = reporter.collect_metadata()
    sub_meta = reporter.read_submodule_metadata()
    report = reporter.generate_json_report(meta, sub_meta)
    
    assert report["data_summary"]["train_rows"] == 100
    assert report["data_summary"]["test_rows"] == 100
    assert (reporter.dirs["json"] / "eda_report.json").exists()


def test_generate_html_report(reporter: AutomatedReporter) -> None:
    meta = reporter.collect_metadata()
    sub_meta = reporter.read_submodule_metadata()
    report = reporter.generate_json_report(meta, sub_meta)
    
    reporter.generate_html_report(meta, report)
    assert (reporter.dirs["dashboards"] / "dashboard_summary.html").exists()
    assert (reporter.dirs["html"] / "eda_report.html").exists()
    assert (reporter.dirs["html"] / "executive_dashboard.html").exists()


def test_generate_pdf_report(reporter: AutomatedReporter) -> None:
    meta = reporter.collect_metadata()
    sub_meta = reporter.read_submodule_metadata()
    report = reporter.generate_json_report(meta, sub_meta)
    
    reporter.generate_pdf_report(meta, report)
    assert (reporter.dirs["pdf"] / "eda_report.pdf").exists()
    assert (reporter.dirs["pdf"] / "executive_summary.pdf").exists()


def test_generate_markdown_report(reporter: AutomatedReporter) -> None:
    meta = reporter.collect_metadata()
    sub_meta = reporter.read_submodule_metadata()
    report = reporter.generate_json_report(meta, sub_meta)
    
    reporter.generate_markdown_report(meta, report)
    assert (reporter.dirs["markdown"] / "eda_report.md").exists()
    assert (reporter.dirs["markdown"] / "analysis_summary.md").exists()
    assert (reporter.dirs["markdown"] / "feature_catalog.md").exists()


def test_run_dvc_indexing(reporter: AutomatedReporter) -> None:
    reporter.run_dvc_indexing()
    assert (reporter.dirs["dvc"] / "dvc_artifact_manifest.json").exists()
    assert (reporter.dirs["dvc"] / "dvc_pipeline.yaml").exists()


def test_run_report_versioning(reporter: AutomatedReporter) -> None:
    meta = reporter.collect_metadata()
    reporter.run_report_versioning(meta)
    assert (reporter.dirs["metadata"] / "report_version_history.json").exists()
    assert (reporter.dirs["markdown"] / "report_changelog.md").exists()


def test_run_quality_validation(reporter: AutomatedReporter) -> None:
    meta = reporter.collect_metadata()
    sub_meta = reporter.read_submodule_metadata()
    report = reporter.generate_json_report(meta, sub_meta)
    
    # Generate files to pass checks
    reporter.generate_html_report(meta, report)
    reporter.generate_pdf_report(meta, report)
    
    checks = reporter.run_quality_validation()
    assert checks["overall_quality_assessment"] == "APPROVED"
    assert checks["verification_score"] == 100.0
    assert (reporter.dirs["metadata"] / "report_validation.json").exists()
    assert (reporter.dirs["metadata"] / "quality_assurance_report.csv").exists()


def test_run_automated_publishing(reporter: AutomatedReporter) -> None:
    checks = {"overall_quality_assessment": "APPROVED"}
    reporter.run_automated_publishing(checks)
    assert (reporter.dirs["metadata"] / "publishing_status.json").exists()
    assert (reporter.dirs["metadata"] / "published_artifacts.csv").exists()


def test_run_all(reporter: AutomatedReporter) -> None:
    reporter.run_all()
    # Check that main artifacts exist
    assert (reporter.dirs["html"] / "eda_report.html").exists()
    assert (reporter.dirs["pdf"] / "eda_report.pdf").exists()
    assert (reporter.dirs["json"] / "eda_report.json").exists()
    assert (reporter.dirs["markdown"] / "eda_report.md").exists()
    assert (reporter.dirs["dashboards"] / "dashboard_summary.html").exists()
    assert (reporter.dirs["metadata"] / "report_validation.json").exists()
    assert (reporter.dirs["metadata"] / "publishing_status.json").exists()
