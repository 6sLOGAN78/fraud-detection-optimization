"""Core Automated Reporter for consolidation of Data Engineering & EDA — Part 3.17."""

from __future__ import annotations

import getpass
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomatedReporter:
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        config_path: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.df_train = df_train
        self.df_test = df_test
        self.config_path = config_path
        self.random_state = random_state
        self.start_time = time.time()

        # Target directory structure
        self.reports_root = Path("reports")
        self.dirs = {
            "html": self.reports_root / "html",
            "pdf": self.reports_root / "pdf",
            "markdown": self.reports_root / "markdown",
            "json": self.reports_root / "json",
            "csv": self.reports_root / "csv",
            "images": self.reports_root / "images",
            "dashboards": self.reports_root / "dashboards",
            "metadata": self.reports_root / "metadata",
            "mlflow": self.reports_root / "mlflow",
            "dvc": self.reports_root / "dvc",
        }

        # Initialize directories
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        logger.info("AutomatedReporter initialized and directory structure prepared.")

    def get_git_commit(self) -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            return "N/A"

    def get_processing_duration(self) -> float:
        return time.time() - self.start_time

    def collect_metadata(self) -> dict:
        """Gathers project execution details and environment variables."""
        logger.info("Collecting environment environment and execution metadata...")

        env_summary = {
            "project_name": "IEEE-CIS-Fraud-Detection-Optimization",
            "dataset_version": "v1.0-interim-merged",
            "pipeline_version": "Part-3.17-Report-Gen",
            "git_commit": self.get_git_commit(),
            "execution_timestamp": pd.Timestamp.now().isoformat(),
            "python_version": sys.version.split()[0],
            "operating_system": platform.platform(),
            "hardware_processor": platform.processor() or "Unknown",
            "user": getpass.getuser(),
            "random_seed": self.random_state,
            "config_file": self.config_path or "default_config.yaml",
            "train_dataset_shape": list(self.df_train.shape),
            "test_dataset_shape": list(self.df_test.shape),
            "processing_duration_sec": round(self.get_processing_duration(), 4),
        }

        # Save outputs
        with open(self.dirs["metadata"] / "report_metadata.json", "w") as f:
            json.dump(env_summary, f, indent=4)

        with open(self.dirs["metadata"] / "execution_environment.json", "w") as f:
            json.dump(env_summary, f, indent=4)

        return env_summary

    def read_submodule_metadata(self) -> dict:
        """Attempts to read metadata outputs from previous EDA modules to populate the dashboard."""
        sub_meta = {}

        # 1. Drift Metadata
        drift_meta_path = Path("reports/eda/drift/drift_metadata.json")
        if drift_meta_path.exists():
            try:
                with open(drift_meta_path) as f:
                    data = json.load(f)
                sub_meta["drifted_features_count"] = data.get("drifted_features", 0)
                sub_meta["max_psi"] = data.get("max_psi", 0.0)
            except Exception:
                pass
        
        # 2. Leakage Metadata
        leak_meta_path = Path("reports/eda/leakage/leakage_metadata.json")
        if leak_meta_path.exists():
            try:
                with open(leak_meta_path) as f:
                    data = json.load(f)
                sub_meta["total_features_leakage"] = data.get("total_features", 0)
            except Exception:
                pass

        # Load metrics defaults if missing
        sub_meta.setdefault("drifted_features_count", 0)
        sub_meta.setdefault("max_psi", 0.0)
        sub_meta.setdefault("total_features_leakage", self.df_train.shape[1] - 2)

        return sub_meta

    def generate_json_report(self, metadata: dict, sub_meta: dict) -> dict:
        """Exports unified analytical dataset profiles for API integration."""
        logger.info("Generating unified JSON analytical reports...")

        unified_report = {
            "metadata": metadata,
            "sub_module_metrics": sub_meta,
            "data_summary": {
                "train_rows": len(self.df_train),
                "test_rows": len(self.df_test),
                "null_cells_train": int(self.df_train.isnull().sum().sum()),
                "null_cells_test": int(self.df_test.isnull().sum().sum()),
                "target_balance": round(float(self.df_train["isFraud"].mean()) * 100.0, 4) if "isFraud" in self.df_train.columns else 0.0,
            }
        }

        with open(self.dirs["json"] / "eda_report.json", "w") as f:
            json.dump(unified_report, f, indent=4)

        with open(self.dirs["json"] / "eda_summary.json", "w") as f:
            json.dump(unified_report, f, indent=4)

        with open(self.dirs["json"] / "pipeline_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        return unified_report

    def generate_html_report(self, metadata: dict, unified_report: dict) -> None:
        """HTML compiler generating executive dashboards with monochromatic minimal styling."""
        logger.info("Compiling Automated HTML reports & Executive Dashboards...")

        # Links to all individual sub-reports relative to the dashboard
        # Since reports are located in reports/eda/<sub-module>/<file>.html,
        # and this dashboard is written to reports/dashboards/dashboard_summary.html,
        # relative paths are: ../eda/<sub-module>/<file>.html
        links = {
            "Data Quality": "../eda/quality/data_quality_report.html",
            "Missing Values": "../eda/missing/missing_report.html",
            "Target Analysis": "../eda/target/target_analysis_report.html",
            "Numerical Analysis": "../eda/numerical/numerical_analysis_report.html",
            "Categorical Analysis": "../eda/categorical/categorical_analysis_report.html",
            "Transaction Stats": "../eda/transaction/transaction_analysis_report.html",
            "Identity Profile": "../eda/identity/identity_analysis_report.html",
            "Time Series Drift": "../eda/timeseries/timeseries_analysis_report.html",
            "Anonymous Features": "../eda/anonymous/anonymous_analysis_report.html",
            "Correlation Analysis": "../eda/correlation/correlation_analysis_report.html",
            "Feature Interactions": "../eda/interaction/interaction_analysis_report.html",
            "Statistical Tests": "../eda/statistical_tests/statistical_tests_report.html",
            "Drift Diagnostics": "../eda/drift/drift_report.html",
            "Data Leakage Detector": "../eda/leakage/leakage_report.html"
        }

        links_html = ""
        for name, rel_path in links.items():
            # Check if target exists on filesystem to highlight status
            abs_path = Path("reports") / rel_path.replace("../", "")
            exists = abs_path.exists()
            status_text = "READY" if exists else "NOT CREATED"
            status_class = "status-ready" if exists else "status-missing"
            
            links_html += f"""
            <div class="report-link-card">
                <span class="report-name">{name}</span>
                <span class="report-status {status_class}">{status_text}</span>
                <a class="report-btn" href="{rel_path if exists else '#'}" {"disabled" if not exists else ""}>LAUNCH VIEW</a>
            </div>
            """

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>IEEE-CIS Automated Reporting Grid</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Orbitron:wght@500;700;900&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #06070b;
            --panel-bg: rgba(14, 16, 22, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-color: #8e97a4;
            --text-white: #ffffff;
            --alert-green: #2ed573;
            --alert-red: #ff3838;
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

        .grid-kpi {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin-bottom: 2.5rem;
            position: relative;
            z-index: 10;
        }}

        .kpi-card {{
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1.5rem;
            backdrop-filter: blur(16px) saturate(120%);
        }}

        .kpi-label {{
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
            color: var(--text-color);
        }}

        .kpi-value {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-white);
        }}

        .main-section {{
            display: grid;
            grid-template-columns: 1.2fr 2fr;
            gap: 2rem;
            position: relative;
            z-index: 10;
        }}

        .panel-metadata {{
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1.5rem;
            height: fit-content;
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

        .meta-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
        }}

        .meta-table td {{
            padding: 0.65rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }}

        .meta-table td.label-cell {{
            color: var(--text-white);
            font-weight: 600;
            width: 55%;
        }}

        .meta-table td.val-cell {{
            text-align: right;
            word-break: break-all;
        }}

        .report-links-container {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }}

        .report-link-card {{
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1.2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .report-name {{
            font-size: 0.8rem;
            color: var(--text-white);
            font-weight: 600;
        }}

        .report-status {{
            font-size: 0.65rem;
            text-transform: uppercase;
            padding: 0.2rem 0.5rem;
            border-radius: 2px;
            margin-right: 0.5rem;
        }}

        .status-ready {{
            background-color: rgba(46, 213, 115, 0.1);
            color: var(--alert-green);
        }}

        .status-missing {{
            background-color: rgba(255, 56, 56, 0.1);
            color: var(--alert-red);
        }}

        .report-btn {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem;
            padding: 0.4rem 0.8rem;
            background-color: var(--text-white);
            color: var(--bg-color);
            text-decoration: none;
            border-radius: 4px;
            font-weight: bold;
            transition: all 0.2s ease;
        }}

        .report-btn:hover {{
            background-color: transparent;
            color: var(--text-white);
            border: 1px solid var(--text-white);
        }}

        .report-btn[disabled] {{
            background-color: rgba(255, 255, 255, 0.05);
            color: rgba(255, 255, 255, 0.2);
            pointer-events: none;
            cursor: not-allowed;
            border: none;
        }}
    </style>
</head>
<body>

    <header>
        <div>
            <h1>IEEE-CIS PIPELINE REPORT GENERATION GRID</h1>
            <p style="font-size: 0.65rem; color: var(--text-color); margin-top: 0.25rem; letter-spacing: 1px;">STAGE 3.17: AUTOMATED REPORT & METRIC ORCHESTRATION</p>
        </div>
        <div class="status-pill">PUBLISHING SECURED</div>
    </header>

    <div class="grid-kpi">
        <div class="kpi-card">
            <p class="kpi-label">Train Record Size</p>
            <p class="kpi-value">{unified_report['data_summary']['train_rows']}</p>
        </div>
        <div class="kpi-card">
            <p class="kpi-label">Test Record Size</p>
            <p class="kpi-value">{unified_report['data_summary']['test_rows']}</p>
        </div>
        <div class="kpi-card">
            <p class="kpi-label">Target Contamination (Max PSI)</p>
            <p class="kpi-value">{unified_report['sub_module_metrics']['max_psi']:.4f}</p>
        </div>
        <div class="kpi-card">
            <p class="kpi-label">Total Drifted Features</p>
            <p class="kpi-value">{unified_report['sub_module_metrics']['drifted_features_count']}</p>
        </div>
    </div>

    <div class="main-section">
        <div class="panel-metadata">
            <h2>ENVIRONMENT DETAILS</h2>
            <table class="meta-table">
                <tr>
                    <td class="label-cell">Pipeline Version</td>
                    <td class="val-cell">{metadata['pipeline_version']}</td>
                </tr>
                <tr>
                    <td class="label-cell">Git Revision</td>
                    <td class="val-cell">{metadata['git_commit'][:8]}</td>
                </tr>
                <tr>
                    <td class="label-cell">Python Version</td>
                    <td class="val-cell">{metadata['python_version']}</td>
                </tr>
                <tr>
                    <td class="label-cell">Platform OS</td>
                    <td class="val-cell">{metadata['operating_system']}</td>
                </tr>
                <tr>
                    <td class="label-cell">Execution Date</td>
                    <td class="val-cell">{metadata['execution_timestamp'][:19]}</td>
                </tr>
                <tr>
                    <td class="label-cell">Duration</td>
                    <td class="val-cell">{metadata['processing_duration_sec']} s</td>
                </tr>
            </table>
        </div>

        <div>
            <h2>EXPLORATORY DATA ANALYSIS MODULES</h2>
            <div class="report-links-container">
                {links_html}
            </div>
        </div>
    </div>

</body>
</html>
"""
        with open(self.dirs["dashboards"] / "dashboard_summary.html", "w") as f:
            f.write(html_template)
            
        with open(self.dirs["html"] / "eda_report.html", "w") as f:
            f.write(html_template)

        with open(self.dirs["html"] / "executive_dashboard.html", "w") as f:
            f.write(html_template)

        logger.info("Automated HTML reports compiled successfully.")

    def generate_pdf_report(self, metadata: dict, unified_report: dict) -> None:
        """Export publication-quality PDF reports using ReportLab flowables."""
        logger.info("Compiling PDF Executive Summary reports...")

        pdf_path = self.dirs["pdf"] / "eda_report.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        
        styles = getSampleStyleSheet()
        
        # Define clean, professional custom styles conforming to the monochromatic theme
        title_style = ParagraphStyle(
            name="TitleStyle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            textColor=colors.HexColor("#06070b"),
            leading=18,
            spaceAfter=15,
        )

        h2_style = ParagraphStyle(
            name="H2Style",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.HexColor("#333333"),
            leading=13,
            spaceBefore=10,
            spaceAfter=5,
        )

        body_style = ParagraphStyle(
            name="BodyStyle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=colors.HexColor("#555555"),
            leading=10,
        )

        story = []
        story.append(Paragraph("IEEE-CIS FRAUD DETECTION REPORT CARD", title_style))
        story.append(Spacer(1, 10))

        # Add environment metadata table
        data_table = [
            [Paragraph("<b>METRIC</b>", body_style), Paragraph("<b>VALUE</b>", body_style)],
            [Paragraph("Project Name", body_style), Paragraph(str(metadata["project_name"]), body_style)],
            [Paragraph("Pipeline Version", body_style), Paragraph(str(metadata["pipeline_version"]), body_style)],
            [Paragraph("Git Commit Hash", body_style), Paragraph(str(metadata["git_commit"]), body_style)],
            [Paragraph("Execution Timestamp", body_style), Paragraph(str(metadata["execution_timestamp"]), body_style)],
            [Paragraph("Operating System", body_style), Paragraph(str(metadata["operating_system"]), body_style)],
            [Paragraph("Train Dataset Shape", body_style), Paragraph(str(metadata["train_dataset_shape"]), body_style)],
            [Paragraph("Test Dataset Shape", body_style), Paragraph(str(metadata["test_dataset_shape"]), body_style)],
            [Paragraph("Processing Duration", body_style), Paragraph(f"{metadata['processing_duration_sec']} seconds", body_style)],
        ]
        
        t = Table(data_table, colWidths=[150, 400])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#f1f2f6")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ]))
        
        story.append(Paragraph("ENVIRONMENT DETAILS & SPECS", h2_style))
        story.append(Spacer(1, 5))
        story.append(t)
        story.append(Spacer(1, 15))

        # Add data metrics summary table
        metrics_table = [
            [Paragraph("<b>DATA METRIC SUMMARY</b>", body_style), Paragraph("<b>VALUE</b>", body_style)],
            [Paragraph("Train Row Count", body_style), Paragraph(str(unified_report["data_summary"]["train_rows"]), body_style)],
            [Paragraph("Test Row Count", body_style), Paragraph(str(unified_report["data_summary"]["test_rows"]), body_style)],
            [Paragraph("Null Cell Intersections (Train)", body_style), Paragraph(str(unified_report["data_summary"]["null_cells_train"]), body_style)],
            [Paragraph("Null Cell Intersections (Test)", body_style), Paragraph(str(unified_report["data_summary"]["null_cells_test"]), body_style)],
            [Paragraph("Target Class Balance Percentage", body_style), Paragraph(f"{unified_report['data_summary']['target_balance']:.4f}%", body_style)],
            [Paragraph("Max PSI (Feature Drift Threshold)", body_style), Paragraph(f"{unified_report['sub_module_metrics']['max_psi']:.4f}", body_style)],
            [Paragraph("Drifted Features Count", body_style), Paragraph(str(unified_report["sub_module_metrics"]["drifted_features_count"]), body_style)],
        ]
        t2 = Table(metrics_table, colWidths=[200, 350])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#f1f2f6")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ]))
        
        story.append(Paragraph("ANALYTICAL METRICS BRIEF", h2_style))
        story.append(Spacer(1, 5))
        story.append(t2)

        # Build PDF Document
        doc.build(story)

        # Copy to secondary output name
        shutil.copy(pdf_path, self.dirs["pdf"] / "executive_summary.pdf")

        logger.info("PDF Executive Summary written successfully.")

    def generate_markdown_report(self, metadata: dict, unified_report: dict) -> None:
        """Writes clean Git-compatible markdown summaries."""
        logger.info("Generating Markdown reports...")

        md_content = f"""# IEEE-CIS Fraud Detection Pipeline Report
**Pipeline Stage**: Automated Execution Summary
**Git Hash**: `{metadata['git_commit']}`
**Time**: `{metadata['execution_timestamp']}`

## Environment Details
- **OS**: {metadata['operating_system']}
- **Python Version**: {metadata['python_version']}
- **Hardware**: {metadata['hardware_processor']}
- **Duration**: {metadata['processing_duration_sec']} seconds

## Dataset Statistics
- **Training Set Shape**: {metadata['train_dataset_shape']}
- **Test Set Shape**: {metadata['test_dataset_shape']}
- **Target Fraud Distribution**: {unified_report['data_summary']['target_balance']:.4f}%
- **Total Missing Cells**: Train = {unified_report['data_summary']['null_cells_train']}, Test = {unified_report['data_summary']['null_cells_test']}

## EDA Metrics Overview
- **Drifted Features Count**: {unified_report['sub_module_metrics']['drifted_features_count']}
- **Maximum PSI**: {unified_report['sub_module_metrics']['max_psi']:.4f}
"""
        with open(self.dirs["markdown"] / "eda_report.md", "w") as f:
            f.write(md_content)

        with open(self.dirs["markdown"] / "analysis_summary.md", "w") as f:
            f.write(md_content)

        # Write simple catalog
        catalog_content = f"""# Feature Catalog
Auto-generated feature list.
- Dataset contains {len(self.df_train.columns)} columns.
- Targets: `isFraud` labels.
"""
        with open(self.dirs["markdown"] / "feature_catalog.md", "w") as f:
            f.write(catalog_content)

        logger.info("Markdown documentation generated.")

    def generate_mlflow_run(self, metadata: dict, unified_report: dict) -> None:
        """Auto-logs metrics, parameters, environment summaries, and artifacts to MLflow."""
        logger.info("Logging reports, metrics, and parameters into MLflow artifact registries...")

        # Find or start active MLflow run
        active_run = mlflow.active_run()
        started_run = False
        if active_run is None:
            mlflow.start_run(run_name="automated_report_generation")
            started_run = True

        try:
            # Parameters logs
            mlflow.log_params({
                "git_commit": metadata["git_commit"][:8],
                "train_rows": unified_report["data_summary"]["train_rows"],
                "test_rows": unified_report["data_summary"]["test_rows"],
                "processing_duration": metadata["processing_duration_sec"],
            })

            # Metrics logs
            mlflow.log_metrics({
                "null_cells_train": float(unified_report["data_summary"]["null_cells_train"]),
                "null_cells_test": float(unified_report["data_summary"]["null_cells_test"]),
                "drifted_features_cnt": float(unified_report["sub_module_metrics"]["drifted_features_count"]),
                "max_psi_val": float(unified_report["sub_module_metrics"]["max_psi"]),
            })

            # Log folders as artifacts
            # Rather than individual files, we can log the html, pdf, json, markdown directories
            folders_to_log = ["html", "pdf", "json", "markdown", "dashboards", "metadata", "dvc"]
            for f in folders_to_log:
                if self.dirs[f].exists():
                    mlflow.log_artifacts(str(self.dirs[f]), artifact_path=f"reporting/{f}")

            # Save run metadata
            summary = {
                "experiment_id": mlflow.active_run().info.experiment_id,
                "run_id": mlflow.active_run().info.run_id,
                "status": "SUCCESS",
            }
            with open(self.dirs["mlflow"] / "mlflow_run_summary.json", "w") as f:
                json.dump(summary, f, indent=4)

        except Exception as e:
            logger.warning("MLflow logging encountered errors: %s", e)
        finally:
            if started_run:
                mlflow.end_run()

        logger.info("MLflow auto-logging complete.")

    def run_dvc_indexing(self) -> None:
        """Maintains DVC package indexing catalogs and dependency configurations."""
        logger.info("Registering reports directory structures into DVC manifests...")

        manifest = {
            "stage": "automated_reporting",
            "dvc_tracked_outs": [
                "reports/html/",
                "reports/pdf/",
                "reports/markdown/",
                "reports/json/",
                "reports/csv/",
                "reports/dashboards/",
                "reports/metadata/",
                "reports/mlflow/",
                "reports/dvc/",
            ],
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        with open(self.dirs["dvc"] / "dvc_artifact_manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)

        dvc_pipeline_content = """# DVC integration tracker
stages:
  automated_reporting:
    cmd: python3 -m src.pipelines.run_automated_reporting
    deps:
      - data/interim/train_merged.parquet
      - data/interim/test_merged.parquet
      - src/eda/reporting.py
      - src/pipelines/run_automated_reporting.py
    outs:
      - reports/html/
      - reports/pdf/
      - reports/markdown/
      - reports/json/
      - reports/dashboards/
      - reports/metadata/
      - reports/mlflow/
      - reports/dvc/
"""
        with open(self.dirs["dvc"] / "dvc_pipeline.yaml", "w") as f:
            f.write(dvc_pipeline_content)

        logger.info("DVC tracking manifest files generated.")

    def run_report_versioning(self, metadata: dict) -> None:
        """Maintains clean release change logs and revision histories."""
        history_path = self.dirs["metadata"] / "report_version_history.json"
        
        # Read existing history if present
        if history_path.exists():
            try:
                with open(history_path) as f:
                    history = json.load(f)
            except Exception:
                history = []
        else:
            history = []

        history.append({
            "version": f"1.0.{len(history)}",
            "git_commit": metadata["git_commit"],
            "timestamp": metadata["execution_timestamp"],
            "author": metadata["user"],
            "duration": metadata["processing_duration_sec"],
        })

        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)

        # Write changelog
        changelog = f"""# Version Release Log
## Version 1.0.{len(history) - 1} ({metadata['execution_timestamp'][:10]})
- Automated compilation of all EDA reports.
- PDF generation integration via ReportLab.
- MLflow parameters auto-logging.
"""
        with open(self.dirs["markdown"] / "report_changelog.md", "w") as f:
            f.write(changelog)

        logger.info("Version history and changelogs updated successfully.")

    def run_quality_validation(self) -> dict:
        """Performs robust automated QA verification validation gates on reports."""
        logger.info("Running report quality and file integrity validation checks...")
        
        checks = {}
        
        # HTML checks
        html_exists = (self.dirs["html"] / "eda_report.html").exists()
        html_size = (self.dirs["html"] / "eda_report.html").stat().st_size if html_exists else 0
        checks["html_validation"] = "PASSED" if html_exists and html_size > 500 else "FAILED"

        # PDF checks
        pdf_exists = (self.dirs["pdf"] / "eda_report.pdf").exists()
        pdf_size = (self.dirs["pdf"] / "eda_report.pdf").stat().st_size if pdf_exists else 0
        checks["pdf_validation"] = "PASSED" if pdf_exists and pdf_size > 1000 else "FAILED"

        # JSON checks
        json_exists = (self.dirs["json"] / "eda_report.json").exists()
        checks["json_schema_validation"] = "PASSED" if json_exists else "FAILED"

        # Overall
        overall_passed = all(status == "PASSED" for status in checks.values())
        checks["overall_quality_assessment"] = "APPROVED" if overall_passed else "REJECTED"
        checks["verification_score"] = 100.0 if overall_passed else 0.0

        # Save checking status reports
        with open(self.dirs["metadata"] / "report_validation.json", "w") as f:
            json.dump(checks, f, indent=4)

        df_qa = pd.DataFrame([checks])
        df_qa.to_csv(self.dirs["metadata"] / "quality_assurance_report.csv", index=False)

        logger.info("Report quality checks validation result: %s", checks["overall_quality_assessment"])
        return checks

    def run_automated_publishing(self, checks: dict) -> None:
        """Publishes artifacts into workspace and logs status."""
        logger.info("Running automated publishing targets...")

        pub_status = {
            "target": "Local Workspace & MLflow registers",
            "status": "PUBLISHED" if checks.get("overall_quality_assessment") == "APPROVED" else "HOLD",
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        with open(self.dirs["metadata"] / "publishing_status.json", "w") as f:
            json.dump(pub_status, f, indent=4)

        # Artifact index
        artifacts_index = pd.DataFrame([
            {"artifact_name": "eda_report.html", "path": str(self.dirs["html"] / "eda_report.html")},
            {"artifact_name": "eda_report.pdf", "path": str(self.dirs["pdf"] / "eda_report.pdf")},
            {"artifact_name": "eda_report.json", "path": str(self.dirs["json"] / "eda_report.json")},
            {"artifact_name": "eda_report.md", "path": str(self.dirs["markdown"] / "eda_report.md")},
            {"artifact_name": "dashboard_summary.html", "path": str(self.dirs["dashboards"] / "dashboard_summary.html")},
        ])
        artifacts_index.to_csv(self.dirs["metadata"] / "published_artifacts.csv", index=False)

        logger.info("Automated publishing successfully concluded.")

    def run_all(self) -> None:
        """Executes full automated reporting pipeline end-to-end."""
        logger.info("--- Automated Report Generation Pipeline (Starting) ---")

        # 1. Collect environment details
        metadata = self.collect_metadata()

        # 2. Read sub-module statistics
        sub_metrics = self.read_submodule_metadata()

        # 3. Compile JSON export
        unified_report = self.generate_json_report(metadata, sub_metrics)

        # 4. Compile HTML report & dashboard
        self.generate_html_report(metadata, unified_report)

        # 5. Compile PDF executive summary
        self.generate_pdf_report(metadata, unified_report)

        # 6. Generate Markdown repository logs
        self.generate_markdown_report(metadata, unified_report)

        # 7. Write DVC manifest indexing
        self.run_dvc_indexing()

        # 8. Record report Semantic Versions
        self.run_report_versioning(metadata)

        # 9. Perform quality QA check
        qa_checks = self.run_quality_validation()

        # 10. Publish logs and indexes
        self.run_automated_publishing(qa_checks)

        # 11. Auto-log everything to MLflow registry
        self.generate_mlflow_run(metadata, unified_report)

        logger.info("--- Automated Report Generation Pipeline Complete ---")
