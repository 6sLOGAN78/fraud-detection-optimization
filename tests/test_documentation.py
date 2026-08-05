"""Unit tests for Part 15 — System Documentation, Technical Specifications & ADRs."""

from pathlib import Path
import pytest

from src.utils.docs_generator import DocumentationStrategyGate, DocumentationSuiteGenerator


def test_15_1_docs_strategy_gate():
    gate = DocumentationStrategyGate()
    assert gate.verify() is True


def test_15_2_to_15_13_docs_suite_validation():
    generator = DocumentationSuiteGenerator()
    report = generator.validate_documentation_suite(".")

    assert report["total_mandatory_docs"] == 12
    assert report["present_docs_count"] == 12
    assert report["missing_docs_count"] == 0
    assert report["documentation_suite_complete"] is True
    assert report["status"] == "COMPLETE"
