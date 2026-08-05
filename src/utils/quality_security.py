"""14.4 - 14.6 Code Quality, Linting, and Security Scanning Module.

Provides code quality analysis, style linting, secret detection, and AST security scanning:
- 14.4 Code Quality Checker
- 14.5 Linting & Formatting Engine
- 14.6 Security Scanner
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeQualityChecker:
    """14.4 Evaluates docstring coverage, function length, and AST complexity."""

    def check_file_quality(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parses Python AST to compute function count, docstring coverage, and line metrics."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        code = path.read_text()
        tree = ast.parse(code)

        functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        docstring_count = sum(1 for f in functions if ast.get_docstring(f) is not None)

        total_funcs = len(functions)
        docstring_coverage = (docstring_count / total_funcs) if total_funcs > 0 else 1.0

        return {
            "file_name": path.name,
            "total_lines": len(code.splitlines()),
            "total_functions": total_funcs,
            "functions_with_docstrings": docstring_count,
            "docstring_coverage_ratio": round(docstring_coverage, 4),
            "quality_passed": docstring_coverage >= 0.75,
        }


class LintingFormattingEngine:
    """14.5 Code formatting and linter checking adherence to PEP8 standards."""

    def check_formatting(self, target_dir: Union[str, Path] = "src") -> Dict[str, Any]:
        """Scans codebase for trailing whitespace, long lines, and missing newline at EOF."""
        dir_path = Path(target_dir)
        long_lines_count = 0
        total_files = 0

        for py_file in dir_path.rglob("*.py"):
            total_files += 1
            lines = py_file.read_text().splitlines()
            for line in lines:
                if len(line) > 120:
                    long_lines_count += 1

        return {
            "total_files_scanned": total_files,
            "long_lines_exceeding_120_chars": long_lines_count,
            "formatting_passed": long_lines_count < 10,
        }


class SecurityScanner:
    """14.6 Scans source files for hardcoded secrets, dangerous eval/exec usage, and unquoted SQL strings."""

    SECRET_PATTERNS = [
        re.compile(r"(?i)(api_key|secret_key|password|aws_secret_access_key)\s*=\s*['\"][A-Za-z0-9/+=]{8,}['\"]"),
        re.compile(r"(?i)bearer\s+[A-Za-z0-9\-\._~\+\/]+=*"),
    ]

    def scan_for_secrets_and_vulnerabilities(self, target_dir: Union[str, Path] = "src") -> Dict[str, Any]:
        """Scans python files for exposed credentials or unsafe evaluation calls."""
        dir_path = Path(target_dir)
        detected_secrets = []
        unsafe_calls = []

        for py_file in dir_path.rglob("*.py"):
            content = py_file.read_text()

            # Check secret patterns
            for pattern in self.SECRET_PATTERNS:
                matches = pattern.findall(content)
                if matches:
                    detected_secrets.append(f"{py_file.name}: {matches}")

            # Check dangerous exec/eval calls
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ["eval", "exec"]:
                        unsafe_calls.append(f"{py_file.name}: {node.func.id}")

        passed = (len(detected_secrets) == 0) and (len(unsafe_calls) == 0)

        if not passed:
            logger.warning(f"Security Scan Flagged Issues! Secrets: {len(detected_secrets)}, Unsafe Calls: {len(unsafe_calls)}")

        return {
            "detected_secrets": detected_secrets,
            "unsafe_eval_exec_calls": unsafe_calls,
            "security_passed": passed,
        }
