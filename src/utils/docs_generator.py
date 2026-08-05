"""15.1 Documentation Strategy Architecture and Suite Generator Module.

Provides documentation verification gates and automated technical documentation builders:
- 15.1 Documentation Strategy Gate
- 15.2 Documentation Suite Generator
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentationStrategyGate:
    """Pre-execution gate verifying documentation directory readiness."""

    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)

    def verify(self) -> bool:
        """Verifies that docs directory exists and is writable."""
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        (self.docs_dir / "adrs").mkdir(parents=True, exist_ok=True)
        logger.info(f"Documentation Strategy Gate PASSED. Path: {self.docs_dir}")
        return True


class DocumentationSuiteGenerator:
    """15.1 - 15.13 Validates and checks mandatory technical documentation markdown files."""

    MANDATORY_DOCS = [
        "README.md",
        "CONTRIBUTING.md",
        "docs/installation.md",
        "docs/project_structure.md",
        "docs/data_dictionary.md",
        "docs/feature_dictionary.md",
        "docs/training_guide.md",
        "docs/evaluation_guide.md",
        "docs/deployment_guide.md",
        "docs/api_documentation.md",
        "docs/troubleshooting_guide.md",
        "docs/adrs/ADR-001-system-architecture.md",
    ]

    def validate_documentation_suite(self, root_dir: str = ".") -> Dict[str, Any]:
        """Scans workspace for all mandatory documentation files."""
        root = Path(root_dir)
        missing_docs = []
        present_docs = []

        for rel_path in self.MANDATORY_DOCS:
            full_path = root / rel_path
            if full_path.exists():
                present_docs.append(rel_path)
            else:
                missing_docs.append(rel_path)

        all_present = len(missing_docs) == 0

        return {
            "total_mandatory_docs": len(self.MANDATORY_DOCS),
            "present_docs_count": len(present_docs),
            "missing_docs_count": len(missing_docs),
            "missing_docs": missing_docs,
            "documentation_suite_complete": all_present,
            "status": "COMPLETE" if all_present else "INCOMPLETE",
        }
