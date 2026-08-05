"""10.10 Reproducibility Framework Module.

Provides seed lock control across libraries, environment snapshotting,
and runtime execution hardware metadata logging.
"""

from __future__ import annotations

import logging
import os
import platform
import random
import sys
from typing import Any, Dict

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReproducibilityFramework:
    """10.10 Manages global random seed locking and environment runtime snapshotting."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def set_global_seed(self) -> int:
        """Sets random seeds deterministically across Python, NumPy, and environment variables."""
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Set PyTorch seed if available
        try:
            import torch
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

        logger.info(f"Global random seed locked to {self.seed}")
        return self.seed

    def capture_environment_snapshot(self) -> Dict[str, Any]:
        """Captures Python version, OS platform, CPU architecture, and environment metadata."""
        return {
            "python_version": sys.version,
            "os_platform": platform.platform(),
            "processor": platform.processor(),
            "seed": self.seed,
            "environment_vars": {
                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", None),
                "MLFLOW_ALLOW_FILE_STORE": os.environ.get("MLFLOW_ALLOW_FILE_STORE", None),
            },
        }
