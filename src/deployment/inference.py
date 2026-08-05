"""11.1 - 11.4 Deployment Architecture, Packaging, Serialization, and Inference Pipeline Module.

Provides pre-execution gates, model packaging, serialization, and unified inference orchestration:
- 11.1 Deployment Architecture & Verification Gate
- 11.2 Model Packaging Engine
- 11.3 Model Serialization & Deserialization Engine
- 11.4 Unified Inference Pipeline
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentPreExecutionGate:
    """Pre-execution verification gate checking required model artifacts and schemas."""

    def __init__(self, required_artifacts: Optional[List[str]] = None):
        self.required_artifacts = required_artifacts or [
            "data/interim/train_cleaned.parquet"
        ]

    def verify(self) -> bool:
        """Verifies that mandatory upstream dependencies exist."""
        missing = []
        for path_str in self.required_artifacts:
            if not Path(path_str).exists():
                missing.append(path_str)

        if missing:
            msg = f"Deployment Pre-Execution Gate FAILED! Missing required artifacts: {missing}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info("Deployment Pre-Execution Gate PASSED.")
        return True


class ModelSerializer:
    """11.3 Handles safe serialization and deserialization of fitted models and preprocessors."""

    @staticmethod
    def save_artifact(obj: Any, file_path: Union[str, Path]) -> Path:
        """Serializes object to disk using joblib."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)
        logger.info(f"Saved artifact to {path}")
        return path

    @staticmethod
    def load_artifact(file_path: Union[str, Path]) -> Any:
        """Loads serialized object from disk safely."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact file not found: {file_path}")
        return joblib.load(path)


class ModelPackager:
    """11.2 Bundles model weights, feature schema, and metadata into a deployment bundle."""

    def create_bundle(
        self,
        model: Any,
        feature_names: List[str],
        version: str = "v1",
        bundle_dir: Union[str, Path] = "artifacts/deployment",
    ) -> Path:
        """Packages model and metadata JSON into bundle directory."""
        dir_path = Path(bundle_dir) / version
        dir_path.mkdir(parents=True, exist_ok=True)

        model_path = dir_path / "model.joblib"
        ModelSerializer.save_artifact(model, model_path)

        meta = {
            "version": version,
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_path = dir_path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Created deployment bundle version '{version}' at {dir_path}")
        return dir_path


class InferencePipeline:
    """11.4 End-to-end inference pipeline validating inputs, applying transformations, and predicting probabilities."""

    def __init__(self, model: Any, feature_names: List[str], decision_threshold: float = 0.5):
        self.model = model
        self.feature_names = feature_names
        self.decision_threshold = decision_threshold

    def preprocess_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validates input feature matrix against expected feature names and fills missing columns."""
        X_proc = X.copy()

        # Align columns
        for col in self.feature_names:
            if col not in X_proc.columns:
                X_proc[col] = 0.0

        X_proc = X_proc[self.feature_names].fillna(0.0)
        return X_proc

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Computes fraud probability predictions for input data."""
        X_clean = self.preprocess_input(X)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_clean)[:, 1]
        else:
            probs = self.model.predict(X_clean)
        return np.clip(probs, 0.0, 1.0)

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Returns binary predictions and probability scores."""
        probs = self.predict_proba(X)
        preds = (probs >= self.decision_threshold).astype(int)
        return preds, probs
