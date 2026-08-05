"""7.9 Best Configuration Registry Module.

Provides registry management, configuration serialization (JSON/YAML),
versioning, and persistence for optimal model hyperparameters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BestConfigurationRegistry:
    """Manages the persistence, serialization, and retrieval of optimal hyperparameter configurations."""

    def __init__(self, registry_dir: Union[str, Path] = "configs/optimized"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def register_configuration(
        self,
        model_name: str,
        best_params: Dict[str, Any],
        best_score: float,
        metric_name: str = "roc_auc",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Saves best hyperparameter configuration to JSON and YAML in the registry directory."""
        config_data = {
            "model_name": model_name,
            "metric_name": metric_name,
            "best_score": float(best_score),
            "hyperparameters": best_params,
            "metadata": metadata or {},
        }

        # Save JSON
        json_path = self.registry_dir / f"{model_name}_best_config.json"
        with open(json_path, "w") as f:
            json.dump(config_data, f, indent=2)

        # Save YAML
        yaml_path = self.registry_dir / f"{model_name}_best_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        logger.info(
            f"Successfully registered optimal configuration for '{model_name}' to {json_path} and {yaml_path}"
        )
        return json_path

    def load_configuration(self, model_name: str) -> Dict[str, Any]:
        """Loads optimal hyperparameter configuration for a model from the registry."""
        json_path = self.registry_dir / f"{model_name}_best_config.json"
        yaml_path = self.registry_dir / f"{model_name}_best_config.yaml"

        if json_path.exists():
            with open(json_path, "r") as f:
                return json.load(f)
        elif yaml_path.exists():
            with open(yaml_path, "r") as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(
                f"No registered configuration found for model '{model_name}' in {self.registry_dir}"
            )

    def list_registered_models(self) -> Dict[str, Path]:
        """Lists all model names currently present in the configuration registry."""
        models = {}
        for p in self.registry_dir.glob("*_best_config.json"):
            model_name = p.stem.replace("_best_config", "")
            models[model_name] = p
        return models
