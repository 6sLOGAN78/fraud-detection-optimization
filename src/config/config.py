"""Configuration Entity and Manager module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


@dataclass(frozen=True)
class DataConfig:
    """Config entity for data files and preprocessing features."""
    train_transaction_file: str
    train_identity_file: str
    test_transaction_file: str
    test_identity_file: str
    target_col: str
    id_col: str
    group_col: str
    selection_threshold: float
    missing_threshold: float
    correlation_threshold: float


@dataclass(frozen=True)
class MLflowConfig:
    """Config entity for MLflow parameters."""
    experiment_name: str
    tracking_uri: str
    log_models: bool


@dataclass(frozen=True)
class PathsConfig:
    """Config entity for folders paths."""
    raw_dir: str
    interim_dir: str
    processed_dir: str
    feature_store_dir: str
    metadata_dir: str
    reports_dir: str
    artifacts_dir: str
    experiments_dir: str
    logs_dir: str


@dataclass(frozen=True)
class ModelParamsConfig:
    """Config entity for model parameters."""
    lightgbm: dict[str, Any] = field(default_factory=dict)
    xgboost: dict[str, Any] = field(default_factory=dict)
    catboost: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectConfig:
    """Aggregated Root configuration entity."""
    project_name: str
    seed: int
    cv_folds: int
    verbose: bool
    data: DataConfig
    mlflow: MLflowConfig
    paths: PathsConfig
    model_params: ModelParamsConfig


class ConfigurationManager:
    """Manager to load config files using Hydra/OmegaConf and construct config entities."""
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir).resolve()
        
    def get_config(self) -> ProjectConfig:
        """Compose config via Hydra and map into typed ProjectConfig entity."""
        if not GlobalHydra.instance().is_initialized():
            hydra.initialize_config_dir(config_dir=str(self.config_dir), version_base="1.3")
        cfg = hydra.compose(config_name="config")
        
        # Load and verify sub-configs composed by Hydra
        data_cfg = DataConfig(
            train_transaction_file=cfg.data.train_transaction_file,
            train_identity_file=cfg.data.train_identity_file,
            test_transaction_file=cfg.data.test_transaction_file,
            test_identity_file=cfg.data.test_identity_file,
            target_col=cfg.data.target_col,
            id_col=cfg.data.id_col,
            group_col=cfg.data.group_col,
            selection_threshold=float(cfg.data.features.selection_threshold),
            missing_threshold=float(cfg.data.features.missing_threshold),
            correlation_threshold=float(cfg.data.features.correlation_threshold),
        )
        
        tracking_uri = cfg.mlflow.tracking_uri
        server_alive = False
        if tracking_uri and (tracking_uri.startswith("http://") or tracking_uri.startswith("https://")):
            import urllib.request
            from urllib.error import URLError
            try:
                with urllib.request.urlopen(tracking_uri, timeout=1.0) as conn:
                    if conn.getcode() is not None:
                        server_alive = True
            except (URLError, Exception):
                pass
        
        if not server_alive:
            tracking_uri = "file:./mlruns"
            
        mlflow_cfg = MLflowConfig(
            experiment_name=cfg.mlflow.experiment_name,
            tracking_uri=tracking_uri,
            log_models=bool(cfg.mlflow.log_models),
        )
        
        paths_cfg = PathsConfig(
            raw_dir=cfg.paths.raw_dir,
            interim_dir=cfg.paths.interim_dir,
            processed_dir=cfg.paths.processed_dir,
            feature_store_dir=cfg.paths.feature_store_dir,
            metadata_dir=cfg.paths.metadata_dir,
            reports_dir=cfg.paths.reports_dir,
            artifacts_dir=cfg.paths.artifacts_dir,
            experiments_dir=cfg.paths.experiments_dir,
            logs_dir=cfg.paths.logs_dir,
        )
        
        # Get model param dictionaries
        model_params = ModelParamsConfig(
            lightgbm=dict(OmegaConf.to_container(cfg.model.params, resolve=True)) if "model" in cfg and "params" in cfg.model else {},
            xgboost=dict(OmegaConf.to_container(cfg.model.params, resolve=True)) if "model" in cfg and "params" in cfg.model else {},
            catboost=dict(OmegaConf.to_container(cfg.model.params, resolve=True)) if "model" in cfg and "params" in cfg.model else {},
        )
        
        return ProjectConfig(
            project_name=cfg.project_name,
            seed=cfg.seed,
            cv_folds=cfg.cv_folds,
            verbose=cfg.verbose,
            data=data_cfg,
            mlflow=mlflow_cfg,
            paths=paths_cfg,
            model_params=model_params
        )
