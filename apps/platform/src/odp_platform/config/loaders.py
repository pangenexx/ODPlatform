"""Configuration loaders."""

from pathlib import Path

import yaml

from odp_platform.config.base import BaseConfig
from odp_platform.config.train_config import TrainConfig
from odp_platform.config.val_config import ValConfig
from odp_platform.config.infer_config import InferConfig


def load_yaml(path: Path) -> dict:
    """Load a YAML file and return as dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_train_config(path: Path) -> TrainConfig:
    """Load training configuration from YAML."""
    data = load_yaml(path)
    return TrainConfig(**data)


def load_val_config(path: Path) -> ValConfig:
    """Load validation configuration from YAML."""
    data = load_yaml(path)
    return ValConfig(**data)


def load_infer_config(path: Path) -> InferConfig:
    """Load inference configuration from YAML."""
    data = load_yaml(path)
    return InferConfig(**data)
