"""Configuration generator for creating default configs."""

from pathlib import Path

from odp_platform.config.train_config import TrainConfig
from odp_platform.config.val_config import ValConfig
from odp_platform.config.infer_config import InferConfig


def generate_train_config(output_path: Path) -> None:
    """Generate a default training config file."""
    config = TrainConfig()
    config.save(output_path)


def generate_val_config(output_path: Path) -> None:
    """Generate a default validation config file."""
    config = ValConfig()
    config.save(output_path)


def generate_infer_config(output_path: Path) -> None:
    """Generate a default inference config file."""
    config = InferConfig()
    config.save(output_path)
