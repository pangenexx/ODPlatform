"""Validation configuration."""

from pathlib import Path

from pydantic import Field

from odp_platform.config.base import BaseConfig


class ValConfig(BaseConfig):
    """Configuration for validation pipeline."""

    model: str = Field(description="Path to model weights")
    data: Path = Field(description="Path to dataset YAML")
    batch_size: int = 32
    imgsz: int = 640
    device: str = "0"
    workers: int = 8
    split: str = "val"
