"""Training configuration."""

from pathlib import Path

from pydantic import Field

from odp_platform.config.base import BaseConfig


class TrainConfig(BaseConfig):
    """Configuration for training pipeline."""

    model: str = "yolo11n.pt"
    data: Path = Field(description="Path to dataset YAML")
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    device: str = "0"
    workers: int = 8
    optimizer: str = "SGD"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    patience: int = 50
    save_period: int = 10
