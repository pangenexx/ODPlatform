"""Inference configuration."""

from pathlib import Path

from pydantic import Field

from odp_platform.config.base import BaseConfig


class InferConfig(BaseConfig):
    """Configuration for inference pipeline."""

    model: str = Field(description="Path to model weights")
    source: Path = Field(description="Path to images/videos directory")
    imgsz: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "0"
    save_results: bool = True
    save_dir: Path | None = None
