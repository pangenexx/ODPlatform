"""Inference service for running YOLO predictions."""

from pathlib import Path

from ultralytics import YOLO

from odp_platform.config.infer_config import InferConfig
from odp_platform.common.logging_utils import setup_logger


logger = setup_logger(__name__)


class InferenceService:
    """Service for running inference with YOLO models."""

    def __init__(self, config: InferConfig):
        self.config = config
        self.model = YOLO(config.model)

    def predict(self, source: Path | None = None) -> list:
        """Run inference on source images/videos."""
        src = source or self.config.source
        logger.info(f"Running inference on: {src}")

        results = self.model(
            source=str(src),
            imgsz=self.config.imgsz,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            save=self.config.save_results,
            project=str(self.config.save_dir) if self.config.save_dir else None,
        )

        logger.info(f"Inference completed. Processed {len(results)} images.")
        return results
