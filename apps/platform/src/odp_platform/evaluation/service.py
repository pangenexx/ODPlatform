"""Evaluation service for orchestrating YOLO validation."""

from pathlib import Path

from ultralytics import YOLO

from odp_platform.config.val_config import ValConfig
from odp_platform.common.logging_utils import setup_logger
from odp_platform.common.paths import get_runs_dir


logger = setup_logger(__name__)


class EvaluationService:
    """Service for managing YOLO model evaluation."""

    def __init__(self, config: ValConfig):
        self.config = config

    def evaluate(self) -> dict:
        """Run evaluation and return metrics."""
        logger.info(f"Starting evaluation with model: {self.config.model}")
        logger.info(f"Dataset: {self.config.data}")

        model = YOLO(self.config.model)

        results = model.val(
            data=str(self.config.data),
            batch=self.config.batch_size,
            imgsz=self.config.imgsz,
            device=self.config.device,
            workers=self.config.workers,
            split=self.config.split,
            project=str(get_runs_dir() / "val"),
            name=self.config.project_name,
        )

        metrics = {
            "map50": float(results.box.map50),
            "map50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }

        logger.info(f"Evaluation completed. mAP50: {metrics['map50']:.4f}")
        return metrics
