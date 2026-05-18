"""Training service for orchestrating YOLO training."""

from pathlib import Path

from ultralytics import YOLO

from odp_platform.config.train_config import TrainConfig
from odp_platform.common.logging_utils import setup_logger
from odp_platform.common.paths import get_runs_dir


logger = setup_logger(__name__)


class TrainingService:
    """Service for managing YOLO training runs."""

    def __init__(self, config: TrainConfig):
        self.config = config

    def train(self) -> Path:
        """Run training and return the path to results."""
        logger.info(f"Starting training with model: {self.config.model}")
        logger.info(f"Dataset: {self.config.data}")
        logger.info(f"Epochs: {self.config.epochs}, Batch size: {self.config.batch_size}")

        model = YOLO(self.config.model)

        results = model.train(
            data=str(self.config.data),
            epochs=self.config.epochs,
            imgsz=self.config.imgsz,
            batch=self.config.batch_size,
            device=self.config.device,
            workers=self.config.workers,
            optimizer=self.config.optimizer,
            lr0=self.config.lr0,
            lrf=self.config.lrf,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            patience=self.config.patience,
            project=str(get_runs_dir() / "train"),
            name=self.config.project_name,
            save_period=self.config.save_period,
        )

        logger.info(f"Training completed. Results saved to: {results.save_dir}")
        return results.save_dir
