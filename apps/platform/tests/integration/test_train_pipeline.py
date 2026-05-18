"""Integration test for training pipeline."""

import pytest
from pathlib import Path

from odp_platform.config.train_config import TrainConfig


class TestTrainPipeline:
    """Test training pipeline workflow."""

    def test_train_config_creation(self, tmp_path):
        """Test training config can be created and saved."""
        config = TrainConfig(
            project_name="test_run",
            model="yolo11n.pt",
            epochs=10,
            batch_size=8,
        )
        output_path = tmp_path / "test_train.yaml"
        config.save(output_path)
        assert output_path.exists()

    def test_train_config_load(self, tmp_path):
        """Test training config can be loaded."""
        config = TrainConfig(
            project_name="test_run",
            epochs=10,
        )
        output_path = tmp_path / "test_train.yaml"
        config.save(output_path)

        loaded = TrainConfig.load(output_path)
        assert loaded.project_name == "test_run"
        assert loaded.epochs == 10
