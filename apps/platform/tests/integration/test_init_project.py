"""Integration test for project initialization."""

import pytest
from pathlib import Path

from odp_platform.config.generator import generate_train_config, generate_val_config, generate_infer_config


class TestInitProject:
    """Test project initialization workflow."""

    def test_generate_train_config(self, tmp_path):
        """Test generating training config."""
        output_path = tmp_path / "train.yaml"
        generate_train_config(output_path)
        assert output_path.exists()

    def test_generate_val_config(self, tmp_path):
        """Test generating validation config."""
        output_path = tmp_path / "val.yaml"
        generate_val_config(output_path)
        assert output_path.exists()

    def test_generate_infer_config(self, tmp_path):
        """Test generating inference config."""
        output_path = tmp_path / "infer.yaml"
        generate_infer_config(output_path)
        assert output_path.exists()
