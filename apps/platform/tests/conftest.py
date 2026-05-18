"""Test configuration."""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    (workspace / ".odp-workspace").touch()
    return workspace


@pytest.fixture
def sample_class_names():
    """Return sample class names for testing."""
    return ["aircraft", "ship", "storage_tank", "baseball_diamond"]
