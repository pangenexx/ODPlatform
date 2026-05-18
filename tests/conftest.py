"""Cross-app E2E test configuration."""

import pytest
from pathlib import Path


@pytest.fixture
def workspace_root(tmp_path):
    """Create a temporary workspace root."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".odp-workspace").touch()
    return workspace
