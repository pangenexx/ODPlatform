"""Path utilities for workspace-aware path resolution."""

from pathlib import Path


def find_workspace_root(start: Path | None = None) -> Path:
    """Find the workspace root by looking for .odp-workspace marker file."""
    current = start or Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".odp-workspace").exists():
            return parent
    raise RuntimeError("Cannot find workspace root. Missing .odp-workspace marker file.")


def get_workspace_root() -> Path:
    """Get the workspace root directory."""
    return find_workspace_root()


def get_data_dir() -> Path:
    """Get the data directory path."""
    return get_workspace_root() / "data"


def get_models_dir() -> Path:
    """Get the models directory path."""
    return get_workspace_root() / "models"


def get_runs_dir() -> Path:
    """Get the runs directory path."""
    return get_workspace_root() / "runs"
