"""Base configuration class."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    """Base configuration for all platform operations."""

    project_name: str = "default"
    project_root: Path = Field(default_factory=Path.cwd)

    class Config:
        extra = "allow"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

    def save(self, path: Path) -> None:
        """Save config to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: Path) -> "BaseConfig":
        """Load config from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
