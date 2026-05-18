"""Configuration merger for combining multiple config sources."""

from odp_platform.config.base import BaseConfig


def merge_configs(base: BaseConfig, override: dict) -> BaseConfig:
    """Merge override dict into base config."""
    base_dict = base.to_dict()
    merged = {**base_dict, **override}
    return type(base)(**merged)
