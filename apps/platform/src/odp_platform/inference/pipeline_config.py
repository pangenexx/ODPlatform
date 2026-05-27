from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _to_bgr_tuple(value: Any) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    raise ValueError(f"color must be [B,G,R] list/tuple, got: {value!r}")


@dataclass
class PipelineConfig:
    camera_raw: dict[str, Any] = field(default_factory=dict)
    viz_enabled: bool = True
    use_label_mapping: bool = True
    label_mapping: dict[str, str] = field(default_factory=dict)
    color_mapping: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    default_color: tuple[int, int, int] = (0, 255, 0)
    font_path: Optional[str] = None
    style_overrides: dict[str, Any] = field(default_factory=dict)
    source_path: Optional[Path] = None

    def build_camera_config(self):
        if not self.camera_raw:
            return None
        from odp_platform.inference.frame_source import CameraConfig
        return CameraConfig(**self.camera_raw)

    def to_audit(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "camera": self.camera_raw or None,
            "visualization": {
                "enabled": self.viz_enabled,
                "use_label_mapping": self.use_label_mapping,
                "label_mapping": self.label_mapping or None,
                "color_mapping": {k: list(v) for k, v in self.color_mapping.items()} or None,
                "default_color": list(self.default_color),
                "font_path": self.font_path,
                "style": self.style_overrides or None,
            },
        }


def load_pipeline_config(yaml_path: Optional[str] = None) -> PipelineConfig:
    from odp_platform.common.paths import CONFIGS_DIR

    if yaml_path is None:
        path = CONFIGS_DIR / "infer_pipeline.yaml"
    else:
        p = Path(yaml_path)
        path = p if (p.is_absolute() or str(p.parent) != ".") else CONFIGS_DIR / p

    if not path.exists():
        logger.warning(
            f"pipeline config not found: {path}, using defaults "
            f"(beautify on, no label mapping, default camera params). "
            f"create the file or use --pipeline-yaml to specify."
        )
        return PipelineConfig()

    import yaml
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fs = raw.get("frame_source") or {}
    vz = raw.get("visualization") or {}

    color_mapping = {
        str(k): _to_bgr_tuple(v) for k, v in (vz.get("color_mapping") or {}).items()
    }
    default_color = _to_bgr_tuple(vz["default_color"]) if vz.get("default_color") else (0, 255, 0)

    return PipelineConfig(
        camera_raw=fs.get("camera") or {},
        viz_enabled=bool(vz.get("enabled", True)),
        use_label_mapping=bool(vz.get("use_label_mapping", True)),
        label_mapping={str(k): str(v) for k, v in (vz.get("label_mapping") or {}).items()},
        color_mapping=color_mapping,
        default_color=default_color,
        font_path=vz.get("font_path"),
        style_overrides=vz.get("style") or {},
        source_path=path,
    )