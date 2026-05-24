from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from odp_platform.common.paths import (
    APP_DIR,
    CHECKPOINTS_DIR,
    CONFIGS_DATASETS_DIR,
    CONFIGS_DIR,
    DATA_DIR,
    ROOT_DIR,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONFIG_TASKS = ["train", "val", "predict"]
BACKEND_BASE_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    def render(self) -> str:
        lines = [
            f"$ {' '.join(self.command)}",
            f"exit code: {self.returncode}",
        ]
        if self.stdout.strip():
            lines.extend(["", "[stdout]", self.stdout.strip()])
        if self.stderr.strip():
            lines.extend(["", "[stderr]", self.stderr.strip()])
        return "\n".join(lines)


def platform_env() -> dict[str, str]:
    env = os.environ.copy()
    src_dir = str(APP_DIR / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def run_python_module(
    module: str,
    args: list[str],
    timeout: int | None = 300,
) -> CommandResult:
    command = [sys.executable, "-m", module, *args]
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT_DIR,
            env=platform_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(command, proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stderr = (stderr + f"\n命令超时: {timeout}s").strip()
        return CommandResult(command, 124, stdout, stderr)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def list_dataset_names() -> list[str]:
    return sorted(path.stem for path in CONFIGS_DATASETS_DIR.glob("*.yaml"))


def dataset_yaml(dataset: str) -> Path:
    return CONFIGS_DATASETS_DIR / f"{dataset}.yaml"


def normalize_names(raw: Any) -> dict[int, str]:
    if isinstance(raw, dict):
        names: dict[int, str] = {}
        for key, value in raw.items():
            try:
                names[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return names
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    return {}


def resolve_dataset_root(config: dict[str, Any], yaml_path: Path) -> Path:
    raw = str(config.get("path") or "").strip()
    if not raw:
        return ROOT_DIR

    raw_path = Path(raw)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    candidates = [
        ROOT_DIR / raw_path,
        yaml_path.parent / raw_path,
        APP_DIR / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if ":" in raw or "\\" in raw:
        return DATA_DIR
    return (ROOT_DIR / raw_path).resolve(strict=False)


def resolve_split_dir(config: dict[str, Any], yaml_path: Path, split: str) -> Path:
    split_value = str(config.get(split) or "").strip()
    if not split_value:
        return Path()
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path
    return resolve_dataset_root(config, yaml_path) / split_path


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_path_for_image(image_path: Path, images_dir: Path) -> Path:
    try:
        relative = image_path.relative_to(images_dir)
    except ValueError:
        relative = Path(image_path.name)
    labels_dir = images_dir.parent / "labels"
    return labels_dir / relative.with_suffix(".txt")


def list_config_files() -> list[str]:
    return sorted(str(path) for path in CONFIGS_DIR.glob("*.yaml"))


def list_model_files() -> list[str]:
    roots = [CHECKPOINTS_DIR, ROOT_DIR / "models" / "checkpoints"]
    seen: set[Path] = set()
    models: list[str] = []
    for root in roots:
        for path in root.glob("*.pt"):
            resolved = path.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            models.append(str(path))
    return sorted(models)


def file_mtime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except OSError:
        return ""


def recent_files(root: Path, pattern: str = "*", limit: int = 10) -> list[Path]:
    if not root.exists():
        return []
    files = [path for path in root.rglob(pattern) if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files[:limit]


def fetch_backend_json(path: str, params: dict[str, Any] | None = None) -> Any:
    query = urllib.parse.urlencode(params or {})
    url = f"{BACKEND_BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)
