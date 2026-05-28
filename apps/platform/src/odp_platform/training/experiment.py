from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import (
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    LOGGING_DIR,
    RUNS_DIR,
    dataset_yaml_path,
)
from odp_platform.training import TrainingHooks
from odp_platform.training.callbacks import _safe_float, normalize_csv_row


logger = get_logger(
    base_path=LOGGING_DIR,
    log_type="train",
    logger_name="odp-train",
)


BACKEND_URL = "http://127.0.0.1:8000"


# =========================================================
# Config
# =========================================================

@dataclass
class ExperimentConfig:
    name: str
    dataset: str
    model: str = "yolo11n.pt"
    task: str = "detect"
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    lr0: float = 0.01
    device: str = ""
    workers: int = 2
    optimizer: str = "auto"
    amp: bool = True
    patience: int = 50
    seed: int = 42
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# =========================================================
# Result
# =========================================================

@dataclass
class ExperimentResult:
    name: str
    dataset: str
    model: str
    imgsz: int
    epochs_run: int
    best_epoch: int
    map50: float
    map50_95: float
    precision: float
    recall: float
    train_duration_sec: float
    model_path: str
    config_snapshot_path: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# =========================================================
# Utils
# =========================================================

def _parse_metrics(csv_path: Path) -> dict:
    """解析 Ultralytics results.csv，自动适配列名变化。"""
    if not csv_path.exists():
        logger.warning(f"results.csv 不存在: {csv_path}")
        return {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.warning("results.csv 为空")
        return {}

    last = normalize_csv_row(rows[-1])

    return {
        "epoch": int(float(last.get("epoch", 0))),
        "box_loss": _safe_float(last.get("box_loss", 0)),
        "val_box_loss": _safe_float(last.get("val_box_loss", 0)),
        "map50": _safe_float(last.get("map50", 0)),
        "map50_95": _safe_float(last.get("map50_95", 0)),
        "precision": _safe_float(last.get("precision", 0)),
        "recall": _safe_float(last.get("recall", 0)),
        "lr": _safe_float(last.get("lr", 0)),
    }


def _sync_to_backend(
    name: str,
    dataset: str,
    model: str,
    metrics: dict,
    model_path: str,
) -> None:
    """训练完成后直接 POST 到后端持久化。"""
    import requests
    try:
        payload = {
            "name": name,
            "dataset": dataset,
            "model": model,
            "best_map50": metrics.get("map50", 0),
            "model_path": model_path,
        }
        r = requests.post(f"{BACKEND_URL}/api/experiments", json=payload, timeout=3)
        r.raise_for_status()
        logger.info("实验结果已持久化到后端: id=%s", r.json().get("id"))
    except Exception as e:
        logger.warning("后端不可达，实验结果仅保存在本地: %s", e)


# =========================================================
# Core
# =========================================================

def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    logger.info("=" * 80)
    logger.info("开始实验: %s", config.name)
    logger.info("=" * 80)

    t0 = time.time()

    # ── 实验目录 ─────────────────────────────────────
    exp_dir = RUNS_DIR / "experiments" / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("实验目录: %s", exp_dir)

    # ── 保存配置快照 ─────────────────────────────────
    snapshot_path = exp_dir / "config_snapshot.json"
    snapshot_path.write_text(config.to_json(), encoding="utf-8")
    logger.info("配置快照已保存: %s", snapshot_path)

    # ── 构造 Ultralytics 参数 ────────────────────────
    data_yaml = dataset_yaml_path(config.dataset)
    if not data_yaml.exists():
        raise FileNotFoundError(f"数据集 YAML 不存在: {data_yaml}")

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": config.epochs,
        "batch": config.batch,
        "imgsz": config.imgsz,
        "lr0": config.lr0,
        "workers": config.workers,
        "optimizer": config.optimizer,
        "amp": config.amp,
        "patience": config.patience,
        "seed": config.seed,
        "project": str(RUNS_DIR / "experiments"),
        "name": config.name,
        "exist_ok": True,
        "task": config.task,
    }
    if config.device:
        train_kwargs["device"] = config.device

    # ── Backend hooks ────────────────────────────────
    hooks = TrainingHooks(
        experiment_name=config.name,
        config_json=config.to_json(),
    )
    hooks.on_train_start(dataset=config.dataset, model=config.model)

    # ── 直接训练（不走子进程） ───────────────────────
    logger.info("训练参数: %s", json.dumps(train_kwargs, ensure_ascii=False, default=str))
    model = YOLO(config.model)
    results = model.train(**train_kwargs)

    # ── 解析指标 ─────────────────────────────────────
    results_csv = exp_dir / "results.csv"
    metrics = _parse_metrics(results_csv)
    logger.info("最终指标: %s", metrics)

    # ── 复制 checkpoint ──────────────────────────────
    best_pt = exp_dir / "weights" / "best.pt"
    checkpoint_name = f"best_{config.name}.pt"
    checkpoint_path = CHECKPOINTS_DIR / checkpoint_name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if best_pt.exists():
        shutil.copy(best_pt, checkpoint_path)
        logger.info("checkpoint 已复制: %s", checkpoint_path)
    else:
        logger.warning("best.pt 未找到: %s", best_pt)

    # ── 耗时 ─────────────────────────────────────────
    duration = time.time() - t0
    logger.info("训练耗时: %.2f sec", duration)

    # ── Backend hooks 结束 ───────────────────────────
    hooks.on_train_end(
        map50=metrics.get("map50", 0),
        model_path=str(checkpoint_path.resolve()),
    )

    # ── 直接持久化到后端 ─────────────────────────────
    _sync_to_backend(
        name=config.name,
        dataset=config.dataset,
        model=config.model,
        metrics=metrics,
        model_path=str(checkpoint_path.resolve()),
    )

    # ── 返回结果 ─────────────────────────────────────
    result = ExperimentResult(
        name=config.name,
        dataset=config.dataset,
        model=config.model,
        imgsz=config.imgsz,
        epochs_run=metrics.get("epoch", config.epochs),
        best_epoch=metrics.get("epoch", 0),
        map50=metrics.get("map50", 0),
        map50_95=metrics.get("map50_95", 0),
        precision=metrics.get("precision", 0),
        recall=metrics.get("recall", 0),
        train_duration_sec=duration,
        model_path=str(checkpoint_path.resolve()),
        config_snapshot_path=str(snapshot_path.resolve()),
    )

    logger.info("=" * 80)
    logger.info("实验完成: %s  mAP50=%.4f", config.name, result.map50)
    logger.info("=" * 80)

    return result


def run_batch(configs: list[ExperimentConfig]) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    logger.info("开始批量实验: total=%d", len(configs))
    for idx, config in enumerate(configs):
        logger.info("[%d/%d] %s", idx + 1, len(configs), config.name)
        try:
            results.append(run_experiment(config))
        except Exception:
            logger.exception("实验失败: %s", config.name)
    logger.info("批量实验结束: success=%d", len(results))
    return results
