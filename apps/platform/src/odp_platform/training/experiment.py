from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import (
    CHECKPOINTS_DIR,
    LOGGING_DIR,
    RUNS_DIR,
)

# backend hooks
# 如果后端尚未完成，可先注释掉
try:
    from odp_platform.training.hooks import (
        on_training_end,
        on_training_start,
    )
except Exception:
    on_training_start = None
    on_training_end = None


logger = get_logger(
    base_path=LOGGING_DIR,
    log_type="train",
    logger_name="odp-train",
)


# =========================================================
# Config
# =========================================================

@dataclass
class ExperimentConfig:
    """
    实验配置
    """

    # =============== 基础 ===============

    name: str
    dataset: str

    # =============== 模型 ===============

    model: str = "yolo11n.pt"
    task: str = "detect"

    # =============== 训练 ===============

    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    lr0: float = 0.01

    # =============== 设备 ===============

    device: str = ""
    workers: int = 2

    # =============== 优化 ===============

    optimizer: str = "auto"
    amp: bool = True
    patience: int = 50

    # =============== 复现 ===============

    seed: int = 42

    # =============== 备注 ===============

    note: str = ""

    # -----------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    # -----------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------

    def to_cli_args(self) -> list[str]:
        """
        转换为 odp-train CLI 参数
        """

        args = [
            "-d",
            self.dataset,

            "--model",
            self.model,

            "--task",
            self.task,

            "--epochs",
            str(self.epochs),

            "--batch",
            str(self.batch),

            "--imgsz",
            str(self.imgsz),

            "--lr0",
            str(self.lr0),

            "--workers",
            str(self.workers),

            "--optimizer",
            self.optimizer,

            "--patience",
            str(self.patience),

            "--seed",
            str(self.seed),

            "--name",
            self.name,
        ]

        if self.device:
            args.extend(["--device", self.device])

        if self.amp:
            args.append("--amp")

        return args


# =========================================================
# Result
# =========================================================

@dataclass
class ExperimentResult:
    """
    实验结果
    """

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

    # 推理直接读取
    model_path: str

    # config snapshot
    config_snapshot_path: str

    # -----------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    # -----------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
        )


# =========================================================
# Utils
# =========================================================

def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


# ---------------------------------------------------------

def _parse_metrics(csv_path: Path) -> dict:
    """
    解析 ultralytics results.csv
    """

    if not csv_path.exists():
        logger.warning(f"results.csv 不存在: {csv_path}")
        return {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.warning("results.csv 为空")
        return {}

    last = rows[-1]

    metrics = {
        "epoch":
            int(float(last.get("epoch", 0))),

        "train_loss":
            _safe_float(last.get("train/box_loss", 0)),

        "val_loss":
            _safe_float(last.get("val/box_loss", 0)),

        "map50":
            _safe_float(last.get("metrics/mAP50(B)", 0)),

        "map50_95":
            _safe_float(last.get("metrics/mAP50-95(B)", 0)),

        "precision":
            _safe_float(last.get("metrics/precision(B)", 0)),

        "recall":
            _safe_float(last.get("metrics/recall(B)", 0)),

        "lr":
            _safe_float(last.get("lr/pg0", 0)),
    }

    return metrics


# =========================================================
# Core
# =========================================================

def run_experiment(
    config: ExperimentConfig,
) -> ExperimentResult:
    """
    核心实验入口

    流程：

    config
      ↓
    odp-train CLI
      ↓
    results.csv
      ↓
    checkpoint copy
      ↓
    ExperimentResult
    """

    logger.info("=" * 80)
    logger.info(f"开始实验: {config.name}")
    logger.info("=" * 80)

    t0 = time.time()

    # =====================================================
    # 实验目录
    # =====================================================

    exp_dir = (
        RUNS_DIR
        / "experiments"
        / config.name
    )

    exp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(f"实验目录: {exp_dir}")

    # =====================================================
    # 保存配置快照
    # =====================================================

    snapshot_path = (
        exp_dir
        / "config_snapshot.json"
    )

    snapshot_path.write_text(
        config.to_json(),
        encoding="utf-8",
    )

    logger.info(f"配置快照已保存: {snapshot_path}")

    # =====================================================
    # backend hook
    # =====================================================

    exp_id = None

    try:

        if on_training_start is not None:

            exp_id = on_training_start(
                config.name,
                config.to_json(),
            )

            logger.info(
                f"实验已注册到 backend: exp_id={exp_id}"
            )

    except Exception as e:

        logger.warning(
            f"on_training_start 调用失败: {e}"
        )

    # =====================================================
    # CLI command
    # =====================================================

    cmd = [
        "odp-train",
        *config.to_cli_args(),
    ]

    logger.info("训练命令:")
    logger.info(" ".join(cmd))

    # =====================================================
    # 启动训练
    # =====================================================

    proc = subprocess.run(
        cmd,
        cwd=exp_dir,
        capture_output=True,
        text=True,
    )

    # stdout
    if proc.stdout:
        logger.info(proc.stdout)

    # stderr
    if proc.stderr:
        logger.warning(proc.stderr)

    # =====================================================
    # 训练失败
    # =====================================================

    if proc.returncode != 0:

        logger.error(
            f"训练失败: exit={proc.returncode}"
        )

        raise RuntimeError(
            f"odp-train 返回非零退出码 {proc.returncode}"
        )

    # =====================================================
    # metrics
    # =====================================================

    results_csv = exp_dir / "results.csv"

    metrics = _parse_metrics(results_csv)

    logger.info(f"metrics: {metrics}")

    # =====================================================
    # checkpoint
    # =====================================================

    best_pt = (
        exp_dir
        / "weights"
        / "best.pt"
    )

    checkpoint_name = (
        f"best_{config.name}.pt"
    )

    checkpoint_path = (
        CHECKPOINTS_DIR
        / checkpoint_name
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if best_pt.exists():

        shutil.copy(
            best_pt,
            checkpoint_path,
        )

        logger.info(
            f"checkpoint 已复制:"
            f" {checkpoint_path}"
        )

    else:

        logger.warning(
            f"best.pt 未找到: {best_pt}"
        )

    # =====================================================
    # duration
    # =====================================================

    duration = time.time() - t0

    logger.info(
        f"训练耗时: {duration:.2f} sec"
    )

    # =====================================================
    # backend hook
    # =====================================================

    try:

        if (
            on_training_end is not None
            and exp_id is not None
        ):

            on_training_end(
                exp_id=exp_id,
                map50=metrics.get("map50", 0),
                model_path=str(
                    checkpoint_path.resolve()
                ),
            )

            logger.info(
                "训练结果已同步到 backend"
            )

    except Exception as e:

        logger.warning(
            f"on_training_end 调用失败: {e}"
        )

    # =====================================================
    # result
    # =====================================================

    result = ExperimentResult(
        name=config.name,
        dataset=config.dataset,
        model=config.model,

        imgsz=config.imgsz,

        epochs_run=metrics.get(
            "epoch",
            config.epochs,
        ),

        best_epoch=metrics.get(
            "epoch",
            0,
        ),

        map50=metrics.get(
            "map50",
            0,
        ),

        map50_95=metrics.get(
            "map50_95",
            0,
        ),

        precision=metrics.get(
            "precision",
            0,
        ),

        recall=metrics.get(
            "recall",
            0,
        ),

        train_duration_sec=duration,

        # 必须绝对路径
        model_path=str(
            checkpoint_path.resolve()
        ),

        config_snapshot_path=str(
            snapshot_path.resolve()
        ),
    )

    logger.info("=" * 80)
    logger.info(
        f"实验完成: {config.name} "
        f"mAP50={result.map50:.4f}"
    )
    logger.info("=" * 80)

    return result


# =========================================================
# Batch
# =========================================================

def run_batch(
    configs: list[ExperimentConfig],
) -> list[ExperimentResult]:
    """
    批量实验

    顺序执行：
        config1
          ↓
        config2
          ↓
        config3
    """

    results = []

    logger.info(
        f"开始批量实验: total={len(configs)}"
    )

    for idx, config in enumerate(configs):

        logger.info(
            f"[{idx + 1}/{len(configs)}] "
            f"{config.name}"
        )

        try:

            result = run_experiment(config)

            results.append(result)

        except Exception as e:

            logger.exception(
                f"实验失败: {config.name}"
            )

    logger.info(
        f"批量实验结束:"
        f" success={len(results)}"
    )

    return results


# =========================================================
# Debug
# =========================================================

if __name__ == "__main__":

    cfg = ExperimentConfig(
        name="rsod_yolo11n_debug",
        dataset="rsod",
        model="yolo11n.pt",
        epochs=10,
        batch=4,
        imgsz=640,
        note="debug experiment",
    )

    result = run_experiment(cfg)

    print(result.to_json())